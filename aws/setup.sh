#!/usr/bin/env bash
# ============================================================
# Prometheus AWS One-Time Setup
#
# Run this ONCE from your local machine after cloning the repo.
# Prerequisites: aws CLI installed and configured (aws configure).
# ============================================================
set -euo pipefail

BUCKET="${1:?Usage: ./aws/setup.sh <bucket-name> [region] [account-id]}"
REGION="${2:-us-east-1}"
ACCOUNT_ID="${3:-$(aws sts get-caller-identity --query Account --output text)}"

echo "=== Prometheus AWS Setup ==="
echo "  Bucket:    $BUCKET"
echo "  Region:    $REGION"
echo "  AccountID: $ACCOUNT_ID"
echo ""

# ── 1. Create S3 bucket ────────────────────────────────────────────────
echo "[1/5] Creating S3 bucket: $BUCKET"
if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" 2>/dev/null || true
else
    aws s3api create-bucket \
        --bucket "$BUCKET" \
        --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION" 2>/dev/null || true
fi

# Versioning + encryption
aws s3api put-bucket-versioning \
    --bucket "$BUCKET" \
    --versioning-configuration Status=Enabled

aws s3api put-bucket-encryption \
    --bucket "$BUCKET" \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

echo "    ✓ Bucket ready: s3://$BUCKET"

# ── 2. Create IAM role for EC2 instances ───────────────────────────────
echo "[2/5] Creating EC2 IAM role: PrometheusEC2Role"

# Substitute bucket and account in policy
sed "s/YOUR-BUCKET-NAME/$BUCKET/g; s/YOUR-ACCOUNT-ID/$ACCOUNT_ID/g" \
    aws/iam_policy.json > /tmp/prometheus_policy.json

aws iam create-role \
    --role-name PrometheusEC2Role \
    --assume-role-policy-document file://aws/ec2_instance_role_trust.json \
    --description "Prometheus training instance role" 2>/dev/null || \
    echo "    (role already exists)"

aws iam put-role-policy \
    --role-name PrometheusEC2Role \
    --policy-name PrometheusS3EC2Access \
    --policy-document file:///tmp/prometheus_policy.json

aws iam create-instance-profile \
    --instance-profile-name PrometheusEC2Role 2>/dev/null || \
    echo "    (instance profile already exists)"

aws iam add-role-to-instance-profile \
    --instance-profile-name PrometheusEC2Role \
    --role-name PrometheusEC2Role 2>/dev/null || true

echo "    ✓ IAM role ready: PrometheusEC2Role"

# ── 3. Create IAM user for local machine ───────────────────────────────
echo "[3/5] Creating IAM user: prometheus-local"
aws iam create-user --user-name prometheus-local 2>/dev/null || \
    echo "    (user already exists)"

sed "s/YOUR-BUCKET-NAME/$BUCKET/g; s/YOUR-ACCOUNT-ID/$ACCOUNT_ID/g" \
    aws/iam_policy.json > /tmp/prometheus_policy.json

aws iam put-user-policy \
    --user-name prometheus-local \
    --policy-name PrometheusS3EC2Access \
    --policy-document file:///tmp/prometheus_policy.json

echo "    Creating access key..."
KEY_JSON=$(aws iam create-access-key --user-name prometheus-local)
ACCESS_KEY=$(echo "$KEY_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['AccessKey']['AccessKeyId'])")
SECRET_KEY=$(echo "$KEY_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['AccessKey']['SecretAccessKey'])")

cat > .env.aws << EOF
AWS_ACCESS_KEY_ID=$ACCESS_KEY
AWS_SECRET_ACCESS_KEY=$SECRET_KEY
AWS_DEFAULT_REGION=$REGION
PROMETHEUS_S3_BUCKET=$BUCKET
EOF
chmod 600 .env.aws
echo "    ✓ Credentials saved to: .env.aws  (keep this file secret!)"

# ── 4. Create security group ───────────────────────────────────────────
echo "[4/5] Creating security group: prometheus-training"
SG_ID=$(aws ec2 create-security-group \
    --group-name prometheus-training \
    --description "Prometheus EC2 training instances" \
    --region "$REGION" \
    --query GroupId --output text 2>/dev/null) || \
    SG_ID=$(aws ec2 describe-security-groups \
        --group-names prometheus-training \
        --region "$REGION" \
        --query 'SecurityGroups[0].GroupId' --output text)

# Allow outbound only (no inbound needed — instance self-terminates)
aws ec2 revoke-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol all --port all --cidr 0.0.0.0/0 \
    --region "$REGION" 2>/dev/null || true

echo "    ✓ Security group: $SG_ID"

# ── 5. Write local config ──────────────────────────────────────────────
echo "[5/5] Writing aws/config.sh"
cat > aws/config.sh << EOF
# Source this file before running aws_train.py
export AWS_DEFAULT_REGION=$REGION
export PROMETHEUS_S3_BUCKET=$BUCKET
export PROMETHEUS_SG=$SG_ID
export PROMETHEUS_IAM_PROFILE=PrometheusEC2Role
# Load credentials:
# source .env.aws
EOF
echo ""
echo "=== Setup Complete ==="
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Install local dependencies:"
echo "   pip install boto3 awscli"
echo ""
echo "2. Load AWS credentials:"
echo "   source .env.aws"
echo ""
echo "3. Estimate cost before launching:"
echo "   python scripts/aws_train.py cost --instance g4dn.xlarge --hours 8"
echo ""
echo "4. Launch training:"
echo "   python scripts/aws_train.py launch \\"
echo "     --bucket $BUCKET \\"
echo "     --repo https://github.com/YOUR_USERNAME/Prometheus \\"
echo "     --instance g4dn.xlarge \\"
echo "     --mode full \\"
echo "     --security-group $SG_ID \\"
echo "     --iam-profile PrometheusEC2Role"
echo ""
echo "5. Stream logs while it trains:"
echo "   python scripts/aws_train.py wait --bucket $BUCKET --job <job-id>"
echo ""
echo "6. Pull checkpoint when done:"
echo "   python scripts/aws_train.py pull --bucket $BUCKET --tag latest"
echo ""
echo "7. Run local analysis:"
echo "   python scripts/analyze.py --checkpoint checkpoints --symbols SPY QQQ GLD --volcano"
