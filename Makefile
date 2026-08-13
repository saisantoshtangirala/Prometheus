# Prometheus Makefile — common commands for local + AWS workflow
# Run `make help` to see all targets.

BUCKET     ?= my-prometheus-bucket
REGION     ?= us-east-1
REPO       ?= https://github.com/YOUR_USERNAME/Prometheus
INSTANCE   ?= g4dn.xlarge
SYMBOLS    ?= SPY QQQ GLD TLT AAPL NVDA BTC-USD
DEVICE     ?= mps
TAG        ?= latest
JOB        ?=

.PHONY: help setup-local setup-aws train wait pull analyze volcano test clean

help:
	@echo ""
	@echo "Prometheus – Local + AWS Workflow"
	@echo "=================================="
	@echo ""
	@echo "LOCAL (Mac mini)"
	@echo "  make setup-local         Install local Python dependencies"
	@echo "  make analyze             Run God's Eye analysis (uses MPS)"
	@echo "  make volcano             Analysis + 4D Probability Volcano HTML"
	@echo "  make test                Run unit tests"
	@echo ""
	@echo "AWS"
	@echo "  make setup-aws           One-time AWS infrastructure setup"
	@echo "  make train               Launch full training on EC2 Spot"
	@echo "  make train-quick         Launch quick test run (pretrain only)"
	@echo "  make wait JOB=job-...    Stream logs for a running job"
	@echo "  make pull                Pull latest checkpoint from S3"
	@echo "  make pull TAG=job-...    Pull specific job checkpoint"
	@echo "  make ls                  List all S3 checkpoints"
	@echo "  make cost                Estimate training cost"
	@echo ""
	@echo "VARIABLES (override on CLI)"
	@echo "  BUCKET=$(BUCKET)"
	@echo "  INSTANCE=$(INSTANCE)  (g4dn.xlarge | g5.2xlarge | p3.2xlarge | p3.8xlarge)"
	@echo "  SYMBOLS=$(SYMBOLS)"
	@echo "  DEVICE=$(DEVICE)     (mps | cpu | cuda)"
	@echo "  TAG=$(TAG)"
	@echo ""

# ── Local ─────────────────────────────────────────────────────────────────

setup-local:
	pip install torch torchvision torchaudio
	pip install -r requirements-local.txt
	@echo ""
	@echo "✓ Local setup complete. Test with: make analyze"

analyze:
	python scripts/analyze.py \
		--symbols $(SYMBOLS) \
		--checkpoint checkpoints \
		--device $(DEVICE) \
		--save-report output/gods_eye_report.json

volcano:
	python scripts/analyze.py \
		--symbols $(SYMBOLS) \
		--checkpoint checkpoints \
		--device $(DEVICE) \
		--volcano \
		--save-html output/volcano.html \
		--save-report output/gods_eye_report.json
	@echo "Volcano saved to output/volcano.html"

test:
	pytest tests/ -v --tb=short

# ── AWS ───────────────────────────────────────────────────────────────────

setup-aws:
	@if [ -z "$(BUCKET)" ] || [ "$(BUCKET)" = "my-prometheus-bucket" ]; then \
		echo "Set BUCKET: make setup-aws BUCKET=your-unique-bucket-name"; \
		exit 1; \
	fi
	bash aws/setup.sh $(BUCKET) $(REGION)

train:
	@if [ -z "$(BUCKET)" ] || [ "$(BUCKET)" = "my-prometheus-bucket" ]; then \
		echo "Set BUCKET: make train BUCKET=your-bucket REPO=https://github.com/you/Prometheus"; \
		exit 1; \
	fi
	python scripts/aws_train.py \
		--bucket $(BUCKET) \
		--region $(REGION) \
		launch \
		--repo $(REPO) \
		--instance $(INSTANCE) \
		--mode full

train-quick:
	@if [ -z "$(BUCKET)" ] || [ "$(BUCKET)" = "my-prometheus-bucket" ]; then \
		echo "Set BUCKET: make train-quick BUCKET=your-bucket REPO=https://github.com/you/Prometheus"; \
		exit 1; \
	fi
	python scripts/aws_train.py \
		--bucket $(BUCKET) \
		--region $(REGION) \
		launch \
		--repo $(REPO) \
		--instance $(INSTANCE) \
		--mode pretrain \
		--n-black-swans 500 \
		--pretrain-epochs 3

wait:
	@if [ -z "$(JOB)" ]; then echo "Specify JOB: make wait JOB=job-20260813T120000Z"; exit 1; fi
	python scripts/aws_train.py \
		--bucket $(BUCKET) \
		--region $(REGION) \
		wait --job $(JOB)

pull:
	python scripts/aws_train.py \
		--bucket $(BUCKET) \
		--region $(REGION) \
		pull --tag $(TAG) --local-dir checkpoints

ls:
	python scripts/aws_train.py \
		--bucket $(BUCKET) \
		--region $(REGION) \
		ls

cost:
	python scripts/aws_train.py \
		--region $(REGION) \
		cost --instance $(INSTANCE) --hours 8

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf output/*.html output/*.json logs/ 2>/dev/null || true
	@echo "✓ Cleaned"
