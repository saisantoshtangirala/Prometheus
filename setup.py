from setuptools import setup, find_packages

setup(
    name="prometheus-causal-brain",
    version="1.0.0",
    description="Prometheus: A causal AI market intelligence system",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.1",
        "pandas>=2.0.0",
        "plotly>=5.17.0",
        "loguru>=0.7.0",
        "rich>=13.6.0",
        "pydantic>=2.4.0",
        "requests>=2.31.0",
        "yfinance>=0.2.31",
        "pyyaml>=6.0",
    ],
    extras_require={
        "full": [
            "dowhy>=0.11.0",
            "diffusers>=0.24.0",
            "transformers>=4.35.0",
            "torch-geometric>=2.4.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "hypothesis>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "prometheus-train=scripts.train:main",
            "prometheus-analyze=scripts.analyze:main",
        ],
    },
)
