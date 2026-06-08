# Create directory structure
New-Item -ItemType Directory -Force -Path src/refinery, tests, examples

# Create package files
$refineryFiles = @(
    "src/refinery/__init__.py",
    "src/refinery/config.py",
    "src/refinery/models.py",
    "src/refinery/ingest.py",
    "src/refinery/cluster.py",
    "src/refinery/schema.py",
    "src/refinery/extract.py",
    "src/refinery/verify.py",
    "src/refinery/memory.py",
    "src/refinery/pipeline.py",
    "main.py",
    "tests/__init__.py",
    "examples/cluster_only.py"
)

foreach ($file in $refineryFiles) {
    New-Item -ItemType File -Force -Path $file
}

# Create pyproject.toml
@'
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "refinery"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
'@ | Set-Content pyproject.toml

# Create .gitignore
@'
__pycache__/
*.pyc
*.pyo
.env
cache/
output/
logs/
*.egg-info/
dist/
build/
.DS_Store
'@ | Set-Content .gitignore

# Create README
@'
# DocTabularize

Document Intelligence Refinery — unsupervised schema discovery and structured extraction from PDFs using UMAP, HDBSCAN, and vision-language models.

## Setup

```bash
pip install -e .
```

## Usage

```bash
python main.py
```

Configure via `pipeline_config.toml`.
'@ | Set-Content README.md

# Back up existing monolithic file
New-Item -ItemType Directory -Force -Path legacy
Copy-Item -Path "*.py" -Destination legacy/ -ErrorAction SilentlyContinue

Write-Host "Done. Run: pip install -e ."