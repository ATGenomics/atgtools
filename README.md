# ATGtools

[![Supported Python Versions](https://img.shields.io/pypi/pyversions/atgtools)](https://pypi.org/project/atgtools/)
[![PyPI version](https://badge.fury.io/py/atgtools.svg)](https://badge.fury.io/py/atgtools)

This is a collection of tools for NGS data analysis. It provides many easy-to-use functionalities to clean, visualize, and analyze NGS results.

## Installation

Latest `atgtools` version: [![PyPI version](https://badge.fury.io/py/atgtools.svg)](https://badge.fury.io/py/atgtools)

### Recommended Installation Methods

#### Using pipx (Recommended)

[pipx](https://pypa.github.io/pipx/) installs Python applications in isolated environments:

```bash
pipx install atgtools
```

#### Using uv (Alternative)

```bash
uv pip install atgtools
```

### Development Setup

For development, clone the repository and set up the environment using uv:

```bash
# Clone the repository
git clone https://github.com/yourusername/atgtools.git
cd atgtools

# Create and activate virtual environment with Python 3.12
uv venv --python 3.12
source .venv/bin/activate  # On Unix/macOS

# Install dependencies
uv sync

# Use as a module
python -m atg --help
```

## Features

### Diversity Analysis

The package provides tools for both alpha and beta diversity analysis.

#### Alpha Diversity

Calculate various alpha diversity indices:

* Shannon (natural log and custom base)
* Gini-Simpson
* Simpson
* Dominance
* Richness
* Pielou's Evenness

```bash
atg div alpha --input abundance.tsv
```

#### Beta Diversity

Calculate dissimilarity between samples using various metrics:

* Bray-Curtis
* Jensen-Shannon
* Jaccard
* Euclidean
* UniFrac (weighted and unweighted)

```bash
atg div beta --input abundance.tsv --metric braycurtis
```

### Utility Tools

#### Git Repository Management

Check multiple git repositories in one pass:

```bash
atg tools git [options]
```

Options:

* `--verbose/-v`: Show files & commits
* `--remote/-r`: Force remote update
* `--untracked/-u`: Show untracked files
* `--dir/-d`: Search specific directory for repositories
* `--all-branch/-a`: Show status of all branches
* `--stash/-s`: Show stashed changes

#### Manifest Creation

Create manifest files for QIIME2:

```bash
atg tools manifest --fastq_dir /path/to/fastq --output manifest.tsv
```

Supported FASTQ naming schemes:

* `sample_R1.fastq.gz`
* `sample_R2.fastq.gz`
* `sample_S01_R1.fastq.gz`
* `sample_S01_R2.fastq.gz`

#### Other Tools

* `atg tools oneliner`: Convert multiline FASTA to single line
* `atg tools countfq`: Count reads in FASTQ files
* `atg tools abundance`: Generate relative abundance tables
* `atg tools download`: Download ENA data
* `atg tools retrieve`: Search and retrieve ENA data
* `atg tools search`: Search ENA database
