# Multi-Label Text Classification System

This repository contains code and resources for a multi-label text classification system.

Important folders
- `src/` - source code for the project
- `notebooks/` - exploratory notebooks and experiments

Ignored folders
- `models/` - trained model artifacts are excluded from version control and listed in `.gitignore`.
- `data/raw/` - raw datasets are large and kept outside the repository; `data/` is in `.gitignore`.

Datasets

Download the datasets and place them under `data/raw/` (created locally, not committed):

- dataset_1: https://drive.google.com/file/d/1XkDyylgWw_tOaHU2eRpyFnG_8-eJYXj9/view?usp=drive_link
- dataset_1_val: https://drive.google.com/file/d/1OInznXIqoga09PUK4LCbnLlZr8g7ubna/view?usp=drive_link
- dataset_1_train: https://drive.google.com/file/d/1_XaoLfgX23qoX5RSBQqL0Qc9TfTyqrC2/view?usp=drive_link

How to use

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
```

## Setup

To install the required Python packages run:

```
pip install -r requirements.txt
```

## Run

Run the main script with:

```bash
python -m src.main
```

2. Download the datasets from the links above and place the files in `data/raw/`.

3. Run notebooks in `notebooks/` or scripts in `src/` to train/evaluate models.

License & notes

Add license and contribution guidelines as needed.
