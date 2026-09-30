# Knee Osteoarthritis Severity Classification

A comparative study of custom and pretrained convolutional neural networks (CNNs) for automated knee osteoarthritis severity classification.

This project is part of the **Data Science in Health Care** course.

## Research Question

How do pretrained convolutional neural networks compare with a custom CNN trained from scratch in classifying knee osteoarthritis severity, as measured by accuracy and AUROC?

## Models

- Custom CNN trained from scratch
- ResNet-18 with transfer learning
- EfficientNet-B0 with transfer learning (optional)

All models will be evaluated using the same dataset splits and evaluation metrics.

## Project Structure

```text
.
├── notebooks/
│   ├── 00_shared_pipeline.ipynb
│   ├── 01_custom_cnn.ipynb
│   ├── 02_transfer_learning.ipynb
│   └── 03_model_evaluation.ipynb
├── src/
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   └── models.py
├── results/
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

The structure may change as the project develops.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
```

### 2. Install uv

Install uv following the official instructions:

https://docs.astral.sh/uv/getting-started/installation/

### 3. Install Dependencies

The project uses **uv** for Python dependency management.

```bash
uv sync
```

This creates a virtual environment and installs the dependencies specified in `pyproject.toml` and `uv.lock`.

To run a Python script:

```bash
uv run python your_script.py
```

To add a dependency:

```bash
uv add package-name
```

Commit changes to `pyproject.toml` and `uv.lock` so everyone uses the same dependencies.

## Working with Google Colab

Google Colab can be used to run experiments without requiring a powerful local computer.

1. Create or open your experiment notebook in Colab.
2. Enable GPU acceleration if available.
3. Clone the repository and install the project dependencies.

Run the following commands in a Colab notebook:

```python
!git clone https://github.com/michael-duer/medical-image-classification.git
%cd medical-image-classification

# Install uv
!pip install uv

# Install project dependencies
!uv sync
```

Run scripts using:

```python
!uv run python your_script.py
```

To import project modules directly into the Colab notebook, ensure that the notebook's Python environment contains the required dependencies. The environment created by `uv sync` is separate from Colab's notebook kernel.

## Dataset

The project uses a knee osteoarthritis dataset containing X-ray images with severity labels.

Dataset source: TODO

The dataset is stored separately from GitHub.

All experiments must use the same training, validation and test splits.

## Collaboration

Each team member works on their own Git branch:

- `custom-cnn`: Custom CNN implementation
- `transfer-learning`: Pretrained models
- `evaluation`: Dataset preparation and evaluation

### Workflow

1. Pull the latest changes from `main`.
2. Switch to your personal branch.
3. Implement and test your changes.
4. Commit and push your work.
5. Create a pull request for review before merging into `main`.

Avoid editing another member's experiment notebook without coordinating first.

Reusable functions should be placed in `src/` rather than duplicated across notebooks.

Do not commit datasets, model checkpoints, credentials or large generated files.

## Evaluation

The models will be compared using:

- Classification accuracy
- Multiclass AUROC
- Confusion matrices

Additional metrics may be included if appropriate.

## Team

- Person 1: Custom CNN
- Person 2: Transfer learning
- Person 3: Dataset preparation and evaluation

Research planning, interpretation and presentation preparation are shared responsibilities.
