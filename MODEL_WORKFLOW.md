# Model Training and Export Workflow

This document describes the steps to train models and generate the `.pkl` files required by the MediGuide FastAPI backend.

## Prerequisites

- Python 3.7+
- Required packages (install with `pip install -r requirements.txt`)
- Prepared dataset: `improved_disease_dataset.csv` in the project root

## Steps

### 1. Place Your Dataset

Ensure `improved_disease_dataset.csv` is present in the project root. The file should include columns for symptoms, `disease`, and `medication`.

### 2. Run the Automation Script

Run the provided script to train all models and save the necessary `.pkl` files:

```sh
python generate_and_save_models.py
```

### 3. Output

The script will create (or update) the following files in the `models/` directory:

- `rf_model.pkl` — Random Forest model
- `nb_model.pkl` — Naive Bayes model
- `svm_model.pkl` — SVM model
- `label_encoder.pkl` — Encodes disease names to numeric classes
- `symptoms_list.pkl` — List of symptom columns (features)
- `disease_to_medication.pkl` — Mapping from encoded diseases to medications

These files are required for your backend’s prediction and response logic.

### 4. Automation (Optional)

You can add automation to retrain models whenever the dataset changes by including the following in your CI/CD (e.g., GitHub Actions):

```yaml
# .github/workflows/train-models.yml
name: Train and Export Models

on:
  push:
    paths:
      - "improved_disease_dataset.csv"
      - "generate_and_save_models.py"
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run model training script
        run: python generate_and_save_models.py
      - name: Commit model artifacts
        run: |
          git config --global user.name 'github-actions'
          git config --global user.email 'github-actions@github.com'
          git add models/
          git commit -m 'Update trained models and artifacts' || echo "No changes to commit"
          git push
```

### 5. Troubleshooting

- **Missing packages**: Install dependencies with `pip install -r requirements.txt`.
- **File not found**: Ensure `improved_disease_dataset.csv` exists.
- **Permission errors**: Ensure you have write access to the repo and working directory.

---

For more details on the script, see [generate_and_save_models.py](./generate_and_save_models.py).