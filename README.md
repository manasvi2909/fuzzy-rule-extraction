# Opening the Black Box: Fuzzy Rule Extraction

Extracting fuzzy rules from a trained neural network for interpretable
classification.

## Objective

Build an end-to-end pipeline that trains an MLP teacher model, extracts
a fuzzy rule base from its predictions, evaluates a Mamdani-style fuzzy
surrogate, and measures the effect of rule pruning on predictive
performance, model fidelity, and rule-base complexity.

## Pipeline

Dataset and preprocessing
→ MLP teacher
→ Candidate rule generation
→ Conflict resolution
→ Fuzzy inference
→ Rule pruning
→ Evaluation and plots

## Repository Structure

- `data/` — dataset documentation
- `src/` — pipeline and model components
- `tests/` — automated tests
- `artifacts/` — model metadata and prediction artifacts
- `results/` — metrics and figures
- `report/` — report materials
- `presentation/` — presentation materials
- `config.yaml` — experiment configuration
- `requirements.txt` — Python dependencies

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running the Project

The main experiment command will be documented here once the pipeline
has been integrated.

## Evaluation

The project will report:

- Accuracy and macro-F1 against true labels
- Fidelity to the MLP teacher
- Rule count and mean rule length
- Rule coverage and fallback rate
- Performance and fidelity versus rule-base size

## Reproducibility

Dataset version, random seeds, preprocessing, feature order, class
mapping, and pruning settings will be documented in the final pipeline.
