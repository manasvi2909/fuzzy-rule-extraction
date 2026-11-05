# Opening the Black Box: Extracting Fuzzy Rules from Trained Neural Networks

### Interpretable Machine Learning Through Fuzzy Rule Extraction

An experimental framework for transforming the predictions of a trained Multi-Layer Perceptron (MLP) into an interpretable fuzzy rule-based classification system.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Research Approach](#research-approach)
5. [Dataset](#dataset)
6. [System Architecture](#system-architecture)
7. [Methodology](#methodology)
   - [Data Preprocessing](#1-data-preprocessing)
   - [Training the Neural Network](#2-training-the-mlp-teacher-model)
   - [Fuzzy Membership Functions](#3-fuzzy-membership-functions)
   - [Fuzzy Rule Extraction](#4-fuzzy-rule-extraction)
   - [Conflict Resolution](#5-conflict-resolution)
   - [Fuzzy Inference](#6-fuzzy-inference)
   - [Rule Ranking and Pruning](#7-rule-ranking-and-pruning)
   - [Rule Ablation](#8-rule-ablation)
8. [Experimental Design](#experimental-design)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Results and Analysis](#results-and-analysis)
11. [Interpretability and Explainability](#interpretability-and-explainability)
12. [Repository Structure](#repository-structure)
13. [Installation and Setup](#installation-and-setup)
14. [Running the Project](#running-the-project)
15. [Testing](#testing)
16. [Reproducibility](#reproducibility)
17. [Technologies Used](#technologies-used)
18. [Limitations](#limitations)
19. [Future Improvements](#future-improvements)
20. [Disclaimer](#disclaimer)

---

## Project Overview

Modern machine learning models can achieve excellent predictive performance while remaining difficult to interpret. Neural networks, particularly Multi-Layer Perceptrons, learn complex nonlinear relationships between input features and target classes. However, understanding the reasoning behind an individual prediction or the overall decision-making process is not straightforward.

This project investigates an approach to improving model interpretability by extracting a fuzzy rule-based surrogate from a trained neural network.

The central idea is to use the neural network as a **teacher model** and construct a fuzzy inference system that approximates its classification behaviour through human-readable IF–THEN rules.

Rather than attempting to inspect the internal weights and hidden-layer activations of the neural network directly, the project uses its predictions to generate a collection of fuzzy rules. These rules are subsequently processed, ranked, pruned, and evaluated to produce a smaller rule base.

The resulting system is designed to balance three objectives:

- **Predictive performance:** How accurately does the fuzzy model classify unseen observations?
- **Model fidelity:** How closely does the fuzzy model reproduce the predictions of the original neural network?
- **Interpretability:** How manageable and understandable is the extracted rule base?

The project implements an end-to-end experimental framework covering neural network training, fuzzy representation, rule extraction, conflict resolution, fuzzy inference, pruning, ablation, and quantitative evaluation.

### Final outcome

The final fuzzy surrogate contains **99 rules**, evaluated on a held-out test set of 86 observations.

| Evaluation metric | Final fuzzy surrogate |
|---|---:|
| Test accuracy | 88.37% |
| Test macro-F1 | 86.99% |
| Fidelity to MLP | 87.21% |
| Coverage | 100% |
| Fallback rate | 0% |

The original MLP teacher achieves 98.84% test accuracy. The difference illustrates the performance cost associated with approximating a more complex model using a finite fuzzy rule base.

---

## Problem Statement

Neural networks are frequently treated as black-box predictive systems because their learned representations are distributed across multiple interconnected layers.

Although these models can capture complex patterns, their predictions are not naturally expressed in terms of simple human-readable decision rules.

This creates an interpretability challenge:

**Can the classification behaviour of a trained neural network be approximated using a compact fuzzy rule-based system without losing an excessive amount of predictive performance?**

This project approaches the problem through fuzzy rule extraction.

Given a trained neural network \(f_{\theta}\), the objective is to construct a fuzzy classifier \(g_{\phi}\) whose predictions approximate those of the teacher:

\[
g_{\phi}(x) \approx f_{\theta}(x)
\]

Here:

- \(f_{\theta}\) represents the trained MLP teacher.
- \(g_{\phi}\) represents the extracted fuzzy surrogate.
- \(x\) represents an input observation.
- \(\theta\) and \(\phi\) represent the respective model parameters and rule-base configuration.

The objective is not to reproduce the internal architecture of the neural network. Instead, it is to approximate its observable classification behaviour through a collection of interpretable fuzzy rules.

The quality of the approximation is assessed through both predictive performance against the actual dataset labels and fidelity against the teacher's predictions.

---

## Objectives

The project is designed around the following objectives:

1. Train a sufficiently accurate Multi-Layer Perceptron on a structured classification dataset.
2. Represent numerical input features using fuzzy membership functions.
3. Generate candidate fuzzy rules using the predictions of the trained neural network.
4. Resolve conflicting rules that share identical antecedents.
5. Implement a fuzzy inference mechanism using fuzzy conjunction and aggregation.
6. Rank and prune extracted rules to reduce rule-base complexity.
7. Investigate the effect of rule removal through ablation experiments.
8. Compare the fuzzy surrogate against the original neural network.
9. Quantify the trade-off between predictive accuracy, model fidelity, coverage, and interpretability.
10. Maintain reproducible experiments through configuration files, saved artifacts, metadata, and automated tests.

---

## Research Approach

The project follows a teacher–surrogate approach.

The neural network is trained using the original dataset labels. Once trained, its predictions are used as the source of the fuzzy rule consequents.

The fuzzy system is therefore trained to approximate the teacher's behaviour rather than independently learning an entirely separate decision boundary from the original labels.

This distinction is important:

- **Ground-truth labels** are used to assess classification performance.
- **MLP predictions** are used to construct and assess the surrogate's imitation of the teacher.

The overall workflow is:

```text
                 Dataset
                    |
                    v
           Data Preprocessing
                    |
                    v
             Train / Validation / Test
                    |
                    v
              MLP Teacher
                    |
                    v
          MLP Predictions
                    |
                    v
       Fuzzy Membership Functions
                    |
                    v
         Candidate Rule Generation
                    |
                    v
            Conflict Resolution
                    |
                    v
           Initial Rule Base
                    |
                    v
          Rule Ranking and Pruning
                    |
                    v
             Rule Ablation
                    |
                    v
          Final Fuzzy Rule Base
                    |
                    v
             Fuzzy Inference
                    |
                    v
          Model Evaluation
                    |
                    v
       Performance and Fidelity Analysis
```

---

## Dataset

### UCI Breast Cancer Wisconsin (Diagnostic)

The project uses the **Breast Cancer Wisconsin (Diagnostic) dataset**, obtained from the UCI Machine Learning Repository.

The dataset contains numerical measurements computed from digitized images of fine needle aspirate samples of breast masses.

Each observation is represented using 30 numerical features describing characteristics of cell nuclei.

The classification task is binary:

- **Benign (0):** Non-malignant observation.
- **Malignant (1):** Malignant observation.

### Dataset characteristics

| Property | Value |
|---|---|
| Dataset | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository |
| Total observations | 569 |
| Input features | 30 |
| Target classes | 2 |
| Feature type | Numerical |
| Classification type | Binary |

### Feature categories

The 30 numerical attributes are organised into three groups:

- Mean measurements
- Standard error measurements
- Worst-case measurements

These describe characteristics such as:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

Each category contains measurements derived from the cell-nucleus characteristics.

### Dataset split

The dataset is partitioned using stratified sampling to preserve the class distribution across the splits.

| Split | Percentage | Observations |
|---|---:|---:|
| Training | 70% | 398 |
| Validation | 15% | 85 |
| Test | 15% | 86 |
| **Total** | **100%** | **569** |

The random seed is fixed at 42.

The three splits serve different purposes:

- **Training set:** Used to fit preprocessing transformations, train the MLP, and establish the fuzzy membership functions.
- **Validation set:** Used for rule-pruning experiments, model selection, and ablation decisions.
- **Test set:** Reserved for the final, independent evaluation of the selected fuzzy surrogate.

The test set is not used to select the final rule configuration.

---

## System Architecture

The implementation is organised into modular components, each responsible for a distinct stage of the pipeline.

| Component | Responsibility |
|---|---|
| Data preprocessing | Imputation, standardisation, and dataset preparation |
| MLP teacher | Learns the original classification task |
| Membership functions | Converts numerical inputs into fuzzy membership degrees |
| Rule extractor | Generates candidate fuzzy rules from teacher predictions |
| Conflict resolver | Handles identical antecedents with different consequents |
| Fuzzy inference engine | Evaluates rules and generates class predictions |
| Rule pruner | Ranks and selects a smaller rule base |
| Evaluation module | Measures accuracy, fidelity, coverage, and complexity |
| Analysis utilities | Investigates pruning, ablation, feature importance, and inference behaviour |
| Reproducibility module | Records experiment configuration and input-file hashes |

The modular architecture separates model training, fuzzy rule generation, inference, and evaluation. This makes it possible to investigate individual components without changing the entire pipeline.

---

## Methodology

### 1. Data Preprocessing

Preprocessing is performed before training the teacher model.

The pipeline uses:

**Median imputation**

Missing numerical values, where present, are handled using median imputation.

**Standardisation**

Numerical features are standardised using `StandardScaler`.

For a feature \(x\), standardisation is performed as:

\[
z = \frac{x-\mu}{\sigma}
\]

where:

- \(x\) is the original feature value.
- \(\mu\) is the training-set mean.
- \(\sigma\) is the training-set standard deviation.
- \(z\) is the standardised feature value.

Both the imputer and scaler are fitted exclusively on the training set.

The fitted preprocessing pipeline is subsequently applied to the validation and test sets without refitting.

This prevents information from the validation and test sets from leaking into the training process.

The fitted preprocessor is saved as:

`artifacts/preprocessor.joblib`

### 2. Training the MLP Teacher Model

A Multi-Layer Perceptron classifier is used as the teacher model.

The selected architecture contains two hidden layers:

\[
30 \rightarrow 16 \rightarrow 8 \rightarrow 1
\]

The final output dimensionality corresponds to the binary classification model's single output unit in scikit-learn's binary classification implementation.

### Model configuration

| Parameter | Configuration |
|---|---|
| Model | `MLPClassifier` |
| Hidden layers | (16, 8) |
| Activation | ReLU |
| Solver | Adam |
| L2 regularisation (`alpha`) | 0.001 |
| Initial learning rate | 0.001 |
| Maximum iterations | 500 |
| Random state | 42 |

The neural network learns nonlinear relationships between the 30 input features and the binary target.

Once trained, it generates predictions and class probabilities for the training, validation, and test observations.

The predictions are saved separately so that subsequent fuzzy extraction and evaluation experiments can use a consistent teacher output.

The trained model is stored at:

`artifacts/mlp_model.joblib`

### 3. Fuzzy Membership Functions

The fuzzy representation converts numerical feature values into linguistic membership terms.

Instead of assigning each feature a single crisp category, fuzzy membership functions assign degrees of membership between 0 and 1.

For every numerical feature, the project defines three triangular membership terms:

| Membership term | Interpretation |
|---|---|
| Low | Lower region of the feature's range |
| Medium | Intermediate region |
| High | Upper region |

A triangular membership function can be represented as:

\[
\mu(x;a,b,c)=
\max\left(
\min\left(
\frac{x-a}{b-a},
\frac{c-x}{c-b}
\right),0
\right)
\]

where:

- \(a\) represents the left endpoint.
- \(b\) represents the peak.
- \(c\) represents the right endpoint.
- \(x\) represents the input value.
- \(\mu(x)\) represents the membership degree.

The membership degree lies in the interval:

\[
0 \leq \mu(x) \leq 1
\]

A value close to 1 indicates strong membership in the corresponding fuzzy set, while a value close to 0 indicates weak membership.

Membership functions are fitted using the training data only.

The implementation stores the fitted membership-function definitions and uses them to fuzzify observations consistently during extraction and inference.

Saved artifact:

`artifacts/membership_functions.joblib`

### 4. Fuzzy Rule Extraction

The rule extraction module converts observations and MLP predictions into candidate fuzzy rules.

Each candidate rule contains:

- An antecedent consisting of fuzzy feature conditions.
- A consequent representing the predicted class.
- A rule weight.
- A support value.

A rule follows the general structure:

```text
IF feature_1 is Medium
AND feature_2 is Low
AND feature_3 is High
...
AND feature_30 is Medium
THEN class is Malignant
```

For each feature, the strongest membership term is selected:

\[
A_j = \arg\max_{t \in \{Low,Medium,High\}}
\mu_{j,t}(x_j)
\]

where \(A_j\) is the selected fuzzy term for feature \(j\).

The selected terms form the rule antecedent.

The consequent is obtained from the MLP teacher's predicted class:

\[
C = f_{\theta}(x)
\]

Thus, each observation contributes a candidate rule describing the fuzzy representation of its input and the teacher's predicted output.

The extracted rules contain all 30 feature conditions in their antecedents.

The initial extraction process produces **290 candidate rules**.

### 5. Conflict Resolution

Different observations may produce identical fuzzy antecedents but different predicted consequents.

For example:

```text
Rule A:
IF feature_1 is High
AND feature_2 is Medium
THEN class is Benign

Rule B:
IF feature_1 is High
AND feature_2 is Medium
THEN class is Malignant
```

Such conflicts must be resolved before constructing the final rule base.

The implementation groups rules with identical antecedents and retains the consequent with the highest aggregate rule weight.

This reduces contradictory rules while preserving the strongest aggregate evidence associated with each antecedent.

Rule weights and support are retained as metadata and are subsequently useful for rule ranking and pruning.

### 6. Fuzzy Inference

The fuzzy inference engine evaluates the extracted rules against a new observation.

The implementation uses a Mamdani-inspired fuzzy rule classifier: minimum t-norm for antecedent evaluation, maximum class-wise aggregation, and winner-take-all classification. It does not implement conventional Mamdani output-set aggregation or centroid defuzzification.

The inference process consists of the following stages.

#### A. Fuzzification

Each numerical input is evaluated against its corresponding membership functions.

This produces membership degrees for the Low, Medium, and High fuzzy sets.

#### B. Antecedent evaluation: Minimum t-norm

The conjunction of antecedent conditions is evaluated using the minimum t-norm.

For a rule with \(n\) antecedents:

\[
\alpha_r =
\min\left(
\mu_{1}(x_1),
\mu_{2}(x_2),
\ldots,
\mu_{n}(x_n)
\right)
\]

Here, \(\alpha_r\) represents the rule's firing strength.

The minimum operator ensures that the conjunction is determined by the weakest membership degree among the antecedents.

#### C. Class-wise maximum aggregation

The strongest firing rule for each class is retained using maximum aggregation.

For class \(c\):

\[
S_c = \max_{r:C_r=c}\alpha_r
\]

where:

- \(S_c\) is the aggregated activation for class \(c\).
- \(C_r\) is the consequent of rule \(r\).
- \(\alpha_r\) is the firing strength of rule \(r\).

#### D. Winner-take-all classification

The final class is selected using:

\[
\hat{y} = \arg\max_c S_c
\]

The class with the highest aggregated activation becomes the predicted class.

A deterministic class-order rule is used to resolve ties.

If no rule matches an observation, the inference engine uses its configured fallback mechanism and records the event for coverage and fallback evaluation.

**Implementation note:** Rule weights and support are retained for extraction metadata, analysis, ranking, and pruning. They do not multiply or otherwise scale the firing strength during inference. Also, because the consequents are categorical class labels, the implementation uses class-wise activation scores and winner-take-all classification rather than centroid defuzzification over a numerical output universe.

### 7. Rule Ranking and Pruning

The initial rule base contains 290 candidate rules.

A large rule base can become difficult to interpret and maintain. Pruning is therefore used to investigate whether a smaller collection of rules can preserve acceptable predictive behaviour.

Rules are ranked using their extraction metadata, including rule weight and support.

Different top-\(k\) configurations are evaluated on the validation set.

The project investigates:

- All extracted rules.
- Top 10 rules.
- Top 25 rules.
- Top 50 rules.
- Top 100 rules.

The purpose is to understand how rule-base size affects:

- Classification accuracy.
- Macro-F1.
- Fidelity to the MLP.
- Coverage.
- Fallback rate.
- Rule-base complexity.

The validation results show that simply increasing the number of rules does not guarantee an improvement in every metric.

The selected configuration is therefore evaluated against explicit validation constraints rather than chosen solely by rule count.

### 8. Rule Ablation

After selecting the 100-rule configuration, an ablation experiment is performed to investigate the contribution of individual rules.

Ablation evaluates the effect of removing a rule and measuring the resulting change in model behaviour.

The investigation identifies the original rule at index 0 as a broad all-Medium benign rule.

Its removal improves validation performance:

| Validation metric | Top 100 rules | After removing rule 0 |
|---|---:|---:|
| Accuracy | 80.00% | 87.06% |
| Macro-F1 | 76.70% | 85.94% |
| Malignant recall | — | 78.12% |
| Coverage | 98.82% | 98.82% |

Removing Rule 0 improves validation accuracy and macro-F1 while validation coverage remains unchanged at 98.82%. The final 99-rule model separately achieves 100% coverage on the held-out test set.

The resulting 99-rule configuration is selected using validation evidence.

The final test set is then used once for independent evaluation of the selected candidate.

---

## Experimental Design

The experiments investigate how different design choices affect the performance and complexity of the fuzzy surrogate.

All model-selection decisions are based on the validation set.

The final test set is reserved for evaluating the selected model.

### Experiments conducted

| Experiment | Purpose |
|---|---|
| Rule pruning | Study the effect of different rule-base sizes |
| Antecedent reduction | Investigate the effect of using fewer input features per rule |
| Class-aware pruning | Examine whether class-sensitive selection improves minority-class performance |
| Threshold experiments | Investigate alternative rule-selection thresholds |
| T-norm comparison | Compare minimum and product antecedent conjunction |
| Rule ablation | Identify the contribution of individual rules |
| False-negative analysis | Investigate missed malignant observations |
| Firing-strength diagnostics | Examine rule activation behaviour and inference bottlenecks |
| Final evaluation | Evaluate the selected 99-rule model on the held-out test set |

**Experimental provenance:** The class-aware pruning, threshold, antecedent-reduction, and feature-importance result files are preserved in `results/`. However, the corresponding generation code for these exploratory analyses is not available in the current repository history. Their CSVs should be treated as recorded exploratory results, not as fully reproducible experiments. The core MLP training, rule extraction, pruning, inference, and final evaluation pipeline is documented separately.

### Antecedent reduction

An additional experiment investigates whether rules can be made shorter by reducing the number of feature conditions.

The preserved experiment results compare configurations using 5, 10, 15, 20, and 30 antecedents per rule.

The available results indicate that the 20-feature configuration provides a compact alternative, while the 30-feature configuration is retained as the final predictive baseline.

**Provenance note:** The result file is present, but the implementation used to generate the feature ranking and select the subsets is not available in the current repository history. The feature-selection procedure and its reported results should therefore be treated as exploratory rather than fully reproducible.

These results suggest that reducing rule length does not necessarily improve the surrogate's fidelity or classification performance.

### Inference operator comparison

Minimum and product t-norms are investigated as alternative methods of evaluating fuzzy conjunction.

The minimum t-norm is selected for the final model.

This makes the inference procedure consistent with the final implementation and its associated evaluation results.

---

## Evaluation Metrics

The fuzzy surrogate is evaluated using multiple metrics because predictive accuracy alone does not fully describe the quality of an extracted model.

### 1. Classification Accuracy

Accuracy measures the proportion of correctly classified observations:

\[
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
\]

### 2. Precision

Precision measures the proportion of predicted positive observations that are actually positive:

\[
Precision = \frac{TP}{TP+FP}
\]

### 3. Recall

Recall measures the proportion of actual positive observations correctly identified:

\[
Recall = \frac{TP}{TP+FN}
\]

### 4. F1-score

The F1-score is the harmonic mean of precision and recall:

\[
F1 = 2\cdot
\frac{Precision \cdot Recall}
{Precision+Recall}
\]

### 5. Macro-F1

Macro-F1 calculates the F1-score independently for each class and takes their unweighted average.

It is particularly useful for evaluating performance when class distributions are unequal.

### 6. Model Fidelity

Fidelity measures how frequently the fuzzy surrogate agrees with the MLP teacher:

\[
Fidelity =
\frac{
\sum_{i=1}^{N}
\mathbb{1}[g_{\phi}(x_i)=f_{\theta}(x_i)]
}{N}
\]

A higher fidelity indicates that the fuzzy model more closely reproduces the teacher's predictions.

Fidelity is distinct from accuracy: a surrogate can agree with the teacher while both models make an incorrect prediction against the ground truth.

### 7. Coverage

Coverage measures the proportion of observations for which at least one fuzzy rule fires.

\[
Coverage =
\frac{\text{Observations covered by at least one rule}}
{\text{Total observations}}
\]

### 8. Fallback Rate

Fallback rate measures the proportion of observations for which no extracted rule fires and the configured fallback mechanism is used.

### 9. Rule-base Complexity

The project also tracks:

- Total number of rules.
- Mean rule length.
- Number of antecedents per rule.

These measurements provide a basic indication of the complexity of the extracted fuzzy model.

---

## Results and Analysis

### Final model performance

The final fuzzy surrogate contains 99 rules.

Its performance is evaluated on the held-out test set of 86 observations.

<details>
<summary>Final fuzzy surrogate: test results</summary>

| Metric | Result |
|---|---:|
| Accuracy | 0.8837 |
| Macro-F1 | 0.8699 |
| Fidelity to MLP | 0.8721 |
| Coverage | 1.0000 |
| Fallback rate | 0.0000 |
| Mean rule length | 30 |

</details>

### Comparison with the MLP teacher

| Metric | MLP teacher | Fuzzy surrogate | Decision-tree surrogate |
|---|---:|---:|---:|
| Test accuracy | 98.84% | 88.37% | 93.02% |
| Test macro-F1 | 98.75% | 86.99% | 92.20% |
| Fidelity to teacher | 100.00%* | 87.21% | 94.19% |
| Coverage | N/A | 100% | N/A |

The MLP teacher achieves the highest predictive performance. The decision-tree surrogate outperforms the fuzzy surrogate on test accuracy, macro-F1, and fidelity to the teacher.

The fuzzy surrogate nevertheless provides a distinct representation through linguistic fuzzy IF–THEN rules, which can be inspected along with their firing behaviour.

*The MLP's 100% fidelity is a self-comparison: it is the teacher model being compared with its own predictions, not an independent surrogate result.*

### Confusion matrix

The final fuzzy model produces the following test confusion matrix, using benign as class 0 and malignant as class 1:

\[
\begin{bmatrix}
52 & 2\\
8 & 24
\end{bmatrix}
\]

| Actual / Predicted | Benign | Malignant |
|---|---:|---:|
| Benign | 52 | 2 |
| Malignant | 8 | 24 |

For malignant observations:

- Precision: 92.31%
- Recall: 75.00%
- F1-score: 82.76%

The model correctly identifies 24 of the 32 malignant observations in the test set.

### Interpretation of results

The final fuzzy system achieves 88.37% test accuracy and 87.21% fidelity to the MLP.

These results indicate that the extracted rule base captures a substantial portion of the teacher's classification behaviour, but does not reproduce it completely.

The 100% test coverage indicates that every test observation activates at least one retained fuzzy rule. The zero fallback rate means that no test prediction depends on the fallback mechanism.

The confusion matrix also shows that the surrogate makes more false-negative predictions than false-positive predictions. This is an important limitation when considering the behaviour of the model on the malignant class.

The results should be interpreted in the context of the relatively small test set and the single random seed used in this experiment.

### Experimental result files

Detailed results are preserved in the `results/` directory:

| File | Description |
|---|---|
| `metrics.csv` | Consolidated experiment metrics |
| `mlp_metrics.json` | Teacher-model evaluation metrics |
| `pruning_validation.csv` | Rule-count pruning experiments |
| `antecedent_reduction_validation.csv` | Feature-count reduction experiments |
| `class_aware_pruning_validation.csv` | Class-aware pruning experiments |
| `threshold_validation.csv` | Rule-selection threshold experiments |
| `activation_diagnostic.csv` | Activation-related diagnostics |
| `feature_importance.csv` | Feature importance results |
| `false_negative_rule_details.csv` | Rule-level false-negative analysis |
| `false_negative_rule_summary.csv` | Summary of false-negative analysis |
| `remaining_false_negatives.csv` | Remaining false-negative observations |
| `rule_0_ablation_changes.csv` | Changes resulting from rule ablation |
| `final_candidate_test_predictions.csv` | Final fuzzy model test predictions |
| `decision_tree_validation.csv` | Decision-tree surrogate validation results |
| `decision_tree_test_predictions.csv` | Decision-tree surrogate test predictions |
| `model_comparison_test.csv` | Held-out test comparison of MLP, fuzzy, and decision-tree models |
| `metrics_summary.csv` | Consolidated metrics summary |
| `reproducibility_manifest.json` | Experiment and file-hash metadata |

---

## Interpretability and Explainability

The central motivation of this project is to provide a more understandable representation of a neural network's classification behaviour.

The MLP operates through learned numerical weights and nonlinear transformations across hidden layers.

The fuzzy surrogate expresses its decisions through rules containing linguistic conditions.

For example, a rule can be represented as:

```text
IF feature_1 is High
AND feature_2 is Medium
AND feature_3 is Low
...
THEN class is Malignant
```

This representation makes it possible to inspect:

- Which fuzzy conditions appear in a rule.
- Which class a rule predicts.
- The rule's weight and support metadata.
- The strength with which a rule fires for an observation.
- Which rules contribute to the final class activation.
- How pruning or ablation changes the rule base.

The inference implementation also produces an audit trace containing information about fired rules, antecedent strengths, firing strengths, and consequents.

This allows individual predictions to be examined at the rule level.

**Important:** The extracted rules are an approximation of the teacher's behaviour. They should not be interpreted as a complete or exact explanation of the MLP's internal computations.

---

## Repository Structure

```text
fuzzy-rule-extraction/
│
├── artifacts/
│   ├── final_model_metadata.json
│   ├── final_rules_top99.joblib
│   ├── membership_functions.joblib
│   ├── mlp_model.joblib
│   ├── preprocessor.joblib
│   ├── selection_metadata.json
│   └── feature_metadata.json
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│   ├── metrics.csv
│   ├── metrics_summary.csv
│   ├── mlp_metrics.json
│   ├── mlp_train_predictions.csv
│   ├── mlp_validation_predictions.csv
│   ├── mlp_test_predictions.csv
│   ├── pruning_validation.csv
│   ├── antecedent_reduction_validation.csv
│   ├── class_aware_pruning_validation.csv
│   ├── threshold_validation.csv
│   ├── feature_importance.csv
│   ├── activation_diagnostic.csv
│   ├── false_negative_rule_details.csv
│   ├── false_negative_rule_summary.csv
│   ├── remaining_false_negatives.csv
│   ├── rule_0_ablation_changes.csv
│   ├── final_candidate_test_predictions.csv
│   ├── decision_tree_validation.csv
│   ├── decision_tree_test_predictions.csv
│   ├── model_comparison_test.csv
│   └── reproducibility_manifest.json
│
├── src/
│   ├── data_pipeline.py
│   ├── train_mlp.py
│   ├── predict.py
│   ├── membership.py
│   ├── rule_types.py
│   ├── extract_rules.py
│   ├── fuzzy_inference.py
│   ├── prune_rules.py
│   ├── evaluate.py
│   ├── evaluate_final_candidate.py
│   ├── train_decision_tree.py
│   ├── run_experiments.py
│   ├── compare_tnorms.py
│   ├── analyze_results.py
│   ├── report_tables.py
│   ├── visualize_results.py
│   ├── reproducibility.py
│   ├── update_metadata.py
│   ├── analyze_ablation_changes.py
│   ├── analyze_false_negative_rules.py
│   ├── analyze_firing_bottlenecks.py
│   ├── analyze_remaining_false_negatives.py
│   ├── compare_competing_antecedents.py
│   ├── inspect_competing_rules.py
│   └── test_rule_ablation.py
│
├── tests/
│   ├── test_analyze_results.py
│   ├── test_evaluate.py
│   ├── test_extract_rules.py
│   ├── test_inference.py
│   ├── test_membership.py
│   ├── test_prune_rules.py
│   ├── test_report_tables.py
│   ├── test_reproducibility.py
│   ├── test_rule_types.py
│   ├── test_run_experiments.py
│   └── test_visualize_results.py
│
├── report/
├── presentation/
├── config.yaml
├── requirements.txt
├── .gitignore
└── README.md
### Important artifacts

| Artifact | Purpose |
|---|---|
| `mlp_model.joblib` | Trained neural network teacher |
| `preprocessor.joblib` | Fitted preprocessing pipeline |
| `membership_functions.joblib` | Fitted fuzzy membership functions |
| `final_rules_top99.joblib` | Final 99-rule fuzzy model |
| `final_model_metadata.json` | Final model configuration and evaluation metadata |
| `selection_metadata.json` | Validation-based model selection details |
| `feature_metadata.json` | Dataset, feature, and split metadata |

---

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- pip

The project was developed and tested using Python 3.12.3.

### 1. Clone the repository

```bash
git clone <repository-url>
cd fuzzy-rule-extraction
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Running the Project

The project is implemented as modular Python components. The saved artifacts are available in `artifacts/`, while experimental outputs are stored in `results/`.

### Core command-line stages

Run these commands from the repository root, in order:

```bash
# 1. Download the dataset, create stratified splits and preprocess the data
python -m src.data_pipeline

# 2. Train the MLP teacher using the training split
python -m src.train_mlp

# 3. Export MLP predictions for the train, validation and test splits
python -m src.predict
```

**Note:** These commands regenerate data, model artifacts and prediction files. They can overwrite existing outputs.

### Fuzzy rule extraction and evaluation

The fuzzy pipeline is implemented through reusable Python functions rather than a single command-line entry point.

| Module | Main function | Purpose |
|---|---|---|
| `src/membership.py` | `fit_membership_functions()` | Fits Low, Medium and High membership functions using training data |
| `src/membership.py` | `fuzzify_dataframe()` | Converts observations into fuzzy membership degrees |
| `src/extract_rules.py` | `extract_rules()` | Generates and resolves fuzzy rules using MLP predictions |
| `src/prune_rules.py` | `prune_rules()` | Ranks and selects the top-N rules |
| `src/fuzzy_inference.py` | `predict_dataset()` | Predicts observations using the fuzzy rule base |
| `src/run_experiments.py` | `run_rule_experiments()` | Evaluates full and pruned rule bases |
| `src/evaluate_final_candidate.py` | Script | Evaluates the final 99-rule candidate using saved intermediate artifacts |

The final evaluation script expects the processed test data, MLP test predictions, membership functions and selected top-100 rules to be available. It removes original Rule #0 before evaluating the final candidate.

**Reproduction note:** The repository does not currently provide a single end-to-end command that connects every training, extraction, pruning and evaluation stage. The saved final artifacts can be inspected independently, but regenerating the complete experiment requires executing the function-based stages in sequence.

## Testing

The repository includes an automated test suite covering the main components of the project.

Run all tests from the repository root:

```bash
pytest -v
```

The current implementation has **89 passing tests**.

The tests cover:

- Membership function behaviour and parameter validation.
- Fuzzification of individual observations and datasets.
- Candidate rule generation.
- Conflict resolution.
- Rule data structures and validation.
- Minimum t-norm inference.
- Maximum aggregation.
- Class prediction and tie handling.
- Fallback behaviour and coverage.
- Rule ranking and pruning.
- Evaluation metrics.
- Experiment configuration validation.
- Reproducibility metadata and file hashing.
- Result-table generation and visualisation utilities.

The test suite is intended to verify both individual components and their expected interfaces.

---

## Reproducibility

Reproducibility is an important part of the experimental workflow.

The project uses a fixed random seed of 42 for the dataset split and model configuration.

The following information is preserved:

- Dataset and split metadata.
- Feature ordering and class mapping.
- Preprocessing configuration.
- MLP architecture and hyperparameters.
- Fuzzy membership-function definitions.
- Rule-selection configuration.
- Inference method.
- Evaluation metrics.
- Input-file hashes.
- Package versions.
- Experiment parameters.

The reproducibility manifest is stored at:

`results/reproducibility_manifest.json`

It records selected experiment inputs, file hashes, environment details, and configuration metadata. This supports change detection, but does not by itself guarantee complete reproduction of every exploratory experiment.

The final model metadata also records the validation-based selection process and confirms that the test set was not used for model selection.

The saved model artifacts allow the final fuzzy system to be inspected without retraining the teacher model.

---

## Technologies Used

| Technology | Application |
|---|---|
| Python | Core implementation |
| NumPy | Numerical computation |
| Pandas | Dataset manipulation and experiment results |
| scikit-learn | Preprocessing, MLP training, and evaluation |
| Joblib | Model and artifact serialization |
| Matplotlib | Result visualisation |
| PyYAML | Experiment configuration |
| Pytest | Automated testing |
| Git and GitHub | Version control and collaboration |

---

## Limitations

Although the project demonstrates the feasibility of extracting a fuzzy surrogate from a trained neural network, several limitations remain.

### 1. Approximation rather than equivalence

The fuzzy system approximates the teacher's classification behaviour. It does not reproduce the MLP's internal representations or guarantee identical predictions.

### 2. Predictive performance trade-off

The final fuzzy surrogate achieves 88.37% test accuracy compared with 98.84% for the MLP teacher.

The reduction in predictive performance illustrates the trade-off involved in replacing a more flexible model with a finite rule base.

### 3. Rule length

The final model contains 99 rules, each with 30 antecedent conditions.

Although the number of rules is reduced, the resulting rules are still relatively long. This limits their immediate interpretability.

### 4. Dataset size

The dataset contains 569 observations, with only 86 observations in the held-out test set.

Consequently, the reported test metrics should be interpreted with appropriate caution.

### 5. Single random seed

The final experiment uses a fixed random seed of 42. Results may vary under different data partitions and training initialisations.

### 6. Domain limitations

The dataset is used as a machine-learning benchmark. The resulting model has not undergone clinical validation and must not be used as a medical diagnostic system.

---

## Future Improvements

Several extensions could improve the performance and interpretability of the current framework.

- **Antecedent simplification:** Reduce the number of feature conditions in individual rules while preserving predictive performance.
- **Advanced rule extraction:** Investigate alternative approaches for generating rules from neural-network predictions.
- **Rule merging:** Identify similar rules and combine them where appropriate.
- **Improved pruning:** Explore optimisation methods that jointly consider predictive performance, fidelity, coverage, and complexity.
- **Alternative membership functions:** Investigate Gaussian, trapezoidal, and adaptive membership functions.
- **Multi-seed evaluation:** Repeat experiments across multiple random seeds to assess stability.
- **Additional datasets:** Evaluate the extraction framework on other structured classification problems.
- **Improved interpretability evaluation:** Introduce measures of rule redundancy, comprehensibility, and feature-level explanation quality.
- **Inference optimisation:** Investigate alternative aggregation and decision mechanisms.

---

## Disclaimer

This project is an academic investigation into interpretable machine learning and fuzzy rule extraction.

The Breast Cancer Wisconsin dataset is used for experimental classification. The extracted fuzzy system is a research prototype and is not a clinically validated diagnostic tool.

---

## Project Summary

**Opening the Black Box: Extracting Fuzzy Rules from Trained Neural Networks** explores how a trained neural network's classification behaviour can be approximated using a fuzzy rule-based surrogate.

The project implements a complete experimental workflow involving neural network training, fuzzy membership functions, rule extraction, conflict resolution, inference, pruning, ablation, and evaluation.

The final system contains 99 fuzzy rules and achieves:

- 88.37% test accuracy.
- 86.99% test macro-F1.
- 87.21% fidelity to the MLP teacher.
- 100% test coverage.

The work demonstrates both the potential and the limitations of using fuzzy rule systems to make neural-network behaviour more accessible and interpretable.

**The central objective is not simply to build another classifier, but to investigate how predictive behaviour can be translated into an inspectable collection of fuzzy IF–THEN rules.**