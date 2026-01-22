# Research Design Document
## Cloud Node Anomaly Detection Using AI in Cloud Computing

## 1. Research Questions

### RQ1: Multi-Modal Feature Integration
Does combining network behavior features (UNSW-NB15) with cloud resource metrics (CloudSim) improve anomaly detection compared to using either data source alone?

### RQ2: Temporal Modeling
Does temporal modeling (sequence-based Autoencoder) outperform static feature-based models?

### RQ3: Ensemble Effectiveness
Does an ensemble of IF + Temporal AE + OCSVM reduce false positives vs individual models?

### RQ4: Adversarial Robustness
How robust is the system against adversarially injected attack behaviors?

## 2. Evaluation Metrics

- Precision, Recall, F1-Score
- ROC-AUC (target > 0.95)
- False Positive Rate (target < 0.05)
- Detection Latency (windows until detection)

## 3. Ablation Studies

- A1: UNSW only vs CloudSim only vs Combined
- A2: Static IF vs IF with temporal features
- A3: Single models vs Ensemble
- A4: With vs Without adversarial injection

## 4. Novel Contributions

1. Multi-modal behavior representation (UNSW-NB15 + CloudSim integration)
2. Temporal unsupervised ensemble (IF + Temporal AE + OCSVM)
3. Adversarial robustness evaluation
4. Explainable anomaly analysis (SHAP)
5. Cloud-specific feature engineering

