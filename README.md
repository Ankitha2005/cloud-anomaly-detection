# 🛡️ Cloud Anomaly Detection System

An ensemble-based machine learning system for detecting security anomalies in cloud computing environments.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 🎯 Overview

This project implements a comprehensive anomaly detection framework that combines:
- **Isolation Forest** - Tree-based anomaly isolation
- **One-Class SVM** - Boundary-based novelty detection
- **Local Outlier Factor** - Density-based outlier detection
- **Temporal Autoencoder** - Attention-enhanced LSTM for sequence anomalies

## 📊 Key Results

| Model | F1 Score | AUC-ROC | FPR |
|-------|----------|---------|-----|
| **Temporal AE (tuned)** | **0.917** | 0.971 | 0.6% |
| Isolation Forest | 0.674 | **0.984** | 1.7% |
| Ensemble | 0.390 | 0.980 | 16.4% |

- **13 attack types** tested with **76.6% overall detection rate**
- **100% detection** for: cryptomining, ransomware, VM escape, data exfiltration

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cloud-anomaly-detection.git
cd cloud-anomaly-detection

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run src/dashboard/app.py

# Run experiments
python experiments/run_experiment.py

# Run tests
pytest tests/ -v
```

## 📁 Project Structure

```
sdp/
├── src/
│   ├── data_preprocessing/    # Data loading & cleaning
│   ├── feature_engineering/   # 74 engineered features
│   ├── models/                # ML models (IF, OCSVM, LOF, Autoencoder)
│   ├── ensemble/              # Model combination
│   ├── evaluation/            # Metrics & adversarial testing
│   ├── explainability/        # SHAP/LIME explanations
│   └── dashboard/             # Streamlit monitoring UI
├── experiments/               # Experiment scripts
├── tests/                     # 47 unit tests
├── results/                   # Output metrics & figures
└── docs/                      # Research paper & documentation
```

## 🔬 Features

### Feature Engineering (74 features)
- **Temporal**: Hour, day of week, business hours (cyclically encoded)
- **Rolling Statistics**: Mean, std, max over 5/10/30 min windows
- **Cluster Features**: Deviation from cluster-wide behavior
- **Interaction**: CPU×RAM, bandwidth/connection, error rate

### Attack Types Detected
| High Detection (100%) | Partial Detection | Challenging |
|----------------------|-------------------|-------------|
| Cryptomining | DDoS (49%) | Botnet C2 (7%) |
| Ransomware | Slowloris (29%) | Covert Channel (11%) |
| VM Escape | | |
| Data Exfiltration | | |

## 📈 Dashboard

The Streamlit dashboard provides:
- Real-time anomaly monitoring
- Model performance comparison
- Attack detection analysis
- Feature importance visualization
- Ablation study results

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v
```

## 📝 Research Paper

See `docs/research_paper.md` for the full research paper including:
- Methodology and architecture
- Experimental setup
- Detailed results and ablation studies
- Discussion and future work

## 🛠️ Technologies

- Python 3.9+
- TensorFlow/Keras
- Scikit-learn
- Streamlit
- SHAP/LIME

## 📄 License

MIT License

## 👥 Authors

Cloud Security Research Team

