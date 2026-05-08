# 🔒 Anomaly Node Detection Using AI in Cloud Computing

### Interactive Demo Website

A state-of-the-art hybrid machine learning system for detecting anomalies in cloud computing environments using UNSW-NB15 dataset.

---

## 🚀 Live Demo

**[View Live Demo](#)** ← Add your Streamlit Cloud URL here after deployment

---

## ✨ Features

- 📊 **Interactive Dashboard** - Real-time visualization of project overview
- 🎮 **Live Anomaly Detection** - Simulate network traffic and detect anomalies
- 📈 **Model Comparisons** - Compare XGBoost, KNN, SVM, and Random Forest
- 🔬 **Complete Methodology** - 11-step ML pipeline explained
- 📉 **Performance Metrics** - Accuracy, Precision, Recall, F1, AUC-ROC

---

## 🎯 About The Project

This research develops a hybrid machine learning model combining:
- **4 Base Classifiers**: XGBoost, KNN, SVM, Random Forest
- **4 Metaheuristic Optimizers**: GOA, PSO, ACO, Cuckoo Search
- **UNSW-NB15 Dataset**: 257,673 network traffic records
- **9 Attack Types**: Fuzzers, DoS, Exploits, Reconnaissance, and more

### Key Results:
- ✅ **95%+ Accuracy** in anomaly detection
- ✅ **Real-time Classification** with confidence scores
- ✅ **Comprehensive Metrics** across all models

---

## 🏃 Run Locally

```bash
# Clone the repository
git clone <your-repo-url>
cd sdp

# Install dependencies
pip install -r requirements_demo.txt

# Run the demo
streamlit run demo_app.py
```

Open **http://localhost:8501** in your browser.

---

## 🌐 Deploy Your Own

### Streamlit Cloud (Recommended)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub and select this repo
4. Set main file: `demo_app.py`
5. Deploy!

### Other Options

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for more deployment options.

---

## 📊 Dataset

**UNSW-NB15**: A comprehensive network intrusion dataset
- Training: 175,341 records
- Testing: 82,332 records
- Features: 45 network attributes
- Classes: Normal + 9 attack types

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **ML Libraries**: scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy

---

## 📝 License

For academic research purposes.

---

## 🙏 Acknowledgments

Special thanks to the UNSW Canberra Cyber team for the UNSW-NB15 dataset.

---

**Built with ❤️ for cybersecurity research**
