# 🍷 Wine Quality Prediction – MLOps Project

[![CI Pipeline](https://github.com/AryaDileep94/wine-quality-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/AryaDileep94/wine-quality-prediction/actions/workflows/ci.yml)

An end-to-end MLOps project using FastAPI, MLflow, pytest, and GitHub Actions 🚀

This project demonstrates how to take a simple ML problem (wine quality prediction) and make it **production-ready** with MLOps best practices.  
Instead of focusing only on accuracy, the emphasis here is on **reproducibility, testing, deployment, and automation**.

---

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository – Wine Quality Data Set](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)
- **Description**: Chemical properties of Portuguese *Vinho Verde* red wines (e.g., acidity, sugar, alcohol).
- **Task**: Predict wine quality as **binary classification**:
  - Good wine (`1`): quality ≥ 6
  - Bad wine (`0`): quality < 6


## 🛠️ Tech Stack

- **Python**: pandas, numpy, scikit-learn
- **Model Serving**: FastAPI, Uvicorn
- **Experiment Tracking**: MLflow
- **Testing**: pytest (unit tests for training + API)
- **CI/CD**: GitHub Actions (automated testing on every push)
- **Version Control**: DVC (ready to integrate)
- **Deployment**: Docker, Ngrok (for quick demo)

## ⚙️ Setup & Installation

Clone this repository:
```bash
git clone https://github.com/AryaDileep94/wine-quality-prediction.git
cd wine-quality-prediction








