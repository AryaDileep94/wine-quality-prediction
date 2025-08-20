# wine-quality-prediction

This project demonstrates how to take a simple ML problem and apply MLOps best practices to make it reproducible, testable, and deployable.
Instead of focusing on complex models, the emphasis here is on production readiness: CI/CD, containerization, experiment tracking, and model serving.

📊 Dataset

Source: UCI Machine Learning Repository – Wine Quality Data Set.

Description: Chemical properties of Portuguese Vinho Verde red wines (e.g., acidity, sugar, alcohol).

Task: Predict wine quality. For simplicity, converted to binary classification:

Good wine (1): quality ≥ 6

Bad wine (0): quality < 6

This dataset is small and public, making it ideal for MLOps demonstrations.

🛠️ Tech Stack (MLOps Focus)

Python, scikit-learn, pandas, numpy – modeling

FastAPI – model serving API

MLflow – experiment tracking & model logging

DVC (Data Version Control) – data/model versioning (ready to integrate)

Docker – containerized deployment

GitHub Actions – CI/CD for testing & builds

pytest – unit testing (training + API)



