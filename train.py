import os, urllib.request, pandas as pd, joblib, json, mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

def main():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Download dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    dst = "data/winequality-red.csv"
    urllib.request.urlretrieve(url, dst)

    df = pd.read_csv(dst, sep=';')
    df.columns = [c.replace(" ", "_") for c in df.columns]

    # Binary label
    df["label"] = (df["quality"] >= 6).astype(int)
    X = df.drop(columns=["quality", "label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment("wine-quality-mlops")

    with mlflow.start_run():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500))
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(pipe, "model")

        # Save artifacts
        joblib.dump(pipe, "artifacts/model.joblib")
        with open("artifacts/feature_order.json", "w") as f:
            json.dump(list(X.columns), f, indent=2)

        return acc, f1

if __name__ == "__main__":
    acc, f1 = main()
    print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}")
