import os
from train import main as train_main

def test_training_creates_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    acc, f1 = train_main()
    assert os.path.exists("artifacts/model.joblib")
    assert os.path.exists("artifacts/feature_order.json")
    assert 0.5 <= acc <= 1.0
    assert 0.5 <= f1 <= 1.0
