# model_pipeline.py
import pandas as pd
import seaborn as sns
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb


def prepare_data():
    """Load, preprocess, and save data as .joblib files."""
    print("Loading mpg dataset...")
    df = sns.load_dataset('mpg')
    df = df.dropna(subset=['horsepower']).reset_index(drop=True)

    X = df.drop(['origin', 'name'], axis=1)
    y = df['origin']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # SAVE AS .joblib
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    joblib.dump(X_train_sc, 'X_train.joblib')
    joblib.dump(X_test_sc, 'X_test.joblib')
    joblib.dump(y_train, 'y_train.joblib')
    joblib.dump(y_test, 'y_test.joblib')
    joblib.dump(le.classes_, 'classes.joblib')

    print("Data saved with .joblib extension:")
    print(f"  Location: {os.getcwd()}")
    print("  Files: X_train.joblib, X_test.joblib, y_train.joblib, y_test.joblib")
    print("         scaler.joblib, label_encoder.joblib, classes.joblib")

    return X_train_sc, X_test_sc, y_train, y_test, le.classes_


def train_model():
    """Load .joblib data and train model."""
    required = [
        'X_train.joblib', 'y_train.joblib',
        'X_test.joblib',  'y_test.joblib'
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"Error: Missing files: {', '.join(missing)}")
        print("Run: python main.py --prepare")
        return

    print("Loading training data...")
    X_train = joblib.load('X_train.joblib')
    y_train = joblib.load('y_train.joblib')
    X_test  = joblib.load('X_test.joblib')
    y_test  = joblib.load('y_test.joblib')

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    )

    print("Training model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
    model.save_model('xgboost_model.json')
    print(f"Model saved: {os.path.join(os.getcwd(), 'xgboost_model.json')}")
    return model


def evaluate_model():
    """Load model and .joblib data, evaluate, save results."""
    if not os.path.exists('xgboost_model.json'):
        print("Error: Model not found. Run --train first.")
        return
    if not os.path.exists('X_test.joblib'):
        print("Error: Test data not found. Run --prepare first.")
        return

    print("Loading model and data...")
    model = xgb.XGBClassifier()
    model.load_model('xgboost_model.json')

    X_train = joblib.load('X_train.joblib')
    X_test  = joblib.load('X_test.joblib')
    y_train = joblib.load('y_train.joblib')
    y_test  = joblib.load('y_test.joblib')
    classes = joblib.load('classes.joblib')

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    report_text = classification_report(y_test, test_pred, target_names=classes)
    report_dict = classification_report(y_test, test_pred, target_names=classes, output_dict=True)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test  Accuracy: {test_acc:.4f}")
    print("\nClassification Report:\n" + report_text)

    results = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "classification_report": report_dict
    }

    # Save results
    with open("results.txt", "w") as f:
        f.write(f"Train Accuracy: {train_acc:.4f}\n")
        f.write(f"Test  Accuracy: {test_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)

    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    joblib.dump(results, "evaluation_results.joblib")  # <-- .joblib

    print(f"\nResults saved in: {os.getcwd()}")
    print("  - results.txt")
    print("  - metrics.json")
    print("  - evaluation_results.joblib")

    return results
