import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score
)
from model import prepare_data


def load_model():
    # loading the trained model and scaler from disk
    model = joblib.load('data/model.pkl')
    scaler = joblib.load('data/scaler.pkl')
    return model, scaler


def evaluate_model(model, scaler, X, y, feature_cols):

    # scaling the features
    X_scaled = scaler.transform(X)

    # getting predictions and probabilities
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # ── CLASSIFICATION REPORT ────────────────────────────────────────────────
    print("Classification Report:")
    print(classification_report(y, y_pred))

    # ── BASELINE COMPARISON ──────────────────────────────────────────────────
    # how much better are we than just always predicting the majority class?
    baseline = max(y.mean(), 1 - y.mean()) * 100
    model_acc = (y == y_pred).mean() * 100
    print(f"Baseline accuracy (majority class): {baseline:.1f}%")
    print(f"Model accuracy:                     {model_acc:.1f}%")
    print(f"Improvement over baseline:          {model_acc - baseline:.1f}%")

    # ── ROC CURVE ────────────────────────────────────────────────────────────
    # ROC curve shows the tradeoff between catching true positives
    # and avoiding false positives at different probability thresholds
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#4DA6FF', linewidth=2, label=f'Model (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — VAH Acceptance Model')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\nAUC Score: {auc:.3f}")
    print("AUC of 0.5 = random guessing, 1.0 = perfect model")

    # ── FEATURE IMPORTANCE ───────────────────────────────────────────────────
    importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', ascending=False)

    print("\nFeature Importance:")
    print(importance.to_string(index=False))

    return y_pred, y_prob, auc


if __name__ == "__main__":
    df_labels = pd.read_csv('data/df_labels.csv')
    X, y, feature_cols = prepare_data(df_labels)
    model, scaler = load_model()
    y_pred, y_prob, auc = evaluate_model(model, scaler, X, y, feature_cols)