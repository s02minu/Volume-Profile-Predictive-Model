import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def prepare_data(df_labels, label_col='label_vah_acceptance'):

    # defining the feature columns the model will learn from
    feature_cols = [
        'prev_poc_position',
        'prev_va_width',
        'prev_va_coverage',
        'prev_delta',
        'prev_buy_ratio',
        'prev_total_volume',
        'poc_direction',
        'price_vs_prev_poc',
        'dist_prev_poc',
        'dist_prev_vah',
        'dist_prev_val'
    ]

    # prev_day_type is categorical so we encode it as numbers
    day_type_map = {
        'accumulation': 0,
        'neutral': 1,
        'trending': 2,
        'distribution': 3
    }
    df_labels['prev_day_type_encoded'] = df_labels['prev_day_type'].map(day_type_map)
    feature_cols.append('prev_day_type_encoded')

    X = df_labels[feature_cols]
    y = df_labels[label_col]

    return X, y, feature_cols


def train_model(X, y):

    # splitting into training and testing sets
    # 80% for training, 20% for testing
    # random_state ensures reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scaling features so no single feature dominates due to its size
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # training logistic regression as our baseline model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # evaluating on the unseen test set
    y_pred = model.predict(X_test_scaled)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # confusion matrix — visual representation of correct vs incorrect predictions
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix — VAH Acceptance')
    plt.tight_layout()
    plt.show()

    return model, scaler, X_test_scaled, y_test, y_pred


def show_feature_importance(model, feature_cols):
    # logistic regression coefficients tell us which features
    # the model is relying on most — positive = bullish signal, negative = bearish
    importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', ascending=False)

    print("\nFeature Importance (coefficients):")
    print(importance.to_string(index=False))


def save_model(model, scaler):
    # saving trained model and scaler to disk for use in main.py
    joblib.dump(model, 'data/model.pkl')
    joblib.dump(scaler, 'data/scaler.pkl')
    print("\nModel and scaler saved to data/")


if __name__ == "__main__":
    df_labels = pd.read_csv('data/df_labels.csv')
    X, y, feature_cols = prepare_data(df_labels)
    model, scaler, X_test_scaled, y_test, y_pred = train_model(X, y)
    show_feature_importance(model, feature_cols)
    save_model(model, scaler)