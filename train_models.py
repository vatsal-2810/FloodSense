"""
FloodSense — Model Training Script
Trains: Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN
+ LSTM-inspired temporal feature engineering
Compares all models with metrics and saves best model artifacts.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "flood_dataset_cleaned.csv"
ARTIFACT_DIR = "models"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ─── Load Data ─────────────────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")
print(f"   Flood distribution:\n{df['flood_occurred'].value_counts()}\n")

# ─── Feature Engineering (LSTM-inspired temporal features) ─────────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Since we don't have real time-series here, we engineer features
    that LSTM would learn automatically from sequences:
    - Rolling statistics simulating temporal context
    - Interaction features (joint effects)
    """
    df = df.copy()
    # Sort by rainfall to simulate temporal order
    df = df.sort_values("rainfall_mm").reset_index(drop=True)

    # LSTM-inspired: rolling window features (window=3)
    for col in ["rainfall_mm", "water_level_m", "barometric_pressure_hpa"]:
        df[f"{col}_roll_mean3"] = df[col].rolling(window=3, min_periods=1).mean()
        df[f"{col}_roll_std3"]  = df[col].rolling(window=3, min_periods=1).std().fillna(0)
        df[f"{col}_diff1"]      = df[col].diff().fillna(0)

    # Interaction features
    df["rain_x_humidity"]       = df["rainfall_mm"] * df["humidity_pct"] / 100
    df["waterlevel_x_discharge"] = df["water_level_m"] * df["river_discharge_m3s"]
    df["pressure_drop"]         = 1013.25 - df["barometric_pressure_hpa"]
    df["soil_saturation_risk"]  = df["soil_moisture_pct"] * df["rainfall_mm"] / 100

    return df

df = add_temporal_features(df)

# ─── Encode Categoricals ───────────────────────────────────────────────────────
le_lc = LabelEncoder()
le_st = LabelEncoder()
df["land_cover_enc"] = le_lc.fit_transform(df["land_cover"])
df["soil_type_enc"]  = le_st.fit_transform(df["soil_type"])

# ─── Select Features ───────────────────────────────────────────────────────────
base_features = [
    "rainfall_mm", "temperature_c", "humidity_pct", "river_discharge_m3s",
    "water_level_m", "elevation_m", "land_cover_enc", "soil_type_enc",
    "infrastructure", "historical_floods", "barometric_pressure_hpa", "soil_moisture_pct",
]
temporal_features = [
    "rainfall_mm_roll_mean3", "rainfall_mm_roll_std3", "rainfall_mm_diff1",
    "water_level_m_roll_mean3", "water_level_m_diff1",
    "barometric_pressure_hpa_roll_mean3", "barometric_pressure_hpa_diff1",
    "rain_x_humidity", "waterlevel_x_discharge", "pressure_drop", "soil_saturation_risk",
]
feature_cols = base_features + temporal_features

X = df[feature_cols]
y = df["flood_occurred"]

# ─── Augment Small Dataset ─────────────────────────────────────────────────────
def augment_dataset(X: pd.DataFrame, y: pd.Series, factor: int = 15) -> tuple:
    """Add Gaussian noise copies to enlarge small datasets."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_list, y_list = [X], [y]
    for _ in range(factor):
        noisy = X.copy()
        for c in num_cols:
            noisy[c] = noisy[c] + np.random.normal(0, X[c].std() * 0.05, len(X))
        X_list.append(noisy)
        y_list.append(y)
    return pd.concat(X_list, ignore_index=True), pd.concat(y_list, ignore_index=True)

print("🔧 Augmenting dataset...")
X_aug, y_aug = augment_dataset(X, y, factor=15)
print(f"   Augmented shape: {X_aug.shape}\n")

# ─── Train/Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_aug, test_size=0.20, random_state=42, stratify=y_aug
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── Model Definitions ─────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        random_state=42, max_iter=1000, C=0.5
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, max_depth=10,
        min_samples_leaf=2, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, random_state=42, learning_rate=0.05,
        max_depth=5, subsample=0.8
    ),
    "SVM (RBF)": SVC(
        probability=True, random_state=42, C=1.0, kernel="rbf", gamma="scale"
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, weights="distance"),
    # LSTM-INSPIRED: uses temporal features + Gradient Boosting as base learner
    # In production with TensorFlow, replace with actual LSTM architecture
    "LSTM-Inspired (GB+Temporal)": GradientBoostingClassifier(
        n_estimators=300, random_state=42, learning_rate=0.03,
        max_depth=4, subsample=0.75, min_samples_leaf=3
    ),
}

# ─── Training & Evaluation ─────────────────────────────────────────────────────
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
trained_models = {}

print("=" * 65)
print("  MODEL TRAINING & EVALUATION")
print("=" * 65)

best_model_name = None
best_f1         = -1

for name, model in models.items():
    print(f"\n▶  {name}")

    model.fit(X_train_sc, y_train)
    trained_models[name] = model

    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    cv_f1  = cross_val_score(
        model, scaler.transform(X_aug), y_aug, cv=cv, scoring="f1"
    )

    r = {
        "accuracy":    round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision":   round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall":      round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "f1":          round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "roc_auc":     round(roc_auc_score(y_test, y_prob) * 100, 2),
        "cv_f1_mean":  round(cv_f1.mean() * 100, 2),
        "cv_f1_std":   round(cv_f1.std() * 100, 2),
    }
    results[name] = r

    print(f"   Accuracy:  {r['accuracy']}%")
    print(f"   Precision: {r['precision']}%")
    print(f"   Recall:    {r['recall']}%")
    print(f"   F1 Score:  {r['f1']}%")
    print(f"   ROC-AUC:   {r['roc_auc']}%")
    print(f"   CV F1:     {r['cv_f1_mean']}% ± {r['cv_f1_std']}%")

    cm = confusion_matrix(y_test, y_pred)
    print(f"   Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    if r["f1"] > best_f1:
        best_f1         = r["f1"]
        best_model_name = name

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  COMPARISON SUMMARY")
print("=" * 65)
print(f"{'Model':<35} {'Acc':>6} {'F1':>6} {'AUC':>6} {'CV-F1':>10}")
print("-" * 65)
for name, r in sorted(results.items(), key=lambda x: -x[1]["f1"]):
    marker = " ← BEST" if name == best_model_name else ""
    print(f"{name:<35} {r['accuracy']:>5}% {r['f1']:>5}% {r['roc_auc']:>5}% {r['cv_f1_mean']:>7}%{marker}")

print(f"\n🏆 Best Model: {best_model_name} (F1={best_f1}%)")

# ─── Save Artifacts ───────────────────────────────────────────────────────────
print(f"\n💾 Saving model artifacts to '{ARTIFACT_DIR}/'...")

joblib.dump(trained_models[best_model_name],             f"{ARTIFACT_DIR}/best_model.pkl")
joblib.dump(trained_models["Random Forest"],             f"{ARTIFACT_DIR}/rf_model.pkl")
joblib.dump(trained_models["Gradient Boosting"],         f"{ARTIFACT_DIR}/gb_model.pkl")
joblib.dump(trained_models["Logistic Regression"],       f"{ARTIFACT_DIR}/lr_model.pkl")
joblib.dump(trained_models["LSTM-Inspired (GB+Temporal)"], f"{ARTIFACT_DIR}/lstm_inspired_model.pkl")
joblib.dump(scaler,                                       f"{ARTIFACT_DIR}/scaler.pkl")
joblib.dump(le_lc,                                        f"{ARTIFACT_DIR}/le_land_cover.pkl")
joblib.dump(le_st,                                        f"{ARTIFACT_DIR}/le_soil_type.pkl")
joblib.dump(feature_cols,                                 f"{ARTIFACT_DIR}/feature_cols.pkl")

with open(f"{ARTIFACT_DIR}/model_results.json", "w") as f:
    json.dump({"results": results, "best_model": best_model_name}, f, indent=2)

with open(f"{ARTIFACT_DIR}/label_classes.json", "w") as f:
    json.dump({
        "land_cover": list(le_lc.classes_),
        "soil_type":  list(le_st.classes_),
    }, f, indent=2)

print("✅ All artifacts saved!\n")
print("Next step: Deploy the FloodSense API → see DEPLOYMENT.md")
