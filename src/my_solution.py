import os
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import joblib

# --- Helper Functions ---
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    res = np.where(denom == 0, 0.0, diff / denom)
    return np.mean(res) * 100.0

def extract_ipq(text):
    if pd.isna(text):
        return 1
    patterns = [r'IPQ: (\d+)', r'(\d+)\s*-?pack', r'(\d+)\s*-?count', r'(\d+)\s*-?ct', r'(\d+)\s*set']
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 1

# --- Create folders if not exist ---
os.makedirs("../outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Load Data ---
train_df = pd.read_csv("../dataset/train.csv")
test_df = pd.read_csv("../dataset/test.csv")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# --- Feature Engineering ---
print("Extracting IPQ and creating enriched text...")
train_df['catalog_content'] = train_df['catalog_content'].fillna("").astype(str)
test_df['catalog_content'] = test_df['catalog_content'].fillna("").astype(str)
train_df['ipq'] = train_df['catalog_content'].apply(extract_ipq)
test_df['ipq'] = test_df['catalog_content'].apply(extract_ipq)

# New: Feature Crossing - adding IPQ as a word to the text
train_df['enriched_content'] = train_df.apply(lambda row: f"{row['catalog_content']} ipq{row['ipq']}", axis=1)
test_df['enriched_content'] = test_df.apply(lambda row: f"{row['catalog_content']} ipq{row['ipq']}", axis=1)

# --- Target Transformation ---
y = np.log1p(train_df['price'].values)

# --- Feature Extraction ---
print("Extracting TF-IDF features...")
tfidf_model_path = "models/tfidf_full.pkl"
tfidf = TfidfVectorizer(max_features=40000, ngram_range=(1,2), stop_words='english', min_df=5)
if os.path.exists(tfidf_model_path):
    tfidf = joblib.load(tfidf_model_path)
    X_text_train = tfidf.transform(train_df['enriched_content'])
    X_text_test = tfidf.transform(test_df['enriched_content'])
else:
    X_text_train = tfidf.fit_transform(train_df['enriched_content'])
    X_text_test = tfidf.transform(test_df['enriched_content'])
    joblib.dump(tfidf, tfidf_model_path)

# --- Combine all features (Text + IPQ) ---
X_ipq_train = csr_matrix(train_df['ipq'].values.reshape(-1, 1))
X_ipq_test = csr_matrix(test_df['ipq'].values.reshape(-1, 1))

X_train = hstack([X_text_train, X_ipq_train])
X_test = hstack([X_text_test, X_ipq_test])

print(f"Combined Train shape: {X_train.shape}")
print(f"Combined Test shape: {X_test.shape}")

# --- 15-Fold Cross-Validation Training ---
# Changed to 15 folds
kf = KFold(n_splits=15, shuffle=True, random_state=42)
oof_preds = np.zeros(X_train.shape[0])
test_preds = np.zeros(X_test.shape[0])

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"\n-------- Fold {fold} --------")
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=128,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        objective="regression_l1"
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(100)]
    )
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits
    joblib.dump(model, f"models/lgbm_fold{fold}.pkl")

# --- Evaluate Full OOF SMAPE ---
oof_price = np.expm1(oof_preds)
full_cv_smape = smape(train_df['price'].values, oof_price)
print(f"\n✅ Full 15-Fold OOF SMAPE: {full_cv_smape:.2f}%")

# --- Test Submission ---
test_price_pred = np.expm1(test_preds)
test_price_pred = np.clip(test_price_pred, 0.01, None)
sub = pd.DataFrame({
    "sample_id": test_df["sample_id"],
    "price": test_price_pred
})
sub.to_csv("../outputs/test_out.csv", index=False)
print("\n✅ 15-Fold submission file saved: ../outputs/test_out.csv")
