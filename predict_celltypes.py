"""
Purpose:UR Science Hackathon — Summer 2026


Task: Cell-type prediction for MERFISH spatial transcriptomics data.

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ---------------------------------------------------------------- load data
counts_train = pd.read_csv("data/counts_train.csv", index_col=0)
counts_test  = pd.read_csv("data/counts_test.csv",  index_col=0)
meta_train   = pd.read_csv("data/meta_train.csv",   index_col=0)
meta_test    = pd.read_csv("data/meta_test.csv",    index_col=0)

CATEGORICAL = ["Gender", "Mouse_ID", "AP_position", "Section_ID",
               "Region", "Excitatory_vs_Inhibitory", "Segment", "Datasets"]

# ------------------------------------------------ feature engineering
def build_features(counts, meta, cat_categories=None):

    c = counts.values.astype(float)
    lib = c.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0                          # guard against divide-by-zero
    lognorm = np.log1p(c / lib * np.median(lib)) # library-normalize + log

    feat = pd.DataFrame(lognorm, index=counts.index, columns=counts.columns)
    feat["volume"]   = meta["volume"].values
    feat["center_x"] = meta["center_x"].values
    feat["center_y"] = meta["center_y"].values

    learned = {}
    for col in CATEGORICAL:
        s = meta[col].astype("object")
        if cat_categories is None:
            catt = pd.Categorical(s)
            learned[col] = catt.categories
            feat[col] = catt.codes                # NA -> -1
        else:
            catt = pd.Categorical(s, categories=cat_categories[col])
            feat[col] = catt.codes                # unseen/NA -> -1
    return feat, learned

X_train, cat_cats = build_features(counts_train, meta_train)
X_test, _         = build_features(counts_test, meta_test, cat_categories=cat_cats)

# ------------------------------------------------ labels
le = LabelEncoder()
y_train = le.fit_transform(meta_train["MERFISH_cell_type_annotation"].values)
print(f"Training on {X_train.shape[0]} cells, {X_train.shape[1]} features, "
      f"{len(le.classes_)} cell types.")

# ------------------------------------------------ train on ALL labeled data (leaky? check values again)
model = xgb.XGBClassifier(
    n_estimators=800, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.6, tree_method="hist",
    n_jobs=-1, random_state=42, eval_metric="mlogloss",
)
model.fit(X_train.values, y_train)

# ------------------------------------------------ predict test set
pred_idx = model.predict(X_test.values)
pred_labels = le.inverse_transform(pred_idx)

# ------------------------------------------------ write submission
# Format matches the example file: header "Cell_ID,MERFISH_cell_type_annotation.y",
# rows in the SAME order as meta_test.csv.
out = pd.DataFrame({
    "Cell_ID": meta_test.index,
    "MERFISH_cell_type_annotation.y": pred_labels,
})
out.to_csv("prediction/prediction.csv", index=False)
print(f"Wrote prediction/prediction.csv with {len(out)} rows.")
print(out.head())
