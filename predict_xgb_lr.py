import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import xgboost as xgb


# ============================================================
# XGBOOST + LOGISTIC REGRESSION
# Hopefully Proofed for replacement / unseen test data
# Final blend is FROZEN at:
#   75% XGBoost
#   25% Logistic Regression
# ============================================================


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

counts_train = pd.read_csv(
    "data/counts_train.csv",
    index_col=0
)

counts_test = pd.read_csv(
    "data/counts_test.csv",
    index_col=0
)

meta_train = pd.read_csv(
    "data/meta_train.csv",
    index_col=0
)

meta_test = pd.read_csv(
    "data/meta_test.csv",
    index_col=0
)


# ------------------------------------------------------------
# CHECK ROW ALIGNMENT
# ------------------------------------------------------------

print(
    "Train IDs aligned:",
    counts_train.index.equals(meta_train.index)
)

print(
    "Test IDs aligned :",
    counts_test.index.equals(meta_test.index)
)


# ------------------------------------------------------------
# CATEGORICAL METADATA USED BY XGBOOST
# ------------------------------------------------------------

CATEGORICAL = [
    "Gender",
    "Mouse_ID",
    "AP_position",
    "Section_ID",
    "Region",
    "Excitatory_vs_Inhibitory",
    "Segment",
    "Datasets",
]


# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

def build_features(
    counts,
    meta,
    cat_categories=None,
    scale=None
):

    # ---------------- Gene counts ----------------

    c = counts.values.astype(float)

    lib = c.sum(
        axis=1,
        keepdims=True
    )

    # Avoid division by zero
    lib[lib == 0] = 1.0


    # --------------------------------------------------------
    # Learn normalization scale ONLY from training data
    # --------------------------------------------------------

    if scale is None:

        scale = np.median(lib)


    # Normalize + log transform
    genes = np.log1p(
        c / lib * scale
    )


    # --------------------------------------------------------
    # Full feature table for XGBoost
    # --------------------------------------------------------

    feat = pd.DataFrame(
        genes,
        index=counts.index,
        columns=counts.columns
    )

    feat["volume"] = meta["volume"].values
    feat["center_x"] = meta["center_x"].values
    feat["center_y"] = meta["center_y"].values


    # --------------------------------------------------------
    # Encode categorical metadata
    #
    # Training:
    #   learn category -> integer mapping
    #
    # Test:
    #   reuse training mapping
    #
    # Any new / unseen / missing category:
    #   encode as -1
    # --------------------------------------------------------

    learned_categories = {}

    for col in CATEGORICAL:

        s = meta[col].astype("object")


        if cat_categories is None:

            # Categories found in TRAIN only
            cats = list(
                pd.unique(
                    s.dropna()
                )
            )

            mapping = {
                value: i
                for i, value
                in enumerate(cats)
            }

            learned_categories[col] = mapping


        else:

            # Reuse mapping learned from TRAIN
            mapping = cat_categories[col]


        encoded = (
            s.map(mapping)
            .fillna(-1)
            .astype(int)
        )


        # Tell us if replacement data contains
        # something never seen during training
        if cat_categories is not None:

            unseen_mask = encoded == -1

            if unseen_mask.any():

                unseen_values = sorted(
                    set(
                        s[unseen_mask]
                        .dropna()
                        .astype(str)
                    )
                )

                print(
                    f"WARNING: unseen/missing category "
                    f"in {col}: {unseen_values}"
                )


        feat[col] = encoded


    return (
        feat,
        genes,
        learned_categories,
        scale
    )


# ------------------------------------------------------------
# BUILD TRAIN FEATURES
# ------------------------------------------------------------

X_train, G_train, cat_cats, train_scale = build_features(
    counts_train,
    meta_train
)


# ------------------------------------------------------------
# BUILD TEST FEATURES
#
# IMPORTANT:
# reuse TRAIN normalization scale
# reuse TRAIN category mappings
# ------------------------------------------------------------

X_test, G_test, _, _ = build_features(
    counts_test,
    meta_test,
    cat_categories=cat_cats,
    scale=train_scale
)


print(
    "\nTraining feature shape:",
    X_train.shape
)

print(
    "Test feature shape    :",
    X_test.shape
)


# ------------------------------------------------------------
# LABELS
# ------------------------------------------------------------

le = LabelEncoder()

y = le.fit_transform(
    meta_train[
        "MERFISH_cell_type_annotation"
    ].values
)

print(
    "Number of cell types:",
    len(le.classes_)
)


# ------------------------------------------------------------
# VALIDATION SPLIT
#
# This is only a diagnostic.
#
# FINAL weights are NOT chosen from this anymore.
# They are already frozen at 0.75 / 0.25.
# ------------------------------------------------------------

itr, ival = train_test_split(
    np.arange(len(y)),
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# XGBOOST FACTORY
# ------------------------------------------------------------

def make_xgb():

    return xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.6,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        eval_metric="mlogloss"
    )


# ------------------------------------------------------------
# VALIDATION — XGBOOST
# ------------------------------------------------------------

print(
    "\nRunning validation..."
)

xgb_model = make_xgb()

xgb_model.fit(
    X_train.values[itr],
    y[itr]
)

p_xgb = xgb_model.predict_proba(
    X_train.values[ival]
)


# ------------------------------------------------------------
# VALIDATION — LOGISTIC REGRESSION
# ------------------------------------------------------------

scaler = StandardScaler()

Gtr = scaler.fit_transform(
    G_train[itr]
)

Gval = scaler.transform(
    G_train[ival]
)

lr_model = LogisticRegression(
    max_iter=2000
)

lr_model.fit(
    Gtr,
    y[itr]
)

p_lr = lr_model.predict_proba(
    Gval
)


# ------------------------------------------------------------
# INDIVIDUAL VALIDATION SCORES
# ------------------------------------------------------------

xgb_acc = accuracy_score(
    y[ival],
    p_xgb.argmax(axis=1)
)

lr_acc = accuracy_score(
    y[ival],
    p_lr.argmax(axis=1)
)


# ------------------------------------------------------------
# FROZEN BLEND VALIDATION SCORE
# ------------------------------------------------------------

XGB_WEIGHT = 0.75
LR_WEIGHT = 0.25


val_probs = (
    XGB_WEIGHT * p_xgb
    +
    LR_WEIGHT * p_lr
)

blend_acc = accuracy_score(
    y[ival],
    val_probs.argmax(axis=1)
)


print(
    "\nXGBoost validation:",
    round(xgb_acc, 4)
)

print(
    "LR validation      :",
    round(lr_acc, 4)
)

print(
    "Frozen 75/25 blend :",
    round(blend_acc, 4)
)


print(
    "\nFinal frozen weights:"
)

print(
    "XGBoost:",
    XGB_WEIGHT
)

print(
    "LR      :",
    LR_WEIGHT
)


# ============================================================
# FINAL MODELS
# Train on ALL labeled cells
# ============================================================

print(
    "\nTraining final XGBoost "
    "on all labeled cells..."
)

xgb_final = make_xgb()

xgb_final.fit(
    X_train.values,
    y
)

test_p_xgb = xgb_final.predict_proba(
    X_test.values
)


# ------------------------------------------------------------
# FINAL LOGISTIC REGRESSION
# ------------------------------------------------------------

print(
    "Training final Logistic Regression "
    "on all labeled cells..."
)

final_scaler = StandardScaler()

G_train_scaled = final_scaler.fit_transform(
    G_train
)

G_test_scaled = final_scaler.transform(
    G_test
)

lr_final = LogisticRegression(
    max_iter=2000
)

lr_final.fit(
    G_train_scaled,
    y
)

test_p_lr = lr_final.predict_proba(
    G_test_scaled
)


# ------------------------------------------------------------
# FINAL FROZEN BLEND
# ------------------------------------------------------------

test_probs = (
    XGB_WEIGHT * test_p_xgb
    +
    LR_WEIGHT * test_p_lr
)


pred_numbers = test_probs.argmax(
    axis=1
)

pred_labels = le.inverse_transform(
    pred_numbers
)


# ------------------------------------------------------------
# WRITE SUBMISSION
# ------------------------------------------------------------

out = pd.DataFrame({
    "Cell_ID":
        meta_test.index,

    "MERFISH_cell_type_annotation.y":
        pred_labels
})


out.to_csv(
    "prediction/prediction.csv",
    index=False
)


# ------------------------------------------------------------
# FINAL SAFETY CHECKS
# ------------------------------------------------------------

print(
    "\nWrote prediction/prediction.csv"
)

print(
    "Rows:",
    len(out)
)

print(
    "Columns:",
    out.columns.tolist()
)

print(
    "Missing predictions:",
    out[
        "MERFISH_cell_type_annotation.y"
    ].isna().sum()
)

print(
    "Cell types predicted:",
    out[
        "MERFISH_cell_type_annotation.y"
    ].nunique(),
    "of",
    len(le.classes_)
)

print()

print(
    out.head()
)