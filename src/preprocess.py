"""
House Price Prediction - Data Preprocessing Pipeline
=====================================================
This script performs step-by-step data preprocessing on housing.csv:
1. Loads raw data without modifying the source file.
2. Checks for duplicates and missing values.
3. Separates features (X) and target (y: price).
4. Splits into Train (80%) and Test (20%) sets to avoid data leakage.
5. Imputes missing numerical values (median) & missing amenities (empty string).
6. Encodes categorical location (One-Hot Encoding) and multi-label amenities (Binary Flags).
7. Scales numerical features using StandardScaler.
8. Exports processed training and test datasets.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def preprocess_data(data_path: str, output_dir: str, random_state: int = 42):
    print("=" * 60)
    print("STEP 2: DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Load Raw Dataset
    print(f"\n[1] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"    Raw data shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Duplicate Detection
    duplicates = df.duplicated().sum()
    print(f"\n[2] Checking for duplicate rows: Found {duplicates} duplicate(s)")
    if duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"    Dataset shape after dropping duplicates: {df.shape}")

    # 3. Missing Value Summary (Before Preprocessing)
    print("\n[3] Missing Values in Raw Dataset:")
    for col, count in df.isnull().sum().items():
        pct = (count / len(df)) * 100
        print(f"    - {col:<12}: {count:>4} missing ({pct:>5.1f}%)")

    # 4. Separate Features (X) and Target (y)
    print("\n[4] Separating Features (X) and Target (y = price)")
    X = df.drop(columns=["price"])
    y = df["price"]

    # 5. Train-Test Split (80% Train, 20% Test) BEFORE any transformation
    # To prevent Data Leakage, all statistics/encoders must be fitted ONLY on training data!
    print("\n[5] Splitting data into 80% Train and 20% Test sets (random_state=42)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    print(f"    X_train: {X_train.shape[0]} samples")
    print(f"    X_test : {X_test.shape[0]} samples")

    # 6. Preprocessing Strategy & Definitions
    num_cols = ["rooms", "area_sqft", "age"]
    cat_cols = ["location"]
    multilabel_col = "amenities"

    # --- A. Numerical Imputation (Median) ---
    print("\n[6a] Imputing Numerical Features (rooms, area_sqft, age) using Median")
    num_imputer = SimpleImputer(strategy="median")
    X_train_num_imp = num_imputer.fit_transform(X_train[num_cols])
    X_test_num_imp = num_imputer.transform(X_test[num_cols])

    for col, med in zip(num_cols, num_imputer.statistics_):
        print(f"     * Learned Train Median for '{col}': {med:.1f}")

    # --- B. Numerical Scaling (StandardScaler) ---
    print("\n[6b] Scaling Numerical Features using StandardScaler (Mean=0, Std=1)")
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num_imp)
    X_test_num_scaled = scaler.transform(X_test_num_imp)
    num_feature_names = num_cols

    # --- C. Categorical Encoding (location - OneHotEncoder) ---
    print("\n[6c] Encoding 'location' with One-Hot Encoding")
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train_loc_encoded = ohe.fit_transform(X_train[cat_cols])
    X_test_loc_encoded = ohe.transform(X_test[cat_cols])
    loc_feature_names = [f"loc_{cat}" for cat in ohe.categories_[0]]
    print(f"     * Unique locations identified ({len(loc_feature_names)}): {list(ohe.categories_[0])}")

    # --- D. Multi-label Text Handling (amenities - Binary Flags) ---
    print("\n[6d] Processing Multi-Label 'amenities' into individual binary indicator columns")
    # Fill missing amenities with empty string
    train_amenities_clean = X_train[multilabel_col].fillna("").astype(str)
    test_amenities_clean = X_test[multilabel_col].fillna("").astype(str)

    # Extract all unique individual amenities present in training set
    all_amenities = set()
    for item in train_amenities_clean:
        if item.strip():
            tags = [tag.strip() for tag in item.split(",") if tag.strip()]
            all_amenities.update(tags)
    sorted_amenities = sorted(list(all_amenities))
    print(f"     * Identified distinct amenities ({len(sorted_amenities)}): {sorted_amenities}")

    # Create binary indicator matrices
    def extract_amenity_flags(series, amenity_list):
        flags = []
        for val in series:
            val_tags = set(t.strip() for t in val.split(",") if t.strip()) if val else set()
            row = [1.0 if amen in val_tags else 0.0 for amen in amenity_list]
            flags.append(row)
        return np.array(flags)

    X_train_amenities = extract_amenity_flags(train_amenities_clean, sorted_amenities)
    X_test_amenities = extract_amenity_flags(test_amenities_clean, sorted_amenities)
    amenity_feature_names = [f"amenity_{a}" for a in sorted_amenities]

    # --- E. Combine All Processed Features ---
    all_feature_names = num_feature_names + loc_feature_names + amenity_feature_names

    X_train_processed_matrix = np.hstack([
        X_train_num_scaled,
        X_train_loc_encoded,
        X_train_amenities
    ])

    X_test_processed_matrix = np.hstack([
        X_test_num_scaled,
        X_test_loc_encoded,
        X_test_amenities
    ])

    X_train_processed = pd.DataFrame(X_train_processed_matrix, columns=all_feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed_matrix, columns=all_feature_names, index=X_test.index)

    # Combine with Target for complete processed datasets
    train_processed_df = X_train_processed.copy()
    train_processed_df["price"] = y_train.values

    test_processed_df = X_test_processed.copy()
    test_processed_df["price"] = y_test.values

    # 7. Save Processed Datasets (without modifying raw data)
    os.makedirs(output_dir, exist_ok=True)
    train_out_path = os.path.join(output_dir, "train_processed.csv")
    test_out_path = os.path.join(output_dir, "test_processed.csv")

    train_processed_df.to_csv(train_out_path, index=False)
    test_processed_df.to_csv(test_out_path, index=False)

    print(f"\n[7] Saved processed training data to: {train_out_path}")
    print(f"    Saved processed testing data to : {test_out_path}")

    # 8. Save Preprocessor Artifact
    import joblib
    models_dir = os.path.join("house_price_prediction", "models")
    os.makedirs(models_dir, exist_ok=True)
    preprocessor_bundle = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_imputer": num_imputer,
        "scaler": scaler,
        "ohe": ohe,
        "sorted_amenities": sorted_amenities,
        "feature_names": all_feature_names
    }
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    joblib.dump(preprocessor_bundle, preprocessor_path)
    print(f"    Saved preprocessor pipeline to  : {preprocessor_path}")

    print("\n" + "=" * 60)
    print("SUMMARY OF PROCESSED FEATURE STRUCTURE")
    print("=" * 60)
    print(f"Total Features: {len(all_feature_names)}")
    print(f"Feature List  : {all_feature_names}")
    print(f"\nProcessed Train Shape (X): {X_train_processed.shape}")
    print(f"Processed Test Shape  (X): {X_test_processed.shape}")
    print("\nFirst 5 Rows of Processed Training Features:")
    print(X_train_processed.head().to_string())

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": all_feature_names,
        "num_features": num_feature_names,
        "loc_features": loc_feature_names,
        "amenity_features": amenity_feature_names,
        "train_shape": X_train_processed.shape,
        "test_shape": X_test_processed.shape
    }


if __name__ == "__main__":
    raw_path = os.path.join("house_price_prediction", "data", "housing.csv")
    out_dir = os.path.join("house_price_prediction", "data")
    preprocess_data(raw_path, out_dir)
