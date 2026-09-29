"""
House Price Prediction - Model Training & Evaluation
====================================================
This script:
1. Loads the processed train and test datasets.
2. Trains 3 regression models strictly on training data:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
3. Evaluates each model on the test data using MAE, MSE, RMSE, and R2.
4. Compares and logs test vs. train performance to assess overfitting.
5. Saves the trained models into `house_price_prediction/models/` using joblib.
6. Generates and saves comparison charts and Actual vs. Predicted plots.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Aesthetic plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.titlesize": 15,
    "figure.titleweight": "bold"
})


def train_and_evaluate(train_path: str, test_path: str, models_dir: str, figures_dir: str):
    print("=" * 65)
    print("STEP 4: MODEL TRAINING AND EVALUATION")
    print("=" * 65)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load Processed Datasets
    print(f"\n[1] Loading processed datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]

    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    print(f"    Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"    Test shape : X={X_test.shape}, y={y_test.shape}")
    print(f"    Features ({X_train.shape[1]}): {list(X_train.columns)}")

    # 2. Define Models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []
    predictions_dict = {}
    saved_paths = {}

    print("\n[2] Training and Evaluating Models...")

    for name, model in models.items():
        print(f"\n---> Training: {name}")
        # Train ONLY on training set
        model.fit(X_train, y_train)

        # Predictions on Train & Test
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        predictions_dict[name] = y_test_pred

        # Calculate Metrics on Test Data
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)

        # Train Metrics (for detecting overfitting)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)

        results.append({
            "Model": name,
            "Test MAE (INR)": test_mae,
            "Test RMSE (INR)": test_rmse,
            "Test MSE": test_mse,
            "Test R2": test_r2,
            "Train R2": train_r2,
            "Train RMSE (INR)": train_rmse
        })

        # Save model artifact
        safe_filename = name.lower().replace(" ", "_") + ".joblib"
        save_path = os.path.join(models_dir, safe_filename)
        joblib.dump(model, save_path)
        saved_paths[name] = save_path
        print(f"     Saved model to: {save_path}")
        print(f"     Test R2: {test_r2:.4f} | Test RMSE: INR {test_rmse:,.2f} | Test MAE: INR {test_mae:,.2f}")

    # 3. Create Comparison DataFrame
    results_df = pd.DataFrame(results)

    # 4. Visualization 1: Metrics Comparison Bar Charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R2 Score comparison (higher is better)
    sns.barplot(data=results_df, x="Model", y="Test R2", ax=axes[0], palette="Blues_d")
    axes[0].set_title("Model Comparison: R2 Score (Higher is Better)")
    axes[0].set_ylabel("R2 Score (0.0 to 1.0)")
    axes[0].set_ylim(0, 1.05)
    for i, v in enumerate(results_df["Test R2"]):
        axes[0].text(i, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")

    # RMSE comparison in Lakhs (lower is better)
    results_df["Test RMSE (Lakhs)"] = results_df["Test RMSE (INR)"] / 1e5
    sns.barplot(data=results_df, x="Model", y="Test RMSE (Lakhs)", ax=axes[1], palette="Reds_d")
    axes[1].set_title("Model Comparison: Test RMSE in Lakhs INR (Lower is Better)")
    axes[1].set_ylabel("Root Mean Squared Error (Lakhs INR)")
    for i, v in enumerate(results_df["Test RMSE (Lakhs)"]):
        axes[1].text(i, v + 0.3, f"INR {v:.2f}L", ha="center", fontweight="bold")

    plt.tight_layout()
    comp_fig_path = os.path.join(figures_dir, "model_metrics_comparison.png")
    plt.savefig(comp_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[3] Saved Model Metrics Comparison plot to: {comp_fig_path}")

    # 5. Visualization 2: Actual vs Predicted Scatter Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Actual vs. Predicted House Prices on Test Set (in Lakhs INR)", fontsize=16, fontweight="bold")

    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for i, (name, y_pred) in enumerate(predictions_dict.items()):
        ax = axes[i]
        ax.scatter(y_test / 1e5, y_pred / 1e5, alpha=0.5, color=colors[i], edgecolors="none", s=35)
        # 45-degree perfect prediction line
        min_val = min(y_test.min(), y_pred.min()) / 1e5
        max_val = max(y_test.max(), y_pred.max()) / 1e5
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1.8, label="Ideal Prediction (y = x)")
        
        r2_val = results_df.loc[results_df["Model"] == name, "Test R2"].values[0]
        rmse_val = results_df.loc[results_df["Model"] == name, "Test RMSE (Lakhs)"].values[0]
        ax.set_title(f"{name}\nR2 = {r2_val:.4f} | RMSE = INR {rmse_val:.2f}L")
        ax.set_xlabel("Actual Price (Lakhs INR)")
        ax.set_ylabel("Predicted Price (Lakhs INR)")
        ax.legend(loc="upper left")

    plt.tight_layout()
    act_pred_fig_path = os.path.join(figures_dir, "actual_vs_predicted_comparison.png")
    plt.savefig(act_pred_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[4] Saved Actual vs. Predicted plot to: {act_pred_fig_path}")

    # 6. Print Summary Table
    print("\n" + "=" * 65)
    print("FINAL MODEL EVALUATION RESULTS TABLE")
    print("=" * 65)
    display_df = results_df[["Model", "Test R2", "Test RMSE (INR)", "Test MAE (INR)", "Train R2"]].copy()
    display_df["Test RMSE (Lakhs)"] = (display_df["Test RMSE (INR)"] / 1e5).round(2)
    display_df["Test MAE (Lakhs)"] = (display_df["Test MAE (INR)"] / 1e5).round(2)
    display_df["Test R2"] = display_df["Test R2"].round(4)
    display_df["Train R2"] = display_df["Train R2"].round(4)

    print(display_df.to_string(index=False))

    # Identify Best Model
    best_idx = results_df["Test R2"].idxmax()
    best_model_name = results_df.loc[best_idx, "Model"]
    best_r2 = results_df.loc[best_idx, "Test R2"]
    best_rmse_lakhs = results_df.loc[best_idx, "Test RMSE (Lakhs)"]

    print("\n" + "=" * 65)
    print(f"BEST PERFORMING MODEL ON TEST DATA: {best_model_name}")
    print(f"R2 Score: {best_r2:.4f} | RMSE: INR {best_rmse_lakhs:.2f} Lakhs")
    print("=" * 65)


if __name__ == "__main__":
    train_csv = os.path.join("house_price_prediction", "data", "train_processed.csv")
    test_csv = os.path.join("house_price_prediction", "data", "test_processed.csv")
    models_out = os.path.join("house_price_prediction", "models")
    figures_out = os.path.join("house_price_prediction", "reports", "figures")
    train_and_evaluate(train_csv, test_csv, models_out, figures_out)
