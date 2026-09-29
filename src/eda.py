"""
House Price Prediction - Exploratory Data Analysis (EDA)
========================================================
This script generates and saves key EDA visualizations:
1. Price Distribution (Histogram + KDE & Boxplot)
2. Area vs. Price (Scatter plot with trendline)
3. Rooms vs. Price (Boxplot by room count)
4. Age vs. Price (Scatter plot with regression trendline)
5. Location vs. Price (Boxplot & Average price per city)
6. Correlation Heatmap (Pearson correlation of numerical features)
7. Unified 6-panel comprehensive dashboard
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold"
})


def run_eda(raw_data_path: str, output_dir: str):
    print("=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(raw_data_path)
    print(f"Loaded dataset from {raw_data_path} ({df.shape[0]} rows, {df.shape[1]} columns)")

    # 1. Price Distribution (Hist + Boxplot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with KDE
    sns.histplot(df["price"] / 1e5, kde=True, ax=axes[0], color="#2b5c8f", bins=30)
    axes[0].set_title("Distribution of House Prices")
    axes[0].set_xlabel("Price (in Lakhs INR / 100,000s)")
    axes[0].set_ylabel("Number of Houses (Frequency)")
    mean_val = df["price"].mean() / 1e5
    median_val = df["price"].median() / 1e5
    axes[0].axvline(mean_val, color="red", linestyle="--", label=f"Mean: INR {mean_val:.1f}L")
    axes[0].axvline(median_val, color="green", linestyle="-", label=f"Median: INR {median_val:.1f}L")
    axes[0].legend()

    # Boxplot for Outliers
    sns.boxplot(x=df["price"] / 1e5, ax=axes[1], color="#79addc", flierprops={"marker": "o", "markersize": 4, "alpha": 0.5})
    axes[1].set_title("Price Spread & Outlier Detection")
    axes[1].set_xlabel("Price (in Lakhs INR / 100,000s)")

    plt.tight_layout()
    p1 = os.path.join(output_dir, "1_price_distribution.png")
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Price Distribution plot to: {p1}")

    # 2. Area vs Price (Scatter Plot + Trendline)
    plt.figure(figsize=(9, 6))
    valid_area = df.dropna(subset=["area_sqft", "price"])
    sns.regplot(
        data=valid_area, 
        x="area_sqft", 
        y=df.loc[valid_area.index, "price"] / 1e5,
        scatter_kws={"alpha": 0.4, "color": "#1f77b4", "s": 30},
        line_kws={"color": "#d62728", "linewidth": 2}
    )
    plt.title("Living Area (sqft) vs. House Price")
    plt.xlabel("Living Area (Square Feet)")
    plt.ylabel("Price (in Lakhs INR / 100,000s)")
    p2 = os.path.join(output_dir, "2_area_vs_price.png")
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Area vs Price plot to: {p2}")

    # 3. Rooms vs Price (Boxplot)
    plt.figure(figsize=(9, 6))
    valid_rooms = df.dropna(subset=["rooms", "price"])
    sns.boxplot(
        data=valid_rooms,
        x="rooms",
        y=df.loc[valid_rooms.index, "price"] / 1e5,
        palette="Blues_r"
    )
    plt.title("Number of Rooms vs. House Price")
    plt.xlabel("Number of Rooms (BHK / Rooms)")
    plt.ylabel("Price (in Lakhs INR / 100,000s)")
    p3 = os.path.join(output_dir, "3_rooms_vs_price.png")
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Rooms vs Price plot to: {p3}")

    # 4. Age vs Price (Scatter with Trendline)
    plt.figure(figsize=(9, 6))
    valid_age = df.dropna(subset=["age", "price"])
    sns.regplot(
        data=valid_age,
        x="age",
        y=df.loc[valid_age.index, "price"] / 1e5,
        scatter_kws={"alpha": 0.4, "color": "#2ca02c", "s": 30},
        line_kws={"color": "#ff7f0e", "linewidth": 2}
    )
    plt.title("Property Age (Years) vs. House Price")
    plt.xlabel("Property Age (Years)")
    plt.ylabel("Price (in Lakhs INR / 100,000s)")
    p4 = os.path.join(output_dir, "4_age_vs_price.png")
    plt.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Age vs Price plot to: {p4}")

    # 5. Location vs Price (Ordered Boxplot by Median Price)
    plt.figure(figsize=(11, 6))
    loc_order = df.groupby("location")["price"].median().sort_values(ascending=False).index
    sns.boxplot(
        data=df,
        x="location",
        y=df["price"] / 1e5,
        order=loc_order,
        palette="crest"
    )
    plt.title("Location vs. House Price (Sorted by Median Price)")
    plt.xlabel("City / Location")
    plt.ylabel("Price (in Lakhs INR / 100,000s)")
    plt.xticks(rotation=20)
    p5 = os.path.join(output_dir, "5_location_vs_price.png")
    plt.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Location vs Price plot to: {p5}")

    # 6. Correlation Heatmap (Numerical Variables)
    plt.figure(figsize=(8, 6))
    num_df = df[["rooms", "area_sqft", "age", "price"]].copy()
    corr_matrix = num_df.corr()

    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        linewidths=1, 
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Heatmap of Numerical Features")
    p6 = os.path.join(output_dir, "6_correlation_heatmap.png")
    plt.savefig(p6, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Correlation Heatmap to: {p6}")

    # 7. Unified 6-Panel Dashboard for Overview
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Comprehensive Housing Dataset EDA Dashboard", fontsize=18, fontweight="bold")

    # [0, 0] Price Distribution
    sns.histplot(df["price"] / 1e5, kde=True, ax=axes[0, 0], color="#2b5c8f", bins=25)
    axes[0, 0].set_title("1. Price Distribution (in Lakhs INR)")
    axes[0, 0].set_xlabel("Price (Lakhs INR)")
    axes[0, 0].set_ylabel("Frequency")

    # [0, 1] Area vs Price
    sns.regplot(
        data=valid_area, x="area_sqft", y=df.loc[valid_area.index, "price"] / 1e5,
        ax=axes[0, 1], scatter_kws={"alpha": 0.3, "color": "#1f77b4", "s": 15},
        line_kws={"color": "#d62728", "linewidth": 2}
    )
    axes[0, 1].set_title("2. Living Area vs. Price")
    axes[0, 1].set_xlabel("Area (sqft)")
    axes[0, 1].set_ylabel("Price (Lakhs INR)")

    # [0, 2] Rooms vs Price
    sns.boxplot(data=valid_rooms, x="rooms", y=df.loc[valid_rooms.index, "price"] / 1e5, ax=axes[0, 2], palette="Blues_r")
    axes[0, 2].set_title("3. Rooms vs. Price")
    axes[0, 2].set_xlabel("Rooms")
    axes[0, 2].set_ylabel("Price (Lakhs INR)")

    # [1, 0] Age vs Price
    sns.regplot(
        data=valid_age, x="age", y=df.loc[valid_age.index, "price"] / 1e5,
        ax=axes[1, 0], scatter_kws={"alpha": 0.3, "color": "#2ca02c", "s": 15},
        line_kws={"color": "#ff7f0e", "linewidth": 2}
    )
    axes[1, 0].set_title("4. Property Age vs. Price")
    axes[1, 0].set_xlabel("Age (Years)")
    axes[1, 0].set_ylabel("Price (Lakhs INR)")

    # [1, 1] Location vs Price
    sns.boxplot(data=df, x="location", y=df["price"] / 1e5, order=loc_order, ax=axes[1, 1], palette="crest")
    axes[1, 1].set_title("5. Location vs. Price (Ranked)")
    axes[1, 1].set_xlabel("City")
    axes[1, 1].set_ylabel("Price (Lakhs INR)")
    axes[1, 1].tick_params(axis="x", rotation=30)

    # [1, 2] Correlation Matrix
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1, 2], square=True, cbar=False)
    axes[1, 2].set_title("6. Pearson Correlation Heatmap")

    plt.tight_layout()
    p_dash = os.path.join(output_dir, "eda_comprehensive_dashboard.png")
    plt.savefig(p_dash, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Unified EDA Dashboard to: {p_dash}")

    # Output Numerical Insights for reporting
    print("\n" + "=" * 60)
    print("KEY STATISTICAL INSIGHTS:")
    print("=" * 60)
    print(f"Mean Price   : INR {df['price'].mean():,.2f} ({df['price'].mean()/1e5:.2f} Lakhs)")
    print(f"Median Price : INR {df['price'].median():,.2f} ({df['price'].median()/1e5:.2f} Lakhs)")
    print(f"Min Price    : INR {df['price'].min():,.2f} ({df['price'].min()/1e5:.2f} Lakhs)")
    print(f"Max Price    : INR {df['price'].max():,.2f} ({df['price'].max()/1e5:.2f} Lakhs)")
    print("\nMedian Price by City (Highest to Lowest):")
    for city, med_p in df.groupby("location")["price"].median().sort_values(ascending=False).items():
        print(f"  - {city:<12}: INR {med_p:,.2f} ({med_p/1e5:.2f} Lakhs)")

    print("\nCorrelation with Price:")
    for col, corr in corr_matrix["price"].items():
        print(f"  - {col:<12}: {corr:+.3f}")


if __name__ == "__main__":
    raw_path = os.path.join("house_price_prediction", "data", "housing.csv")
    fig_dir = os.path.join("house_price_prediction", "reports", "figures")
    run_eda(raw_path, fig_dir)
