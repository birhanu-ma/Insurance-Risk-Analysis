import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set(style="whitegrid")

class EDAVisualizer:
    def __init__(self, df):
        """Initialize the visualizer with a DataFrame."""
        self.df = df.copy()

        # Convert date column
        if "Transactionmonth" in self.df.columns:
            self.df["Transactionmonth"] = pd.to_datetime(self.df["Transactionmonth"], errors="coerce")

        # Create Loss Ratio Column
        if "Totalclaims" in self.df.columns and "Totalpremium" in self.df.columns:
            self.df["LossRatio"] = self.df["Totalclaims"] / self.df["Totalpremium"].replace(0, np.nan)

        # Columns to exclude from categorical plotting
        self.exclude_categorical = ["Vehicleintrodate", "Capitaloutstanding", "Model"]

    # ---------------- 🔹 UNIVARIATE ANALYSIS ---------------- #
    def plot_numeric(self, cols=None, bins=50, kde=True):
        """Plot histograms for numeric columns."""
        if cols is None:
            cols = self.df.select_dtypes(include=["int64", "float64"]).columns

        for col in cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[col], bins=bins, kde=kde)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col); plt.ylabel("Frequency")
            plt.show()

    def plot_categorical(self, cols=None, top_n=20):
        """Plot bar charts for categorical columns."""
        if cols is None:
            cols = [c for c in self.df.select_dtypes(include=["object", "bool"]).columns
                    if c not in self.exclude_categorical]

        for col in cols:
            plt.figure(figsize=(10, 4))
            self.df[col].value_counts().head(top_n).plot(kind="bar")
            plt.title(f"Distribution of {col}")
            plt.xlabel(col); plt.ylabel("Count")
            plt.show()

    # ---------------- 🔹 BIVARIATE / MULTIVARIATE ANALYSIS ---------------- #
    def bivariate_analysis(self):
        """Scatterplot + Correlation Heatmap + Claims by Province."""
        
        # Scatter Plot Premium vs Claims
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=self.df, x="Totalpremium", y="Totalclaims", alpha=0.4)
        plt.title("Total Premium vs Total Claims")
        plt.xlabel("Total Premium"); plt.ylabel("Total Claims")
        plt.show()

        # Correlation Matrix
        num_cols = ["Totalpremium", "Totalclaims", "Customvalueestimate", "Suminsured"]
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix of Key Financial Fields")
        plt.show()

        # Claims by Province
        plt.figure(figsize=(12, 5))
        sns.boxplot(data=self.df, x="Province", y="Totalclaims")
        plt.title("Claims Severity by Province")
        plt.xticks(rotation=45)
        plt.show()

    # ---------------- 🔹 GEOGRAPHIC + TEMPORAL TRENDS ---------------- #
    def geographic_and_temporal(self):
        """Monthly Trend + Premium vs Claims by Postal Code."""
        # Monthly Total Premium & Claims Trend
        monthly = self.df.groupby("Transactionmonth")[["Totalpremium", "Totalclaims"]].sum()

        plt.figure(figsize=(12, 5))
        monthly.plot(marker="o")
        plt.title("Monthly Trend of Total Premium & Total Claims")
        plt.ylabel("Amount")
        plt.show()

        # Premium vs Claims by Top 20 Zip Codes
        top_zip = self.df["Postalcode"].value_counts().head(20).index
        zip_data = self.df[self.df["Postalcode"].isin(top_zip)]

        plt.figure(figsize=(10,6))
        sns.scatterplot(data=zip_data, x="Totalpremium", y="Totalclaims",
                        hue="Postalcode", alpha=0.7)
        plt.title("Claims vs Premium Across Top 20 Postal Codes")
        plt.show()

    # ---------------- 🔹 OUTLIER ANALYSIS ---------------- #
    def detect_outliers(self):
        """Boxplot for numerical risk-related columns."""
        num_cols = ["Totalpremium", "Totalclaims", "Customvalueestimate", "Suminsured"]

        plt.figure(figsize=(12,6))
        sns.boxplot(data=self.df[num_cols])
        plt.title("Outlier Analysis for Financial Columns")
        plt.xticks(rotation=45)
        plt.show()

    # ---------------- 🔹 BUSINESS INSIGHT KPI VISUALS ---------------- #
    def business_insight_plots(self):
        """Loss Ratio by province + Top risky vehicle makes."""
        
        # Loss Ratio by Province
        group = self.df.groupby("Province")["LossRatio"].mean().sort_values(ascending=False)

        plt.figure(figsize=(12, 5))
        group.plot(kind="bar")
        plt.title("Average Loss Ratio by Province")
        plt.ylabel("Loss Ratio")
        plt.show()

        # Top high-risk vehicle makes
        top_risk = self.df.groupby("Make")["Totalclaims"].mean().nlargest(10)

        plt.figure(figsize=(10,5))
        top_risk.plot(kind="bar")
        plt.title("Top 10 Vehicle Makes by Avg Claim Amount")
        plt.ylabel("Avg Claims")
        plt.show()
