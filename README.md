# 🛡️ Insurance Risk Analysis & Modeling Project

## 1. Project Overview

This project focuses on **Exploratory Data Analysis (EDA)** and **pre-modeling data preparation** for a motor insurance portfolio. The primary goal is to understand the distribution, quality, and relationships of key variables—particularly financial, vehicle, and geographical features—to support the development of **Generalized Linear Models (GLMs)** for predicting claims frequency and severity (Loss Ratio).

## 2. Data Summary

The dataset covers a large motor insurance policy book, predominantly from South Africa, with data spanning late 2014 to mid-2015.

| Category | Key Findings |
| :--- | :--- |
| **Financial/Target** | Heavily **zero-inflated** and **right-skewed** (Totalclaims, Totalpremium, LossRatio). Most policies have zero claims, but a few high-severity claims create extreme outliers. |
| **Vehicle Specs** | Highly **skewed** towards standard vehicles (e.g., 4 Cylinders, 4 Doors, Sedan/Hatchback 'B/S' Bodytype, and TOYOTA make). |
| **Geographical** | Policies are concentrated in economic hubs: **Gauteng** has the highest policy count and the highest average **Loss Ratio** (highest risk environment). |
| **Time Trend** | The policy book showed rapid, **exponential growth** in both Premium and Claims from Oct 2014 to Aug 2015, with a massive claims volatility spike in **April 2015**. |

## 3. Key Variables and Distribution Insights

### A. Highly Informative/Volatile Variables

| Variable | Distribution Pattern | Implication for Modeling |
| :--- | :--- | :--- |
| **Totalclaims** | Extreme **Zero-Inflation** | Requires specialized **Zero-Inflated** models (e.g., Tweedie GLM or Zero-Inflated Poisson). |
| **LossRatio** | Extreme **Zero-Inflation** & Right-Skewness | Highly volatile target variable. Modeling must account for the heavy mass at zero. |
| **Kilowatts** | **Bimodal** (Peaks at 75 kW and 110 kW) | Indicates two distinct vehicle populations (low vs. high power) that should be analyzed separately or captured as a feature. |
| **Underwrittencoverid** | Extreme **Multimodality** | Suggests a categorical structure (standard policy packages) rather than continuous values. |

### B. Low-Variance / Near-Constant Variables

These variables are largely non-informative due to a single dominant category.

* **Non-Informative:** $\text{Language}$ (100% English), $\text{Country}$ (100% South Africa), $\text{Termfrequency}$ (100% Monthly).
* **Near-Constant:** $\text{Cylinders}$ (Dominated by 4), $\text{Isvatregistered}$ (Dominated by False), $\text{Maritalstatus}$ (Dominated by 'Not specified').

## 4. Analytical Findings

* **Correlation:** There is **no significant linear correlation** ($r \approx 0.00$) between $\text{Totalpremium}$, $\text{Totalclaims}$, $\text{Suminsured}$, and $\text{Customvalueestimate}$.
* **Claims Severity:** While the median claim is near zero across all provinces, **Gauteng** and **Western Cape** show the **highest potential for extreme, high-cost claims** (outliers on the box plot).
* **Risk Ranking (Loss Ratio):** **Gauteng** has the highest average Loss Ratio, making it the least profitable region by this metric, while **Free State** and **Northern Cape** are the lowest-risk.
* **Claim Cost by Make:** **SUZUKI** vehicles have the highest average claim amount, indicating high-severity repair/replacement costs.

## 5. Next Steps (Pre-Modeling)

1.  **Outlier Management:** Investigate and transform extreme outliers in all financial columns ($\text{Totalclaims}$, $\text{Suminsured}$).
2.  **Zero-Handling:** Prepare target variables ($\text{Totalclaims}$, $\text{LossRatio}$) for specialized zero-inflated modeling techniques.
3.  **Feature Engineering:** Consolidate low-frequency categories in highly skewed categorical variables (e.g., $\text{Bodytype}$, $\text{Make}$, $\text{Province}$) into an 'Other' bin.
4.  **GLM Development:** Begin fitting models using $\text{Totalclaims}$ and $\text{Totalpremium}$ to model claim frequency and severity components of the Loss Ratio.

## 6. Dependencies

(List programming languages, libraries, and tools used for the analysis, e.g.)

* **Language:** Python 3.x
* **Core Libraries:** pandas, numpy
* **Visualization:** seaborn, matplotlib
* **Modeling:** scikit-learn, statsmodels