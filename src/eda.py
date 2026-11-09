import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from .data import TARGET, RISK_ORDER, EXPECTED_CATEGORIES


#data quality
def missing_summary(df):
    missing = df.isna().sum()
    summary = pd.DataFrame({
        "missing values": missing,
        "percentage": missing / len(df) * 100,
    })
    return summary[summary["missing values"] > 0]


def count_duplicates(df):
    return df.duplicated().sum()


def check_categories(df):
    for col, expected in EXPECTED_CATEGORIES.items():
        observed = set(df[col].dropna().unique())
        unexpected = observed - set(expected)
        print(f"{col}: unexpected values = {unexpected if unexpected else 'None'}")


def detect_outliers_iqr(df, columns):
    rows = []
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append([col, lower, upper, n_outliers])
    return pd.DataFrame(rows, columns=["column", "lower", "upper", "n_outliers"])


#target
def class_balance(df):
    counts = df[TARGET].value_counts().reindex(RISK_ORDER)
    ratio = counts.max() / counts.min()
    return counts, ratio


def plot_target(df):
    counts, _ = class_balance(df)
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values)
    plt.title("Distribution of Mental_Health_Risk")
    plt.ylabel("Number of students")
    plt.show()


#features
def plot_boxplots(df, columns):
    for col in columns:
        plt.figure(figsize=(8, 3))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()


def plot_histograms(df, columns):
    df[columns].hist(bins=30, figsize=(16, 12), edgecolor="black", grid=False)
    plt.tight_layout()
    plt.show()


def plot_year_of_study(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Year_of_Study")
    plt.title("Distribution of Year_of_Study")
    plt.show()


#relations with the target 
def plot_numeric_vs_target(df, columns):
    for col in columns:
        plt.figure(figsize=(6, 3))
        sns.boxplot(data=df, x=TARGET, y=col, order=RISK_ORDER)
        plt.title(f"{col} by risk level")
        plt.show()


def plot_categorical_vs_target(df, columns):
    for col in columns:
        table = pd.crosstab(df[col], df[TARGET], normalize="index")[RISK_ORDER] * 100
        table.plot.bar(stacked=True, figsize=(6, 3))
        plt.title(f"{col}: % of each risk level")
        plt.ylabel("%")
        plt.show()


def plot_correlation(df, numeric_cols):
    data = df[numeric_cols].copy()
    data["Risk_Ord"] = df[TARGET].map({"Low": 0, "Medium": 1, "High": 2})
    plt.figure(figsize=(11, 9))
    sns.heatmap(data.corr(method="spearman"), annot=True, fmt=".2f", cmap="RdBu_r")
    plt.title("Spearman correlation matrix")
    plt.show()


#which variables are most linked to the risk?
def rank_numeric_features(df, numeric_cols):
    risk_ord = df[TARGET].map({"Low": 0, "Medium": 1, "High": 2})
    corr = {col: abs(df[col].corr(risk_ord, method="spearman")) for col in numeric_cols}
    return pd.Series(corr).sort_values(ascending=False)   # high = strong link


def rank_categorical_features(df, categorical_cols):
    pvalues = {}
    for col in categorical_cols:
        table = pd.crosstab(df[col], df[TARGET])
        pvalues[col] = chi2_contingency(table)[1]         
    return pd.Series(pvalues).sort_values()             