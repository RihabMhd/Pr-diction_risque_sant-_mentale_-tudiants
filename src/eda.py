from .data import TARGET, RISK_ORDER, EXPECTED_CATEGORIES


def audit_data(df, numeric_cols):
    """data-quality audit used by the Airflow DAG. Returns a JSON-serializable dict"""
    counts = df[TARGET].value_counts().reindex(RISK_ORDER, fill_value=0)
    outliers = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers[col] = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())

    return {
        "shape": list(df.shape),
        "missing": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "duplicates": int(df.duplicated().sum()),
        "unexpected_categories": {
            col: sorted(set(df[col].dropna()) - set(exp))
            for col, exp in EXPECTED_CATEGORIES.items()
            if set(df[col].dropna()) - set(exp)
        },
        "outliers_iqr": outliers,
        "class_counts": {k: int(v) for k, v in counts.items()},
        "imbalance_ratio": float(counts.max() / counts[counts > 0].min()),
    }