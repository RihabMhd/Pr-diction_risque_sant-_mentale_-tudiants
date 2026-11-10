from sklearn.model_selection import train_test_split

from .data import TARGET, RISK_ORDER, EXPECTED_CATEGORIES

RANDOM_STATE = 42


def prepare_data(df):
    """clean the raw dataframe and return X_train, X_test, y_train, y_test (80/20, stratified)"""
    df = df.copy()

    for col in df.select_dtypes("object"):
        df[col] = df[col].str.strip()

    df = df.drop_duplicates()

    for col, values in EXPECTED_CATEGORIES.items():
        df.loc[~df[col].isin(values) & df[col].notna(), col] = None

    df = df[~(df["Screen_Time_Hours"] < df["Social_Media_Hours"])]

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    X = df.drop(columns=[TARGET, "Student_ID"])
    y = df[TARGET].map({r: i for i, r in enumerate(RISK_ORDER)})

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)