import pandas as pd

BINARY_MAP = {
    "Yes": 1,
    "No": 0,
    "Male": 1,
    "Female": 0
}

BINARY_COLUMNS = [
    "Gender",
    "Senior_Citizen",
    "Partner",
    "Dependents",
    "Phone_Service",
    "Paperless_Billing"
]

ONEHOT_COLUMNS = [
    "Multiple_Lines",
    "Internet_Service",
    "Online_Security",
    "Online_Backup",
    "Device_Protection",
    "Tech_Support",
    "Streaming_TV",
    "Streaming_Movies",
    "Contract",
    "Payment_Method"
]

def preprocess_input(data: dict, model_columns: list) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # binary encode
    for col in BINARY_COLUMNS:
        df[col] = df[col].map(BINARY_MAP)

    # one-hot
    df = pd.get_dummies(df, columns=ONEHOT_COLUMNS, drop_first=False)

    # kolon hizalama (ALTIN KURAL)
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]
    return df
