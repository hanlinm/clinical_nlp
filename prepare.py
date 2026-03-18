import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare():
    df = pd.read_csv("data/mtsamples.csv")

    # strip whitespace from specialty labels
    df["medical_specialty"] = df["medical_specialty"].str.strip()

    # keep only the 5 most clinically distinct specialties
    target_specialties = [
        "Surgery",
        "Cardiovascular / Pulmonary",
        "Orthopedic",
        "Radiology",
        "Neurology",
    ]

    df = df[df["medical_specialty"].isin(target_specialties)].copy()

    # drop rows with missing transcriptions
    df = df.dropna(subset = ['transcription'])

    # use transcription as our input text
    df = df[["transcription", "medical_specialty"]].copy()
    df.columns = ["text", "label"]

    print(f"Dataset size after filtering: {len(df)} rows")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    # split into train and test sets
    # stratify=True ensures each split has proportional class representation
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state = 42,
        stratify=df["label"]
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    load_and_prepare()