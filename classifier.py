import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

def load_and_prepare():
    df = pd.read_csv("data/mtsamples.csv")
    df["medical_specialty"] = df["medical_specialty"].str.strip()

    target_specialties = [
        "Surgery",
        "Cardiovascular / Pulmonary",
        "Orthopedic",
        "Radiology",
        "Neurology",
    ]

    df = df[df["medical_specialty"].isin(target_specialties)].copy()
    df = df.dropna(subset = ['transcription'])
    df = df[['transcription','medical_specialty']].copy()
    df.columns = ['text', 'label']

    return train_test_split(
        df['text'],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

def build_and_evaluate():
    X_train, X_test, y_train, y_test = load_and_prepare()

    # build a pipeline: TF-IDF vectorizer + Logistic Regression
    # class_weight = 'balanced' handles the class imbalance we noticed
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1,2),
            stop_words='english',
            sublinear_tf=True,  #dampens the effect of very frequent terms
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight='balanced', #handles Surgery vs Neurology imbalance
            random_state=42,
        ))
    ])

    # train
    print("Training classifier...")
    pipeline.fit(X_train, y_train)

    # evaluate
    y_pred = pipeline.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    print("---Confusion Matrix---")
    print(confusion_matrix(y_test, y_pred))

    # save the trained model to disk
    os.makedirs("models", exist_ok=True)
    with open("models/classifier.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print("\nModel saved to models/classifier.pkl")

    # save test set for the LLM evaluation step
    test_df = pd.DataFrame({"text": X_test, "label": y_test, "predicted": y_pred})
    test_df.to_csv("data/test_results.csv", index=False)
    print("Test results saved to data/test_results.csv")

    return pipeline, test_df

if __name__ == "__main__":
    build_and_evaluate()