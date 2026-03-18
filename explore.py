import pandas as pd

df = pd.read_csv("data/mtsamples.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst Row:")
print(df.iloc[0])
print("\nMedical specialties and counts:")
print(df["medical_specialty"].value_counts())