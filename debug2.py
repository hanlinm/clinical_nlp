import pandas as pd

test_df = pd.read_csv("data/test_results.csv")
print("=== TEST RESULTS ===")
print(f"Total rows: {len(test_df)}")
print(f"Columns: {test_df.columns.tolist()}")
print(f"\nOverall classifier accuracy: {(test_df['label'] == test_df['predicted']).mean():.1%}")
print(f"\nTrue label distribution:\n{test_df['label'].value_counts()}")
print(f"\nPredicted label distribution:\n{test_df['predicted'].value_counts()}")
print(f"\nFirst 5 rows:")
print(test_df[['label', 'predicted']].head())