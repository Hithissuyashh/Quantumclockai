import os

import pandas as pd

# ------------------------------------------
# Load Dataset
# ------------------------------------------

DATASET = "data/clock_dataset_v2.csv"

df = pd.read_csv(DATASET)

print("=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

print(df.info())

print()

print("=" * 80)
print("FIRST FIVE ROWS")
print("=" * 80)

print(df.head())

print()

print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print(df.describe().T)

print()

print("=" * 80)
print("MISSING VALUES")
print("=" * 80)

print(df.isnull().sum())

print()

print("=" * 80)
print("DUPLICATE ROWS")
print("=" * 80)

print(df.duplicated().sum())

print()

print("=" * 80)
print("DATASET SHAPE")
print("=" * 80)

print(df.shape)
