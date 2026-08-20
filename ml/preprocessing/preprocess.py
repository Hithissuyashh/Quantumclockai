import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler


RAW_DATA = "data/raw/clock_dataset_v3.csv"

OUTPUT_DIR = "data/processed"

TRAIN_FILE = "clock_train_v3_scaled.csv"
VAL_FILE = "clock_val_v3_scaled.csv"
TEST_FILE = "clock_test_v3_scaled.csv"

SCALER_FILE = "feature_scaler_v3.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("V3 CHRONOLOGICAL PREPROCESSING")
print("=" * 70)

# -------------------------------------------------
# Load raw dataset
# -------------------------------------------------

df = pd.read_csv(RAW_DATA)

print(f"Raw samples : {len(df)}")

# -------------------------------------------------
# Feature configuration
# -------------------------------------------------

feature_columns = [
    "temperature",
    "magnetic",
    "laser_power",
    "pressure",
    "humidity",

    "white",
    "random_walk",
    "flicker",
    "laser_phase",
    "qpn",

    "temperature_shift",
    "zeeman",
    "blackbody",
    "aging",
    "total_noise",

    "measured_offset",
    "estimated_offset",
    "servo_correction",

    "excitation_probability_plus",
    "excitation_probability_minus",
]

target_column = "true_detuning"
time_column = "time"

# -------------------------------------------------
# Chronological split
# -------------------------------------------------

n = len(df)

train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print("\nChronological split:")
print(f"Train : {len(train_df)}")
print(f"Val   : {len(val_df)}")
print(f"Test  : {len(test_df)}")

# -------------------------------------------------
# Fit scaler ONLY on training data
# -------------------------------------------------

scaler = StandardScaler()

scaler.fit(train_df[feature_columns])

# -------------------------------------------------
# Transform each split
# -------------------------------------------------

def process_split(split_df):

    X = scaler.transform(
        split_df[feature_columns]
    )

    processed = pd.DataFrame(
        X,
        columns=feature_columns
    )

    processed[target_column] = (
        split_df[target_column].values
    )

    processed[time_column] = (
        split_df[time_column].values
    )

    return processed


train_processed = process_split(train_df)
val_processed = process_split(val_df)
test_processed = process_split(test_df)

# -------------------------------------------------
# Save datasets
# -------------------------------------------------

train_path = os.path.join(
    OUTPUT_DIR,
    TRAIN_FILE
)

val_path = os.path.join(
    OUTPUT_DIR,
    VAL_FILE
)

test_path = os.path.join(
    OUTPUT_DIR,
    TEST_FILE
)

train_processed.to_csv(
    train_path,
    index=False
)

val_processed.to_csv(
    val_path,
    index=False
)

test_processed.to_csv(
    test_path,
    index=False
)

# -------------------------------------------------
# Save scaler
# -------------------------------------------------

scaler_path = os.path.join(
    OUTPUT_DIR,
    SCALER_FILE
)

joblib.dump(
    scaler,
    scaler_path
)

# -------------------------------------------------
# Report
# -------------------------------------------------

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(f"Train dataset : {train_path}")
print(f"Validation    : {val_path}")
print(f"Test dataset  : {test_path}")
print(f"Scaler        : {scaler_path}")

print("\nFeature count :", len(feature_columns))
print("Target        :", target_column)

print("\nShapes:")
print("Train :", train_processed.shape)
print("Val   :", val_processed.shape)
print("Test  :", test_processed.shape)