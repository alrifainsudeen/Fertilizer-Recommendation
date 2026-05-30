import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils import resample
import joblib
import os
import warnings

# CORRECTED WARNING SUPPRESSION: Target the message text for better compatibility
warnings.filterwarnings('ignore', message='A value is trying to be set on a copy of a slice from a DataFrame.')

# Define file paths
DATA_PATH = "aiml dataset/data_core.csv"
MODEL_FILE = "fertilizer_model.joblib"
TARGET_ENCODER_FILE = "target_encoder.joblib"
HIGH_CARD_ENCODERS_FILE = "high_card_encoders.joblib"
FEATURE_COLUMNS_FILE = "feature_columns.joblib"
CATEGORY_DATA_FILE = "category_data.joblib"

print("--- Starting Model Training and Serialization ---")

if not os.path.exists(DATA_PATH):
    print(f"Error: '{DATA_PATH}' not found. Cannot train model.")
    exit()

df = pd.read_csv(DATA_PATH)
if "Temparature" in df.columns:
    df.rename(columns={"Temparature": "Temperature"}, inplace=True)

# Feature Engineering
if 'Moisture' in df.columns:
    df['Temp_Moisture'] = df['Temperature'] * df['Moisture']

# Balance Classes
max_count = df['Fertilizer Name'].value_counts().max()
balanced_df = pd.DataFrame()
for label in df['Fertilizer Name'].unique():
    subset = df[df['Fertilizer Name'] == label]
    subset_resampled = resample(subset, replace=True, n_samples=max_count, random_state=42)
    balanced_df = pd.concat([balanced_df, subset_resampled])

# Features & Target
X = balanced_df.drop("Fertilizer Name", axis=1)
y = balanced_df["Fertilizer Name"]

# Target Encoder 
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Split (Needed only to define training subset)
X_train, _, y_train, _ = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Identify Columns
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
high_card_cols = [col for col in categorical_cols if X[col].nunique() > 10]
low_card_cols = [col for col in categorical_cols if X[col].nunique() <= 10]

# High Card Label Encoders
high_card_encoders = {}
for col in high_card_cols:
    le = LabelEncoder()
    le.fit(X[col].astype(str)) 
    high_card_encoders[col] = le
    # Use .loc for clean modification
    X_train.loc[:, col] = le.transform(X_train[col].astype(str)) 

# Preprocessing Pipeline (Only scale/OHE)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('low_card_cat', OneHotEncoder(handle_unknown='ignore'), low_card_cols)
    ],
    remainder='passthrough'
)

# Final Model Pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

print("Training model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete!")

# --- SAVE ALL COMPONENTS ---
try:
    joblib.dump(model_pipeline, MODEL_FILE)
    joblib.dump(le_target, TARGET_ENCODER_FILE)
    joblib.dump(high_card_encoders, HIGH_CARD_ENCODERS_FILE)
    joblib.dump(X.columns.tolist(), FEATURE_COLUMNS_FILE)
    
    # Save categories for UI dropdowns
    category_data = {}
    if 'Soil Type' in X.columns: category_data['Soil Type'] = X['Soil Type'].astype(str).unique()
    if 'Crop Type' in X.columns: category_data['Crop Type'] = X['Crop Type'].astype(str).unique()
    joblib.dump(category_data, CATEGORY_DATA_FILE)
    
    print(f"\nModel and encoders successfully saved.")
    print(f"Run 'streamlit run app.py' now for fast prediction.")

except Exception as e:
    print(f"Error saving files: {e}")
