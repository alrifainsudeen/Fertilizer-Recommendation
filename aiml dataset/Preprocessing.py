import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("data_core.csv")

# Fix column name
df.rename(columns={"Temparature": "Temperature"}, inplace=True)

# Features and target
X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Define categorical & numerical columns
categorical_cols = ["Soil Type", "Crop Type"]
numerical_cols = ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]

# Preprocessing: scale numerical + one-hot encode categorical
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Final pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# Apply preprocessing
X_processed = pipeline.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

print("Preprocessing complete!")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
