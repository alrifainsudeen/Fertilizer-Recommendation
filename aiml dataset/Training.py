# Fertilizer_Recommender_Interactive.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample

# -----------------------
# Step 1: Load Dataset
# -----------------------
df = pd.read_csv("data_core.csv")

if "Temparature" in df.columns:
    df.rename(columns={"Temparature": "Temperature"}, inplace=True)

# -----------------------
# Step 2: Feature Engineering
# -----------------------
if 'Moisture' in df.columns:
    df['Temp_Moisture'] = df['Temperature'] * df['Moisture']

# -----------------------
# Step 3: Balance Classes
# -----------------------
max_count = df['Fertilizer Name'].value_counts().max()
balanced_df = pd.DataFrame()

for label in df['Fertilizer Name'].unique():
    subset = df[df['Fertilizer Name'] == label]
    subset_resampled = resample(subset, replace=True, n_samples=max_count, random_state=42)
    balanced_df = pd.concat([balanced_df, subset_resampled])

# -----------------------
# Step 4: Features & Target
# -----------------------
X = balanced_df.drop("Fertilizer Name", axis=1)
y = balanced_df["Fertilizer Name"]

le_target = LabelEncoder()
y = le_target.fit_transform(y)

# -----------------------
# Step 5: Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# Step 6: Identify Categorical & Numeric Columns
# -----------------------
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Separate high and low cardinality for encoding
high_card_cols = [col for col in categorical_cols if X[col].nunique() > 10]
low_card_cols = [col for col in categorical_cols if X[col].nunique() <= 10]

# Label encode high-cardinality categorical columns
for col in high_card_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# -----------------------
# Step 7: Preprocessing Pipeline
# -----------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('low_card_cat', OneHotEncoder(handle_unknown='ignore'), low_card_cols)
    ]
)

# -----------------------
# Step 8: Model Pipeline
# -----------------------
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# -----------------------
# Step 9: Train Model
# -----------------------
model.fit(X_train, y_train)

# -----------------------
# Step 10: Evaluate Model
# -----------------------
y_pred = model.predict(X_test)

print("Model Evaluation on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# -----------------------
# Step 11: Interactive Prediction
# -----------------------
def predict_new_sample():
    print("\nEnter new sample details:")
    sample = {}
    for col in X.columns:
        while True:
            value = input(f"{col}: ")
            try:
                if col in numeric_cols:
                    sample[col] = float(value)
                else:
                    sample[col] = value
                break
            except:
                print("Invalid input, please enter a number for numeric columns.")
    
    # Feature engineering for new sample
    if 'Temp_Moisture' in X.columns:
        sample['Temp_Moisture'] = sample['Temperature'] * sample.get('Moisture', 0)
    
    sample_df = pd.DataFrame([sample])
    
    # Label encode high-cardinality columns
    for col in high_card_cols:
        le = LabelEncoder()
        le.fit(X[col])
        sample_df[col] = le.transform(sample_df[col])
    
    prediction = model.predict(sample_df)
    fertilizer_name = le_target.inverse_transform(prediction)
    print(f"\nPredicted Fertilizer: {fertilizer_name[0]}")

# Run interactive prediction
predict_new_sample()
