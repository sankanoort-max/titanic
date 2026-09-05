import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("titanic.csv")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)


# ==========================================
# 2. SELECT FEATURES
# ==========================================

df = df[
    [
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]
]


# ==========================================
# 3. HANDLE MISSING VALUES
# ==========================================

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

print("Missing values handled")


# ==========================================
# 4. CREATE ENCODERS
# ==========================================

sex_encoder = LabelEncoder()
embarked_encoder = LabelEncoder()


# ==========================================
# 5. ENCODE CATEGORICAL FEATURES
# ==========================================

df["Sex"] = sex_encoder.fit_transform(df["Sex"])

df["Embarked"] = embarked_encoder.fit_transform(
    df["Embarked"]
)


print("Sex classes:", sex_encoder.classes_)
print("Embarked classes:", embarked_encoder.classes_)


# ==========================================
# 6. SAVE ENCODERS
# ==========================================

with open("sex_encoder.pkl", "wb") as f:
    pickle.dump(sex_encoder, f)

with open("embarked_encoder.pkl", "wb") as f:
    pickle.dump(embarked_encoder, f)

print("Encoders saved successfully")


# ==========================================
# 7. CREATE X AND Y
# ==========================================

X = df[
    [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]
]

y = df["Survived"]


# ==========================================
# 8. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ==========================================
# 9. CREATE MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ==========================================
# 10. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

print("Model training completed")


# ==========================================
# 11. TEST MODEL
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print()
print("==========================================")
print("MODEL PERFORMANCE")
print("==========================================")
print("Accuracy:", accuracy)

print()
print("Classification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 12. SAVE TRAINED MODEL
# ==========================================

with open("titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print()
print("==========================================")
print("MODEL SAVED")
print("==========================================")
print("titanic_model.pkl")
print("sex_encoder.pkl")
print("embarked_encoder.pkl")