import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from kagglehub import dataset_download


path = dataset_download("uciml/breast-cancer-wisconsin-data")
csv_path = os.path.join(path, "data.csv")

df = pd.read_csv(csv_path)

df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])


selected_features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]

X = df[selected_features]
y = df['diagnosis']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/classifier.pkl")
joblib.dump(confusion_matrix(y_test, model.predict(X_test)), "model/conf_matrix.pkl")

print(f"Model trained. Accuracy: {model.score(X_test, y_test) * 100:.2f}%")
