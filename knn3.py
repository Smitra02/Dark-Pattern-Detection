import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Load the CSV file
df = pd.read_csv('D:/Projects/Hackathon/dark-patterns.csv')

# Preprocessing
df = df.dropna(subset=['Pattern String', 'Pattern Type'])
df['Pattern String'] = df['Pattern String'].astype(str)

# Tokenize and convert text data to numerical form (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Pattern String'])

# Label encoding for target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Pattern Type'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the k-NN model
knn_model = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn_model.fit(X_train, y_train)

# Save the model
joblib.dump(knn_model, 'knn_model.joblib')

# Evaluate the model
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')

# User Input Prediction
user_input = input("Enter a text for prediction: ")
user_input_transformed = vectorizer.transform([user_input])

# Map predicted class labels to "yes" or "no"
user_prediction_label = knn_model.predict(user_input_transformed)
user_prediction_output = "yes" if user_prediction_label[0] == 1 else "no"

print(f'Prediction for user input: {user_prediction_output}')
