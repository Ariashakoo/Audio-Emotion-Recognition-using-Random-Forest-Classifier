import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfccs

data = []
labels = []

#put the data path here
data_directory = r''  
print(f"Searching in directory: {data_directory}")

for file in os.listdir(data_directory):
    if file.endswith('.wav'): 
        file_path = os.path.join(data_directory, file)
        features = extract_features(file_path)
        label = file.split('_')[-1].replace('.wav', '')  
        data.append(features)
        labels.append(label)

if len(data) == 0 or len(labels) == 0:
    print("No data or labels were loaded. Please check your directory structure and audio files.")
    exit()  


df = pd.DataFrame(data)
df['label'] = labels


label_counts = df['label'].value_counts()
print("Label counts:\n", label_counts)  


X = df.drop('label', axis=1).values  
y = df['label'].values 

# Splitting the dataset into training and test sets (80% train, 20% test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


predicted_counts = pd.Series(y_pred).value_counts()
predicted_counts.plot(kind='bar')
plt.title('Predicted Distribution of Emotions')
plt.xlabel('Predicted Labels')
plt.ylabel('Counts')
plt.show()
