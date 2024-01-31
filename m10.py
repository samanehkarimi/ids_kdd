import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from keras.models import Sequential
from keras.layers import Dense, Flatten

# Load the data
x_train = pd.read_csv('F://final_project//Intrusion-Detection-Systems-master//kddtrain.csv', header=None)
x_test = pd.read_csv('F://final_project//Intrusion-Detection-Systems-master//kddtest.csv', header=None)

# Prepare the data
X = x_train.iloc[:, 1:42]
Y = x_train.iloc[:, 0]
C = x_test.iloc[:, 0]
T = x_test.iloc[:, 1:42]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

x_train = np.array(trainX)
trainlabel = np.array(Y)

x_test = np.array(testT)
testlabel = np.array(C)

# Create the MLP model
model = Sequential([
    Flatten(input_shape=(x_train.shape[1],)),
    Dense(256, activation='sigmoid'),
    Dense(256, activation='sigmoid'),
    Dense(128, activation='sigmoid'),
    Dense(128, activation='sigmoid'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, trainlabel, epochs=20, batch_size=16, validation_split=0.2)

# Make predictions on the test data
predicted_probabilities = model.predict(x_test)
predicted_labels = np.argmax(predicted_probabilities, axis=1)

# Calculate evaluation metrics
accuracy = accuracy_score(testlabel, predicted_labels)
precision = precision_score(testlabel, predicted_labels, average='weighted')
recall = recall_score(testlabel, predicted_labels, average='weighted')
f1 = f1_score(testlabel, predicted_labels, average='weighted')

# Print the evaluation metrics
print("Accuracy:", "%.3f" %accuracy)
print("Precision:", "%.3f" %precision)
print("Recall:", "%.3f" %recall)
print("F1-score:", "%.3f" %f1)