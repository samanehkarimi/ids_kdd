import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

from keras.models import Sequential
from keras.layers import Dense, Flatten

# Load the data
x_train = pd.read_csv('F://finalproject//Intrusion-Detection-Systems-master//kddtrain.csv', header=None)
x_test = pd.read_csv('F://finalproject//Intrusion-Detection-Systems-master//kddtest.csv', header=None)

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
mlp_model = Sequential([
    Flatten(input_shape=(x_train.shape[1],)),
    Dense(256, activation='sigmoid'),
    Dense(256, activation='sigmoid'),
    Dense(128, activation='sigmoid'),
    Dense(128, activation='sigmoid'),
    Dense(10, activation='softmax')
])

mlp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

mlp_model.fit(x_train, trainlabel, epochs=20, batch_size=16, validation_split=0.2)

# Make predictions using MLP model on the test data
mlp_predicted_probabilities = mlp_model.predict(x_test)
mlp_predicted_labels = np.argmax(mlp_predicted_probabilities, axis=1)

# Calculate evaluation metrics for MLP model
mlp_accuracy = accuracy_score(testlabel, mlp_predicted_labels)
mlp_precision = precision_score(testlabel, mlp_predicted_labels, average='weighted')
mlp_recall = recall_score(testlabel, mlp_predicted_labels, average='weighted')
mlp_f1 = f1_score(testlabel, mlp_predicted_labels, average='weighted')

# Print the evaluation metrics for MLP model
print("MLP Model Evaluation Metrics:")
print("Accuracy:", "%.3f" % mlp_accuracy)
print("Precision:", "%.3f" % mlp_precision)
print("Recall:", "%.3f" % mlp_recall)
print("F1-score:", "%.3f" % mlp_f1)

# Create the SVM model
svm_model = SVC()
svm_model.fit(x_train, trainlabel)

# Make predictions using SVM model on the test data
svm_predicted_labels = svm_model.predict(x_test)

# Calculate evaluation metrics for SVM model
svm_accuracy = accuracy_score(testlabel, svm_predicted_labels)
svm_precision = precision_score(testlabel, svm_predicted_labels, average='weighted')
svm_recall = recall_score(testlabel, svm_predicted_labels, average='weighted')
svm_f1 = f1_score(testlabel, svm_predicted_labels, average='weighted')

# Print the evaluation metrics for SVM model
print("\nSVM Model Evaluation Metrics:")
print("Accuracy:", "%.3f" % svm_accuracy)
print("Precision:", "%.3f" % svm_precision)
print("Recall:", "%.3f" % svm_recall)
print("F1-score:", "%.3f" % svm_f1)