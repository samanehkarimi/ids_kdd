import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)

traindata = pd.read_csv('F://final_project//Intrusion-Detection-Systems-master//kddtrain.csv', header=None)
testdata = pd.read_csv('F://final_project//Intrusion-Detection-Systems-master//kddtest.csv', header=None)

X = traindata.iloc[:,1:42]
Y = traindata.iloc[:,0]
C = testdata.iloc[:,0]
T = testdata.iloc[:,1:42]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)


traindata = np.array(trainX)
trainlabel = np.array(Y)

testdata = np.array(testT)
testlabel = np.array(C)



#traindata = X_train
#testdata = X_test
#trainlabel = y_train
#testlabel = y_test

print("-----------------------------------------LR---------------------------------")
model = LogisticRegression()
model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/expected.txt', expected, fmt='%01d')
predicted = model.predict(testdata)
proba = model.predict_proba(testdata)

np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/predictedlabelLR.txt', predicted, fmt='%01d')
np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/predictedprobaLR.txt', proba)

y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)

# fit a Naive Bayes model to the data
print("-----------------------------------------NB---------------------------------")
model = GaussianNB()
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
proba = model.predict_proba(testdata)

np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/predictedlabelNB.txt', predicted, fmt='%01d')
np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/predictedprobaNB.txt', proba)

y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)

# fit a k-nearest neighbor model to the data
# print("-----------------------------------------KNN---------------------------------")
# model = KNeighborsClassifier()
# model.fit(traindata, trainlabel)
# print(model)
# # make predictions
# expected = testlabel
# predicted = model.predict(testdata)
# proba = model.predict_proba(testdata)


# np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/predictedlabelKNN.txt', predicted, fmt='%01d')
# np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/predictedprobaKNN.txt', proba)

# # summarize the fit of the model

# y_train1 = expected
# y_pred = predicted
# accuracy = accuracy_score(y_train1, y_pred)
# recall = recall_score(y_train1, y_pred , average="binary")
# precision = precision_score(y_train1, y_pred , average="binary")
# f1 = f1_score(y_train1, y_pred, average="binary")


# print("----------------------------------------------")
# print("accuracy")
# print("%.3f" %accuracy)
# print("precision")
# print("%.3f" %precision)
# print("racall")
# print("%.3f" %recall)
# print("f1score")
# print("%.3f" %f1)

print("-----------------------------------------DT---------------------------------")

model = DecisionTreeClassifier()
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
proba = model.predict_proba(testdata)

np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/predictedlabelDT.txt', predicted, fmt='%01d')
np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/predictedprobaDT.txt', proba)
# summarize the fit of the model

y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")


print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)


print("-----------------------------------------Adaboost---------------------------------")

model = AdaBoostClassifier(n_estimators=100)
model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
predicted = model.predict(testdata)
proba = model.predict_proba(testdata)

np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/predictedlabelAB.txt', predicted, fmt='%01d')
np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/predictedprobaAB.txt', proba)
# summarize the fit of the model

y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")


print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)

model = RandomForestClassifier(n_estimators=100)
model = model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
predicted = model.predict(testdata)
proba = model.predict_proba(testdata)
np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/predictedlabelRF.txt', predicted, fmt='%01d')
np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/predictedprobaRF.txt', proba)

# summarize the fit of the model

print("--------------------------------------RF--------------------------------------")

y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")


print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)


model = svm.SVC(kernel='rbf',probability=True)
model = model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
predicted = model.predict(testdata)
proba = model.predict_proba(testdata)
np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/predictedlabelSVM-rbf.txt', predicted, fmt='%01d')
np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/predictedprobaSVM-rbf.txt', proba)

print("--------------------------------------SVMrbf--------------------------------------")
y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)


model = svm.SVC(kernel='linear', C=1000,probability=True)
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
proba = model.predict_proba(testdata)

np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/predictedlabelSVM-linear.txt', predicted, fmt='%01d')
np.savetxt('F:/final_project/Intrusion-Detection-Systems-master/classical/predictedprobaSVM-linear.txt', proba)

# summarize the fit of the model
print("--------------------------------------SVM linear--------------------------------------")
y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)



