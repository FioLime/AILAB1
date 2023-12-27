import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

input_file = 'income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
  for line in f.readlines():
    if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
      break
    if '?' in line:
      continue

data = line[:-1].split(', ')

if data[-1] == '<=50K' and count_class1 < max_datapoints:
  X.append(data)
  count_class1 += 1
if data[-1] == '>50K' and count_class2 < max_datapoints:
  X.append(data)
  count_class2 += 1

X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
  if item.isdigit():
    X_encoded[:, i] = X[:, i]
  else:
    label_encoder.append(preprocessing.LabelEncoder())
    X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy using RBF kernel:", accuracy)