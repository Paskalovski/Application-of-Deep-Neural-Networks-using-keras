import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# print(dataset)

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values


## Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)

labelencoder_X_2 = LabelEncoder()
X[:,4] = labelencoder_X_2.fit_transform(X[:,4])

X = X[:,1:]

np.set_printoptions(threshold=sys.maxsize)
# print(X)


## Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


## Initializing the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()

# Input layer and first hidden layer
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu',input_dim=11))

# Second hidden layer
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))

# Output layer
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer="adam",loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


## Predictign the test set
y_pred = classifier.predict(X_test)

# Setting a threshold (for users that are likely to leave the bank)
y_pred = (y_pred > 0.5)
# print(y_pred)


## Testing the model on new samples

''' Predicting sample with the following information
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of products: 2
Has credit card: Yes
Is active member: Yes
Estimated salary: 50000
'''

sample = sc.transform(np.array([[0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000 ]]))
new_prediction = classifier.predict(sample)
new_prediction = (new_prediction > 0.5)

# prediction result
# print(new_prediction)


## Visualizing the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# print(cm)


## Evaluating the model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


# Building the classifier with the same model architecture used above
def model_classifier():
    classifier = Sequential()
    # Input layer and first hidden layer
    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=11))
    # Second hidden layer
    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
    # Output layer
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = model_classifier, batch_size=10, epochs=100)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
# print(accuracies)

mean = accuracies.mean()
# print("Mean classifier value:", mean)
variance = accuracies.std()
# print("model variance:", variance)


## Parameter Tuning
from sklearn.model_selection import GridSearchCV

def model_classifier(optimizer):
    classifier = Sequential()
    # Input layer and first hidden layer
    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=11))
    # Second hidden layer
    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
    # Output layer
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return classifier


classifier = KerasClassifier(build_fn = model_classifier())
parameters = {'batch_size':[25,30],
              'nb_epoch':[100,500],
              'optimizer':['adam','rmsprop']}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
print("best parameters:",best_parameters)

best_accuracy = grid_search.best_score_
print("best_accuracy:",best_accuracy)
