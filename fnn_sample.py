"""
Updated on November 21, 2023
@author: Aditya Meka
Added preprocessing, splitting of testing and training datasets,
and selection of Scenario.
"""
########################################
# Part 1 -Pre Processing Dataset

# To load a dataset file in Python, you can use Pandas. Import pandas using the line below
import pandas as pd
# Import numpy to perform operations on the dataset
import numpy as np

# We can provide input for which scenario we want to run.
# Accepts a, b or c as input
ScenarioA = ['Train-a1-a3', 'Test-a2-a4']
ScenarioB = ['Training-a1-a2', 'Test-a1']
ScenarioC = ['Training-a1-a2', 'Test-a1-a2-a3']

while 1:
    Scenario = input ('Please enter the scenario you wish to run - either a, b or c:')

    if Scenario.lower() == 'a':
        TrainingData = ScenarioA[0]
        TestingData  = ScenarioA[1]
        break
    elif Scenario.lower() == 'b':
        TrainingData = ScenarioB[0]
        TestingData  = ScenarioB[1]
        break
    elif Scenario.lower() == 'c':
        TrainingData = ScenarioC[0]
        TestingData  = ScenarioC[1]
        break

# Batch Size
BatchSize=10
# Epohe Size
NumEpoch=10

import data_preprocessor as dp
X_train, y_train = dp.get_processed_data(TrainingData+'.csv', './categoryMappings/', classType ='binary')
X_test,  y_test  = dp.get_processed_data(TestingData+'.csv',  './categoryMappings /', classType ='binary')


########################################
# Part 2: Building the FNN

# Importing the Keras libraries and packages
#import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()

# Adding the input layer and the first hidden layer, 6 nodes, input_dim specifies the number of variables
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(X_train[0])))

# Adding the second hidden layer, 6 nodes
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer, 1 node, 
# sigmoid on the output layer is to ensure the network output is between 0 and 1
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN, 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# Train the model so that it learns a good (or good enough) mapping of rows of input data to the output classification.
# add verbose=0 to turn off the progress report during the training
# To run the whole training dataset as one Batch, assign batch size: BatchSize=X_train.shape[0]
classifierHistory = classifier.fit(X_train, y_train, batch_size = BatchSize, epochs = NumEpoch)

# evaluate the keras model for the provided model and dataset
loss, accuracy = classifier.evaluate(X_train, y_train)
print('Print the loss and the accuracy of the model on the dataset')
print('Loss [0,1]: %.4f' % (loss), 'Accuracy [0,1]: %.4f' % (accuracy))

########################################
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.9)   # y_pred is 0 if less than 0.9 or equal to 0.9, y_pred is 1 if it is greater than 0.9
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Print the Confusion Matrix:')
print('[ TN, FP ]')
print('[ FN, TP ]=')
print(cm)

########################################
# Part 4 - Visualizing the Plots


# Import matplot lib libraries for plotting the figures. 
import matplotlib.pyplot as plt

# You can plot the accuracy
print('Plot the accuracy')
# Keras 2.2.4 recognizes 'acc' and 2.3.1 recognizes 'accuracy'
plt.plot(classifierHistory.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('accuracy_sample.png')
plt.show()

# You can plot history for loss
print('Plot the loss')
plt.plot(classifierHistory.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('loss_sample.png')
plt.show()