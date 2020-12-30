

from load_data_pretrain import load_data_4
import numpy as np
from keras.callbacks import History
from keras import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input,  Dense, concatenate
from keras import regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(6, activation='relu', kernel_initializer='normal', input_dim=6))#Second  Hidden Layer
classifier.add(Dense(6, activation='relu', kernel_initializer='normal'))#Output Layer
classifier.add(Dense(6, activation='relu', kernel_initializer='normal'))#Output Layer

classifier.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

data, label = load_data_4()
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.4, random_state=42)


#Fitting the data to the training dataset
classifier.fit(train_data, train_label, batch_size=100, epochs=10)
eval_model = classifier.evaluate(train_data, train_label)


loss, acc = classifier.evaluate(test_data, test_label)
print("Testing Accuracy:  {:.4f}".format(acc))

# ## tn, fp, fn, tp
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(test_label, y_pred, labels=[0,1]).ravel()
# print(cm)





### lstm version ###