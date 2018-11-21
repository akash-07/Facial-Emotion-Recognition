from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pickle

model = Sequential()
model.add(BatchNormalization(input_shape= (2278,)))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))   
model.add(Activation('relu'))
model.add(Dense(7))
model.add(Activation('softmax'))

optimizer = Adam(lr = 0.0001)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

f = open('data_landmarks.pickle', 'rb')
data = pickle.load(f)

X_train = np.array(data['X_train'])
X_test = np.array(data['X_test'])
X_val = np.array(data['X_val'])
y_train = to_categorical(data['y_train'], num_classes = 7)
y_test = to_categorical(data['y_test'], num_classes = 7)
y_val = to_categorical(data['y_val'], num_classes = 7)

model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 100, batch_size = 64)

score = model.evaluate(X_test, y_test, batch_size = 100)
