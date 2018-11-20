from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import pickle

def processString(s):
    s = s.split()
    img = np.zeros((48, 48))
    for i in range(48):
        for j in range(48):
            img[i][j] = s[48*i + j]
    img = np.repeat(img[:, :, np.newaxis], 3, axis = 2)
    return img


def getDataset():
    data_path = '../Data/fer2013/fer2013.csv'
    df = pd.read_csv(data_path)
    X_train = []; y_train = []
    X_test = []; y_test = []
    X_val = []; y_val = []
    for index, row in df.iterrows():
        if(row['Usage'] == 'Training'):
            if(len(X_train) < 1e+4):
                X_train.append(processString(row['pixels']))
                y_train.append(row['emotion'])
        elif(row['Usage'] == 'PublicTest'):
            X_val.append(processString(row['pixels']))
            y_val.append(row['emotion'])
        else:
            X_test.append(processString(row['pixels']))
            y_test.append(row['emotion'])
    
    #preprocess the data
    mean = np.mean(X_train, axis = 0)
    X_train -= mean
    X_test -= mean
    X_val -= mean
    data = {}
    data['X_train'] = np.array(X_train); data['y_train'] = np.array(y_train)
    data['X_val'] = np.array(X_val); data['y_val'] = np.array(y_val)
    data['X_test'] = np.array(X_test); data['y_test'] = np.array(y_test)
    return data

model = VGG16(weights = 'imagenet', include_top = False, pooling = 'max', input_shape = (48, 48, 3))

for layer in model.layers:
    layer.trainable = False

x = model.output
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.25)(x)
x = Dense(256, activation = 'relu')(x)
predictions = Dense(7, activation = 'softmax')(x)

model_final = Model(input = model.input, output = predictions)

sgd = SGD(lr = 0.001, decay = 1e-5, momentum = 0.9, nesterov = True)
model_final.compile(loss = 'categorical_crossentropy', 
                    optimizer = 'adam', metrics = ['accuracy'])    

# loading data
data = getDataset()
X_train = data['X_train'][:10]
X_test = data['X_test'][:10]
X_val = data['X_val'][:10]
y_train = np_utils.to_categorical(data['y_train'][:10])
y_test = np_utils.to_categorical(data['y_test'][:10])
y_val = np_utils.to_categorical(data['y_val'][:10])

# train the model
model_final.fit(X_train, y_train, validation_data = (X_val, y_val), batch_size = 50, epochs = 1)

# use the model for feature extraction
model_for_feat = Model(inputs = model_final.input, outputs = model_final.get_layer('dense_2').output)

f = open('data_landmarks.pickle', 'rb')
data_landmarks = pickle.load(f)

def extractFeatures(X_train, X_landmarks):
	X_train_feats = []
	for i in range(len(X_train)):
		print("{} / {}".format(i, len(X_train)))
		X = model_for_feat.predict(np.array([X_train[i]]))
		X = np.append(X, X_landmarks[i])
		X_train_feats.append(X)			
	return np.array(X_train_feats)

X_train_feats = extractFeatures(X_train, data_landmarks['X_train'])
X_test_feats = extractFeatures(X_test, data_landmarks['X_test'])
X_val_feats = extractFeatures(X_val, data_landmarks['X_val'])

data['X_train'] = X_train_feats
data['X_test'] = X_test_feats
data['X_val'] = X_val_feats