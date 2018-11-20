import numpy as np
import pandas as pd

def processString(s):
    s = s.split()
    img = np.zeros((48, 48))
    for i in range(48):
        for j in range(48):
            img[i][j] = s[48*i + j]
    return np.array(img) 

def getDataset():
    data_path = '../Data/fer2013/fer2013.csv'
    df = pd.read_csv(data_path)
    X_train = []; y_train = []
    X_test = []; y_test = []
    X_val = []; y_val = []
    for index, row in df.iterrows():
        print('Reading row {}'.format(index))
        if(row['Usage'] == 'Training'):
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