import dlib
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor_path = 'E:\\Study-Ebooks\\Sem-6\\Attendance-System\\Models\\shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	return (x, y, w, h)

def shape_to_np(shape, dtype = 'int'):
	cords = np.zeros((68, 2), dtype = dtype)
	for i in range(68):
		cords[i] = (shape.part(i).x, shape.part(i).y)

	return cords

def detect_faces(gray):
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	return faces

def landmarks(gray, showImage = False):
	# We can give only faces
	x, y = gray.shape[0], gray.shape[1]
	rect = dlib.rectangle(0, 0, x, y)
	shape = predictor(gray, rect)
	shape = shape_to_np(shape)

	(x, y, w, h) = rect_to_bb(rect)
	
	cv2.rectangle(gray, (x, y), (x + w, y + h), 255, 2)
	
	cv2.putText(gray, "Face", (x - 5, y - 5), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

	for (x, y) in shape:
		cv2.circle(gray, (x, y), 1, 255, -1)

	if(showImage):
		plt.imshow(gray, cmap = 'gray')
		plt.show()
	
	return shape			

def features(X):
	return np.array([np.linalg.norm(X[i] - X[j]) for i in range(68) for j in range(68)])
	
def transform(X_train):
	X_train_new = []
	for i in range(len(X_train)):
		print('{} / {}'.format(i, len(X_train)))
		image = X_train[i]
		gray = image.astype('uint8')
		marks = landmarks(gray)
		X1 = features(marks)
		X_train_new.append(X1)
	return np.array(X_train_new)

def transformData(data):
	X_train = data['X_train']
	X_test = data['X_test']
	X_val = data['X_val']
	data['X_train'] = transform(X_train)
	data['X_val'] = transform(X_val)
	data['X_test'] = transform(X_test)
	return data

f = open('../Data/data.pickle', 'rb')
data = pickle.load(f)
print('Read dataset')

data1 = transformData(data)

# image = data['X_train'][2]
# gray = image.astype('uint8')
# marks = landmarks(gray, showImage = True)

# image = cv2.imread('img.bmp')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.resize(image, (100, 100))
# rects = detector(gray, 1)
# image = data['X_train'][0]
# gray = image.astype('uint8')
# rects = detect_faces(gray)
