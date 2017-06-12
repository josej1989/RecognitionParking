import creacion_datos_entrenamiento as training_data
import datos_test
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

DIR_ENTRENADOR = 'C:/Users/JOSEO/Documents/PyQt/TensorFlow/train/'
DIR_TEST ='C:/Users/JOSEO/Documents/PyQt/TensorFlow/test/'
tamano = 50
LR = 1e-3

MODEL_NAME = 'freeparking-{}-{}.model'.format(LR,'2conv-basic')

def label_img(img):
	word_label = img.split('.')[-3]
	if word_label == 'cat': return [1,0]
	elif word_label == 'dog': return [0,1]
	

#train_data = training_data.create_train_data()
#test_data = datos_test.process_test_data()


convnet = input_data(shape=[None, tamano, tamano, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


train_data = np.load('train_data.npy')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
	model.load(MODEL_NAME)
	print('Model Loaded!!!')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,tamano,tamano,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,tamano,tamano,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
