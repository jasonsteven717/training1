# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:24:43 2018

@author: TsungYuan
"""

from PIL import Image
from captcha.image import ImageCaptcha
import numpy as np
import random
import pickle
from os.path import join, exists
from os import makedirs
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from PIL import Image, ImageEnhance
from scipy.misc import imshow
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import pandas as pd
    
np.set_printoptions(threshold=np.inf)

VNUM = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
CAPTCHA_LENGTH = 4
VNUM_L = len(VNUM)
DATA_LENGTH = 10000
DATA_PATH = 'data'

def depoint(img):
    img = np.asarray(img)
    for y in range(1,24):
        for x in range(1,59):
            count = 0
            if img[x,y-1] > 245: #上
                count = count + 1
            if img[x,y+1] > 245: #下
                count = count + 1
            if img[x-1,y] > 245: #左
                count = count + 1
            if img[x+1,y] > 245: #右
                count = count + 1
            if img[x-1,y-1] > 245: #左上
                count = count + 1
            if img[x-1,y+1] > 245: #左下
                count = count + 1
            if img[x+1,y-1] > 245: #右上
                count = count + 1
            if img[x+1,y+1] > 245: #右下
                count = count + 1
            if count > 4:
                img[x,y] = 255
    return img

def cut_number(img):
    region1 = (5,0,30,60)
    cropImg1 = img.crop(region1)
    region2 = (31,0,56,60)
    cropImg2 = img.crop(region2)
    region3 = (57,0,82,60)
    cropImg3 = img.crop(region3)
    region4 = (83,0,108,60)
    cropImg4 = img.crop(region4)
    return cropImg1,cropImg2,cropImg3,cropImg4
    
def random_text():
    seed1 = random.randint(0,9)
    seed2 = random.randint(0,9)
    seed3 = random.randint(0,9)
    seed4 = random.randint(0,9)
    text = VNUM[seed1] + VNUM[seed2] + VNUM[seed3] + VNUM[seed4]
    return text


def generate_captcha(captcha_text):
    image = ImageCaptcha()
    captcha = image.generate(captcha_text)
    captcha_image = Image.open(captcha) 
    
    captcha_image = captcha_image.convert('1')
    ci1,ci2,ci3,ci4 = cut_number(captcha_image)

    captcha_array1 = np.asarray(ci1)
    #captcha_array1 = depoint(ci1)
    
    captcha_array2 = np.asarray(ci2)
    #captcha_array2 = depoint(ci2)

    captcha_array3 = np.asarray(ci3)
    #captcha_array3 = depoint(ci3)

    captcha_array4 = np.asarray(ci4)
    #captcha_array4 = depoint(ci4)

    captcha_array1 = captcha_array1.tolist()
    captcha_array2 = captcha_array2.tolist()
    captcha_array3 = captcha_array3.tolist()
    captcha_array4 = captcha_array4.tolist()
    return captcha_array1,captcha_array2,captcha_array3,captcha_array4

def text2onehot(text):
    if len(text) > VNUM_L:
        return False
    vector1 = np.zeros(VNUM_L)  
    vector2 = np.zeros(VNUM_L)
    vector3 = np.zeros(VNUM_L)
    vector4 = np.zeros(VNUM_L)
    vector1[int(text[0])] = 1
    vector2[int(text[1])] = 1
    vector3[int(text[2])] = 1
    vector4[int(text[3])] = 1
    return vector1,vector2,vector3,vector4

def onehot2text(vector):
    text = []
    for j in range(4):
        for i in range(10):  
            if vector[j,i] != 0:
                text.append(i)
    return text

def write2pickle(x_data,y_data):
    if not exists(DATA_PATH):
        makedirs(DATA_PATH)
    x = np.asarray(x_data, np.float32)
    y = np.asarray(y_data, np.float32)
    with open(join(DATA_PATH, 'data1.pkl'), 'wb') as f:
        pickle.dump(x, f)
        pickle.dump(y, f)

def generate_data():
    x = np.zeros((DATA_LENGTH*4,60,25,3))
    x_data, y_data = [], []
    for i in range(DATA_LENGTH):
        text = random_text()
        ca1,ca2,ca3,ca4 = generate_captcha(text)
        #print(captcha, captcha.shape)
        l1,l2,l3,l4 = text2onehot(text)
        x_data.append(ca1)
        x_data.append(ca2)
        x_data.append(ca3)
        x_data.append(ca4)
        y_data.append(l1)
        y_data.append(l2)
        y_data.append(l3)
        y_data.append(l4)
    #print(x_data,y_data)
    x[:,:,:,:] = np.asarray(x_data, np.float32)
    #print(np.shape(x))
    
    y = np.asarray(y_data, np.float32)
    #print(np.shape(y))
    #write2pickle(x,y)
    print("generate trainingdata successfully")
    return x,y

    
if __name__== "__main__":

    #x_data,y_data = generate_data()
    
    with open('data//data1.pkl', 'rb') as f:
        x_data = pickle.load(f)
        y_data = pickle.load(f)
    
    model=Sequential()
    model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(60,25,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))
    print(model.summary())
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    train_history=model.fit(x=x_data,y=y_data,validation_split=0.2,epochs=100, batch_size=100,verbose=2)

    prediction=model.predict_classes(x_data[9990:10000,:,:,:])
    print(prediction)
    print(onehot2text(y_data[9990:10000,:]))
    model.save('my_model1.h5')
    print("save_model")

    model = keras.models.load_model('my_model1.h5')
    with open('test//test.pkl', 'rb') as f:
        x_data = pickle.load(f)
        y_data = pickle.load(f)
    prediction=model.predict_classes(x_data)
    print(prediction)
    print(onehot2text(y_data))






   
                 
