# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:08:33 2018

@author: TsungYuan
"""
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

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

x_data = []
x = np.zeros((4,60,25,1))
img = Image.open("pic/test3.jpg")
captcha_image = img.convert('1')
captcha_image.show()
ci1,ci2,ci3,ci4 = cut_number(captcha_image)
captcha_array1 = np.asarray(ci1)
captcha_array1 = depoint(ci1)
captcha_array2 = np.asarray(ci2)
captcha_array2 = depoint(ci2)
captcha_array3 = np.asarray(ci3)
captcha_array3 = depoint(ci3)
captcha_array4 = np.asarray(ci4)
captcha_array4 = depoint(ci4)
ca1 = captcha_array1.tolist()
ca2 = captcha_array2.tolist()
ca3 = captcha_array3.tolist()
ca4 = captcha_array4.tolist()
x_data.append(ca1)
x_data.append(ca2)
x_data.append(ca3)
x_data.append(ca4)
x[:,:,:,0] = np.asarray(x_data, np.float32)

model = keras.models.load_model('my_model1.h5')
prediction=model.predict_classes(x)
print(prediction)