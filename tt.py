# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:03:13 2018

@author: TsungYuan
"""
from PIL import Image
from captcha.image import ImageCaptcha

for i in range(10):
    captcha_text = "123"
    image = ImageCaptcha()
    captcha = image.generate(captcha_text+str(i))
    captcha_image = Image.open(captcha) 
    captcha_image.save('pic/test'+str(i)+'.jpg')