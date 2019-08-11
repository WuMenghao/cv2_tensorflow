# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 20:57:51 2019

@author: wmh
"""
# In[1]:
import tensorflow as tf
import cv2

# In[2]:
#tensorflow hellow
hello = tf.constant('hello tf!')
sess = tf.Session()
print(sess.run(hello))

# In[3]:
#image read and display
path = 'D:/Documents/Pictures/images.png'
img = cv2.imread(path,1)
cv2.imshow('image',img)
cv2.waitKey(0);

# In[4]:
#image read and write
pathIn = 'D:/Documents/Pictures/images.png'
pathOut = 'D:/Documents/Pictures/images1.png'
img2 = cv2.imread(pathIn,1)
cv2.imwrite(pathOut,img2,[cv2.IMWRITE_PNG_COMPRESSION,9])
#cv2.IMWRITE_JPEG_QUALITY 1-100
#cv2.IMWRITE_PNG_COMPRESSION 0-9

# In[5]:
#像素的读取与写入
pathIn = 'D:/Documents/Pictures/images.png'
img = cv2.imread(pathIn,1)
(b,g,r) = img[100,100]
print((b,g,r))

for i in range(1,100):
    img[10+i,100] = (255,0,0)
cv2.imshow('image',img)
cv2.waitKey(1000)