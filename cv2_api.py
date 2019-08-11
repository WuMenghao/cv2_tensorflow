# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 10:24:53 2019

图片几何变换

@author: wmh
"""

# In[1]
import cv2
import numpy as np

# In[]

"""1.图片缩放"""
# 图片缩放 1:load 2:info 3:resize 4:check
# 放大 缩小 等比例缩放 非等比例缩放
# 1 opencv API resize 2 算法原理 3 源码

# 1 load
img = cv2.imread('D:/Documents/Pictures/images.png',1)
# 2 info
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
mode = imgInfo[2]
print(imgInfo)
# 3 resize
# 等比例缩放
# 算法： 最近临域插值 双线性插值 像素关系重采样 立方插值

# In[2]
"""最近临域插值"""
dstHight = int(height*0.5)
dstWidth = int(width*0.5)
dst = cv2.resize(img,(dstWidth,dstHight))
# 4 check
cv2.imshow('image',dst)
cv2.waitKey(0)

# In[3]
"""双线性插值"""
dstHight = int(height*2.0)
dstWidth = int(width*2.0)
dstImage = np.zeros((dstHight,dstWidth,3),np.uint8) #uint8 0-255
for y in range(0,dstHight):
    for x in range(0,dstWidth):
        yNew = int(y*(height*1.0/dstHight))
        xNew = int(x*(width*1.0/dstWidth))
        dstImage[y,x] = img[yNew,xNew]
        
cv2.imshow('dst',dstImage)
cv2.waitKey(0)

# In[4]
"""图片剪切"""
img = cv2.imread('D:/Documents/Pictures/images.png',1)
dst = img[100:200,100:200] #数组切片
cv2.imshow('dst',dst)
cv2.waitKey(0)

# In[5]
"""
2.图片移位

原理：
    [1,0,100],[0,1,100] 2*2 2*1
    [[1,0],[0,1]] 2*2 A
    [[100],[100]] 2*1 B
    [x,y] C
    运算：
    A*C+B = [[1*x+0*y],[0*x+1*y]]+[[100],[100]]
          = [[x+100],[y+100]]
    像素：
    (10,20) -> (110,120)
"""
img = cv2.imread('D:/Documents/Pictures/images.png',1)
height = img.shape[0]
width = img.shape[1]
# 移位操作 warpAffine 1:data 2:mat 3:info
matShift = np.float32([[1,0,100],[0,1,100]])
dst = cv2.warpAffine(img,matShift,(height,width))

cv2.imshow('src',img)
cv2.imshow('dst',dst)
cv2.waitKey(0)

# In[6]
img = cv2.imread('D:/Documents/Pictures/images.png',1)
height = img.shape[0]
width = img.shape[1]

dst = np.zeros(img.shape,np.uint8)
for h in range(0,height):
    for w in range(0,width):
        if (w+100) < width :
            dst[h,w+100] = img[h,w]
        
cv2.imshow('dst',dst)
cv2.waitKey(0)

# In[7]
"""
3.图片镜像
    1、创建一个足够大的画板
    2、将一副图像分别从前往后、从后往前绘制
    3、绘制中心分割线

"""
img = cv2.imread('D:/Documents/Pictures/images.png',1)
height = img.shape[0]
width = img.shape[1]
deep = img.shape[2]

dst = np.zeros((height*2,width,deep),np.uint8)
#绘制
for h in range(0,height):
    for w in range(0,width):
        dst[h,w] = img[h,w]
        dst[height*2-h-1,w] = img[h,w]
#分割线
for i in range(0,width):
    dst[height,i] = (0,0,255) #BGR
    
cv2.imshow('dst',dst)
cv2.waitKey(0)

# In[8]
"""
4.图片仿射变换
    通过把原图片上三个点映射到目标图片上三个点
"""
img = cv2.imread('D:/Documents/Pictures/images.png',1)
height = img.shape[0]
width = img.shape[1]
deep = img.shape[2]

matSrc = np.float32([[0,0],[0,height-1],[width-1,0]])
matDst = np.float32([[50,50],[100,height-100],[width-100,100]])
#组合
matAffine = cv2.getAffineTransform(matSrc,matDst)
dst = cv2.warpAffine(img,matAffine,(width,height))

cv2.imshow('img',img)
cv2.imshow('dst',dst)
cv2.waitKey(0)

# In[9]
"""
5.图片旋转
    
"""
img = cv2.imread('D:/Documents/Pictures/images.png',1)
height = img.shape[0]
width = img.shape[1]
# 旋转矩阵 1:center 2:angle 3:scale 缩放系数
matRotate = cv2.getRotationMatrix2D((height*0.5,width*0.5),45,0.5)
# 仿射变换
dst = cv2.warpAffine(img,matRotate,(height,width))

cv2.imshow('dst',dst)
cv2.waitKey(0)

# In[10]
"""
6.灰度处理
    1 最重要
    2 边缘检测 图片识别的基础
    3 实时性
"""

"""
1 imread
"""
img1 = cv2.imread('D:/Documents/Pictures/images.png',1)
img2 = cv2.imread('D:/Documents/Pictures/images.png',0)

print(img1.shape)
print(img2.shape)
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.waitKey(0)

# In[11]
"""
2 cvtColor
"""

img = cv2.imread('D:/Documents/Pictures/images.png',1)
# 1:data 2:BGR gray
dst = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('dst',dst)
cv2.waitKey(0)

# In[11]
"""
源码实现
"""

img = cv2.imread('D:/Documents/Pictures/images.png',1)
height = img.shape[0]
width = img.shape[1]
# RGB R==G==B = gray
dst = np.zeros((height,width,3),np.uint8)
for h in range(0,height):
    for w in range(0,width):
        (b,g,r) = img[h,w]
        gray = (int(b)+int(g)+int(r))/3 #均值
        dst[h,w] = np.uint8(gray)
        
cv2.imshow('dst',dst)
cv2.waitKey(0)

# In[11]
"""
心理学计算公式
    gray = r*0.299+g*0.587+b*0.114
"""
img = cv2.imread('D:/Documents/Pictures/images.png',1)
height = img.shape[0]
width = img.shape[1]
# RGB R==G==B = gray
dst = np.zeros((height,width,3),np.uint8)
for h in range(0,height):
    for w in range(0,width):
        (b,g,r) = img[h,w]
        gray = int(r)*0.299+int(g)*0.587+int(b)*0.114
        dst[h,w] = np.uint8(gray)
        
cv2.imshow('dst',dst)
cv2.waitKey(0)

# In[11]
"""
优化
"""
img = cv2.imread('D:/Documents/Pictures/images.png',1)
height = img.shape[0]
width = img.shape[1]
# RGB R==G==B = gray
dst = np.zeros((height,width,3),np.uint8)
for h in range(0,height):
    for w in range(0,width):
        (b,g,r) = img[h,w]
        #gray = int(r)*0.299+int(g)*0.587+int(b)*0.114
        #1:浮点运算转定点运算 
        #gray = (r*1+g*2+g*1)/4
        #2:乘法转移位运算
        gray = (r+(g<<1)+b)>>2
        dst[h,w] = np.uint8(gray)
        
cv2.imshow('dst',dst)
cv2.waitKey(0)

# In[10]
"""
7.颜色反转
    1： 0-255 255-当前
"""
#灰色图片
img = cv2.imread('D:/Documents/Pictures/images.png',1)
height = img.shape[0]
width = img.shape[1]

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dst = dst = np.zeros((height,width,3),np.uint8)
for h in range(0,height):
    for w in range(0,width):
        grayPixel = gray[h,w]
        dst[h,w] = 255 - grayPixel
        
cv2.imshow('dst',dst)
cv2.waitKey(0)

# In[10]
#彩色图片
img = cv2.imread('D:/Documents/Pictures/images.png',1)
height = img.shape[0]
width = img.shape[1]

dst = dst = np.zeros((height,width,3),np.uint8)
for h in range(0,height):
    for w in range(0,width):
        (b,g,r) = img[h,w]
        dst[h,w] = (255-b,255-g,255-r)
        
cv2.imshow('dst',dst)
cv2.waitKey(0)
