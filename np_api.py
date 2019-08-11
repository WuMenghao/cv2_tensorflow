# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 23:01:50 2019

@author: wmh
"""
# In[1]
import numpy as np

# 矩阵定义
data1 = np.array([1,2,3,4,5])
data2 = np.array([[1,2],[3,4]])

# zeros ones
data3 = np.zeros([2,3])
data4 = np.ones([2,2])

#查找
rs1 = data2[1,0]
data2[1,0] = 5
rs2 = data2[1,0]

#基本运算
data5 = np.ones([2,3])
rs3 = data5*2
rs4 = data5/3
rs5 = data5+3

# 矩阵运算
data6 = np.array([[1,2,3],[4,5,6]])
rs6 = data5+data6
rs7 = data5*data6

print('data1:',data1)
print('data2:',data2)
print('data1.shape:',data1.shape,', data2.shape:',data2.shape)
print('data3:',data3)
print('data4:',data4)
print('rs1:',rs1)
print('rs2:',rs2)
print('rs3:',rs3)
print('rs4:',rs4)
print('rs5:',rs5)
print('rs6:',rs6)
print('rs7:',rs7)
