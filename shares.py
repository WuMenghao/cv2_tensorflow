# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 22:01:12 2019

神经网络逼近股票收盘均价

@author: wmh
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = np.linspace(1,15,15)
endPrice = np.array([2511.90,2538.26,2510.68,2591.66,2732.98,2701.69,2701.29,2678.67,2726.50,2681.50,2739.17,2715.07,2823.58,2864.90,2819.08])
beginPrice = np.array([2438.71,2500.88,2534.95,2512.52,2594.04,2743.26,2697.47,2695.24,2678.24,2678.23,2722.13,2874.93,2744.13,2717.46,2832.73])

plt.figure()
for i in range(0,15):
    # 柱状图
    day = np.zeros([2])
    day[0] = i
    day[1] = i
    
    price = np.zeros([2])
    price[0] = beginPrice[i]
    price[1] = endPrice[i]
    
    if endPrice[i] > beginPrice[i] :
        plt.plot(day,price,'r',lw=8)
    else:
        plt.plot(day,price,'g',lw=8)

# plt.show()
# A(15x1)*w1(1x10)+b1(1x10) = B(15x10)
# B(15x10)*w2(10x1)+b2(15*1) = C(15*1)

""" A """
dataNormal = np.zeros([15,1])
priceNomal = np.zeros([15,1])
# 归一化
for i in range(0,15):
    dataNormal[i,0] = i/14.0
    priceNomal[i,0] = endPrice[i]/3000.0
    
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

""" B """
w1 = tf.Variable(tf.random_uniform([1,10],0,1))
b1 = tf.Variable(tf.zeros([1,10]))
wb1 = tf.matmul(x,w1)+b1
layerB = tf.nn.relu(wb1) # 激励函数

""" C """
w2 = tf.Variable(tf.random_uniform([10,1],0,1))
b2 = tf.Variable(tf.zeros([15,1]))
wb2 = tf.matmul(layerB,w2)+b2
layerC = tf.nn.relu(wb2)

# 误差
loss = tf.reduce_mean(tf.square(y-layerC))
# 进行误差减少
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 训练
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 训练10000次
    for i in range(0,10000):
        sess.run(train_step,feed_dict={x:dataNormal,y:priceNomal})
    
    # 使用训练完成的神经网络进行预测股票价格
    predict = sess.run(layerC,feed_dict={x:dataNormal})
    predPrice = np.zeros([15,1])
    for i in range(0,15):
        predPrice[i,0] = (predict*3000)[i,0]
   
    plt.plot(data,predPrice,'b',lw=1)
    
plt.show()
        