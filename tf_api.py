# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:40:29 2019

@author: wmh
"""

# In[1]
import tensorflow as tf

# In[2]
'''
# tensorflow常量/变量定义
'''
data1 = tf.constant(2.5)
data2 = tf.Variable(10,name='var',dtype=tf.int32)
print(data1)
print(data2)


# In[3]
'''# tensorflow的实质 ： 张量tensor + 计算图graphs
# tensor 数据  ， op 操作 ， graphs 数据图运算 ， session tensorflow会话
'''
#session
session = tf.Session()
#initializer 用于初始化变量
initializer = tf.global_variables_initializer()
#with关键词可以在session使用完后自动关闭
with session:
    session.run(initializer)
    print(session.run(data1))
    print(session.run(data2))
 
    
# In[4]
''' 使用tensorflow进行四则运算 '''
# tensor
data1 = tf.Variable(3)
data2 = tf.Variable(6)
# op
rs1 = tf.add(data1,data2)
rs2 = tf.multiply(data1,data2)
rs3 = tf.subtract(data1,data2)
rs4 = tf.divide(data1,data2)
# assign()将后值赋予给前值的操作
rs5 = tf.assign(data2,rs2)
#with关键词可以在session使用完后自动关闭
with tf.Session() as session:
    #initializer 用于初始化变量
    initializer = tf.global_variables_initializer()
    session.run(initializer)
    
    print(session.run(rs1))
    print(session.run(rs2))
    print(session.run(rs3))
    print(session.run(rs4))
    
    print('tf.assign(data2,rs2)',session.run(rs5))
    print('rs5.eval()',rs5.eval())
    print('tf.get_default_session().run(rs5)',tf.get_default_session().run(rs5))
    
print('end!')

# In[5]
'''
placeholder 用于预定义参数
'''
data1 = tf.placeholder(tf.float32)
data2 = tf.placeholder(tf.float32)
rs = tf.add(data1,data2)

with tf.Session() as sess:
    result = sess.run(rs,feed_dict = {data1:6,data2:2})
    print('Result is :',result)
    
print('end!')

# In[6]
''' 矩阵运算 '''
data1 = tf.constant([[6,6]])
data2 = tf.constant([[2],[2]])
data3 = tf.constant([[3,3]])
data4 = tf.constant([[1,2],[3,4],[5,6]])
data5 = tf.constant([1,2,3,4,5,6,7])

print(data4.shape)

''' 矩阵相加/相乘 '''
rs1 = tf.matmul(data1,data2)
rs2 = tf.add(data1,data3)
rs3 = tf.multiply(data1,data2)

with tf.Session() as sess:
    print('data4:',sess.run(data4))
    print('data4[0]',sess.run(data4[0]))
    print('data4[:,0]',sess.run(data4[:,0]))
    print('data4[0,0]',sess.run(data4[0,0]))
    print('data4[1,0]',sess.run(data4[1,0]))
    print('data4[2,0]',sess.run(data4[2,0]))
    print('data5[:4]',sess.run(data5[:4]))

    print('tf.matmul(data1,data2) :',sess.run(rs1))
    print('tf.add(data1,data3) :',sess.run(rs2))
    print('tf.multiply(data1,data2) :',sess.run(rs3))

# In[7]
''' 矩阵基础 赋值 '''
#全0矩阵
mat1 = tf.zeros([2,3])
#全1矩阵
mat2 = tf.ones([2,3])
#填充矩阵
mat3 = tf.fill([2,3],15)
#zeros_like 类似格式0填充
sourse= tf.constant([[1],[2],[3]])
mat4 = tf.zeros_like(sourse)
#linspace 值分割
mat5 = tf.linspace(0.0,2.0,11)
#random_uniform 随机值
mat6 = tf.random_uniform([2,3],-1,2)

with tf.Session() as sess:
    print('mat1',sess.run(mat1))
    print('mat2',sess.run(mat2))
    print('mat3',sess.run(mat3))
    print('mat3',sess.run(mat4))
    print('mat3',sess.run(mat5))
    print('mat3',sess.run(mat6))
