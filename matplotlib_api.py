# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:43:21 2019

@author: wmh
"""
"""
matplotlib是一个用来进行图形绘制的库

    三大图表：折线图 饼状图 柱状图
    
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5,6,7,8])
y = np.array([23,1,55,67,77,8,4,112])

# 折线图 参数 1:x 2:y 3:color 4:line weight
plt.plot(x,y,'r')
plt.plot(x,y,'g',lw = 10)

#柱状图 1:x 2:y 3:width 4:alpha
plt.bar(x,y,0.9,alpha=1,color='b')
plt.show()