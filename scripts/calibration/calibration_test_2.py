import pickle
import matplotlib.pyplot as plt
import numpy as np
  
## WORKING in ax+by=c form ========== 
# x = np.linspace(-10 ,10)
# a = -2
# b = 7
# c = 10

# # a*x + b*y = c
# # a*x/c + b*y/c = 1
# y = -a/b*x + c/b

x_ = np.array([
-1.21200000e+08,
-1.41400000e+08,
-1.61600000e+08,
-2.02000000e+08,
-2.42400000e+08,
-2.92900000e+08,
-3.33300000e+08,
-3.53500000e+08,
-3.83800101e+08,
-4.04000000e+08,
], np.float64)

y_ = np.array([
    -1.22880000e+08,
    -1.47560000e+08,
    -1.74080000e+08,
    -2.24800000e+08,
    -2.80320000e+08,
    -3.54090000e+08,
    -4.22070000e+08,
    -4.70750000e+08,
    -5.37320138e+08,
    -5.95200000e+08,
], np.float64) 
 
 
c_ = -np.array([ -21566.14790793,  -35943.57984655,  -40736.05715943,  -50321.01178517,
  -59905.96641092,  -69490.92103667,  -83868.35297529,  -91057.04498222,
 -110227.00215848, -119811.93282185], np.float64)

fig, ax = plt.subplots(1,1)


plt.plot(x_, y_, "*r")

arr1 = np.vstack([x_, y_]).T
arr2 = c_

a_, b_ = np.linalg.lstsq(arr1, arr2, rcond=None)[0]
x__ = np.linspace(x_[0], x_[-1], len(y_))
c_ = np.mean(a_*x__ + b_*y_)
# print("calculated a_, b_, c_", a_, b_, c_)

y__ = a_*x__/b_ + c_/b_


# y = -a/b*x + c/b
plt.plot(x__, y__, "-g")
plt.show()

                                                                                

## WORKING in y=mx+d ================
# a*x + b*y = c
# y = -a/b*x + c/b

# y = m*x    +  d
# m = -a/b
# d = c/b                                                                                                                               
# print("Real m, d", m, d)
# y = (c - a*x)/b
# fig, ax = plt.subplots(1,1)
# ax.plot(x,y, "-b")

# y =  m*x + d
# ax.plot(x,y, "-r")

# x_ = x
# y_ = y + np.random.normal(0, 0.5, len(y))
# d_ = y_ - m*x_
# plt.plot(x_, y_, "*r")

# # --------------------

# A = np.vstack([x_, d_]).T
# B = y_

# m__, d__ = np.linalg.lstsq(A, B, rcond=None)[0]
# print(np.linalg.lstsq(A, B, rcond=None))
# print("calculated m, d", m__, -d__/a*c*m)
# plt.plot(x, m__*x + -d__/a*c*m, "-g")
# plt.show()