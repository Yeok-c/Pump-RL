import pickle
import matplotlib.pyplot as plt
import numpy as np
  

cyan = '#8ECFC9'
orange = '#FFBE7A'
orange_1 = '#FA9F6F' # '#FA7F6F'
magenta = '#FA7F6F'
blue = '#82B0D2'
violet = '#BEB8DC'
beige = '#E7DAD2'
grey = '#999999'
grey_darker = '#444444'

[
    P_Ll,
    P_Cl,
    P_2l,
    V_1l,
    V_2l,
    P_Lr,
    P_Cr,
    P_2r,
    V_1r,
    V_2r,
        ]=pickle.load(open( "./scripts/calibration/calibration_results.p", "rb" ) )

P_Ll = np.array(P_Ll)
P_Cl = np.array(P_Cl)
P_2l = np.array(P_2l)
V_1l = np.array(V_1l)
V_2l = np.array(V_2l)

P_Lr = np.array(P_Lr)
P_Cr = np.array(P_Cr)
P_2r = np.array(P_2r)
V_1r = np.array(V_1r)
V_2r = np.array(V_2r)

P_0 = 1.01*1e5  # Pa

def calculate(P_L1, P_C1, P_2, V_1, V_2):
    # ax+by=c form
    a = (P_L1*P_0 - P_2*P_0)
    b = P_L1*(P_L1-P_0) + P_2*(P_0-P_2)
    c = (P_2-P_C1)*P_0 * V_1

    # y=mx+c form


    assert(V_1 == V_2)
    return a, b, c
    # a*V_L + b*C_Ld = C*V_c


X=[]
Y=[]
C=[]
for i in range(0, len(P_Ll)):
    x,y,c = calculate(P_Ll[i], P_Cl[i], P_2l[i], V_1l[i], V_2l[i])
    X.append(x)
    Y.append(y)
    C.append(c)

X=np.array(X)
Y=np.array(Y)
C=np.array(C)

# a*x + b*y = c
fig, ax = plt.subplots(1,2)
ax = ax.flatten()
ax[0].plot(X, Y, "*b")

arr1 = np.vstack([X, Y]).T
arr2 = C

print(np.shape(arr1), np.shape(arr2))

# This is the A and B in 
# A = V_L , load volume
# B = C_Ld, load deformation term 
A, B = np.linalg.lstsq(arr1, arr2, rcond=None)[0]
C_ = np.mean(A*X + B*Y)

X_ = np.linspace(-6e8, 6e7, len(C))
print("calculated a_, b_, c_", A, B, C)

# From AX + BY = C
Y_ = -A*X_/B + C_/B
ax[0].plot(X_, Y_, "-g") # Theoretically correct 

Y_ = +A/B*X_ - C_ # NO B ??
ax[0].plot(X_, Y_, "-r") # The one that yields closest line

ax[0].legend(["Data", "Theoretically correct...?", "Theoretically wrong but closest line"])

# RIGHT SIDE


X=[]
Y=[]
C=[]
for i in range(1, len(P_Lr)):
    x,y,c = calculate(P_Lr[i], P_Cr[i], P_2r[i], V_1r[i], V_2r[i])
    X.append(x)
    Y.append(y)
    C.append(c)
X=np.array(X)
Y=np.array(Y)
C=np.array(C)

# a*x + b*y = c
ax[1].plot(X, Y, "*b")

arr1 = np.vstack([X, Y]).T
arr2 = C

print(arr1, arr2)
print(np.shape(arr1), np.shape(arr2))

# This is the A and B in 
# A = V_L , load volume
# B = C_Ld, load deformation term 

print(arr1, arr2)

A, B = np.linalg.lstsq(arr1, arr2, rcond=None)[0]
C_ = np.mean(A*X + B*Y)

X_ = np.linspace(0, 10000, len(C))
print("calculated a_, b_, c_", A, B, C_)

# From AX + BY = C
# Y = (C - AX)/B
# Y_ = (C_-A*X_)/B
# ax[1].plot(X_, Y_, "-g") # Theoretically correct 

Y_ = +A/B*X_ - C_ # NO B ??
ax[1].plot(X_, Y_, "-r") # The one that yields closest line

ax[1].legend(["Data", "Theoretically correct...?", "Theoretically wrong but closest line"])
plt.show()


# def calculate(P_L1, P_C1, P_2, V_1, V_2):
#     # ax+by=c form
#     x = (P_L1*P_0 - P_2*P_0)
#     y = P_L1*(P_L1-P_0) + P_2*(P_0-P_2)
#     c = (P_2-P_C1)*P_0 * V_1

#     assert(V_1 == V_2)
#     return x/c, y/c
#     # return a, b, c
#     # a*V_L + b*C_Ld = C*V_c


# X=[]
# Y=[]
# # 0 had a inf so start at 1
# for i in range(1, len(P_Ll)):
#     x,y = calculate(P_Ll[i], P_Cl[i], P_2l[i], V_1l[i], V_2l[i])
#     X.append(x)
#     Y.append(y)

# X=np.array(X)
# Y=np.array(Y)

# arr1 = np.row_stack((X, Y)).T
# arr2 = np.ones(len(arr1)) # np.expand_dims(C, axis=0)
# print(arr1, arr2)
# print("Shapes" ,arr1.shape, arr2.shape)
# print("Result")
# [a,b] = np.linalg.lstsq(arr1,arr2, rcond=None)[0]
# print(a,b)

# # fig = plt.figure()
# # ax = fig.add_subplot(projection='3d')
# fig, ax = plt.subplots()
# ax.plot(X, Y, "*r") # plot all datapoints
# x = np.linspace(20000, 50000)
# y = -a*x + 1/b
# ax.plot(x,y, color=orange_1, label='Calculated line')
# plt.show()
