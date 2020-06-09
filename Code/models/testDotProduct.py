'''
@Author: Xin Han
@Date: 2020-06-07 10:05:49
@LastEditTime: 2020-06-08 20:58:17
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \StreamHC\Code\models\testDotProduct.py
'''
import numpy as np
from numba import jit
import math

def _fast_dot(x,y):
  
  """Compute the dot product of x and y using numba.

    Args:
    x - a numpy vector (or list).
    y - a numpy vector (or list).

    Returns:
    x_T.y
    """
  
  return np.dot(x,y)



def _fast_norm(x):
    """Compute the number of x using numba.

    Args:
    x - a numpy vector (or list).

    Returns:
    The 2-norm of x.
    """
    s = 0.0
    for i in range(len(x)):
        s += x[i] ** 2
    return s



def _fast_norm_diff(x, y):
    """Compute the norm of x - y using numba.

    Args:
    x - a numpy vector (or list).
    y - a numpy vector (or list).

    Returns:
    The 2-norm of x - y.
    """
    return _fast_norm(x - y)

a = np.array([0,1,0,1,0,0]) 
b = np.array([1,0,0,1,0,0])
c = np.array([0,1,0,0,0,1])
d = np.array([1,0,0,0,0,1])
e = np.array([0,0,1,1,0,0])

f = [b,c,d,e]


dit = 0.0
for i in range(len(f)):
        dit += _fast_norm_diff(a,f[i])
print(dit/4)

print(_fast_dot(a, b+c+d+e))
dotpr = (4*2*2 - 2*_fast_dot(a, b+c+d+e))/4
print(dotpr)