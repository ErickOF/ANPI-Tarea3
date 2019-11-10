# -*- coding: utf-8 -*-
import numpy as np
from relajacion import *


def trazador_cubico(points):
    """
    Natural Cubic Splines Interpolation

    @points - list or tuple with points to compute interpolation
    
    return a, b, c, d - coeficients
    """
    # Validate some conditions
    if not isinstance(points, (list, tuple)):
        raise ValueError("'points' must be a list or tuple")
    elif len(points) < 4:
        raise ValueError("The length of 'points' must be greater than 1")
    # Convert points to numpy array
    points = np.array([np.array(p) for p in points])
    # Compute hk
    hk = points[1:, 0] - points[:-1, 0]
    # Compute delta_yk
    delta_yk = points[1:, 1] - points[:-1, 1]
    # Matrix and vector for solve linear system equation
    A, b = [], []
    # Amount of points - 1 (n)
    k = hk.shape[0]
    for i in range(1, k):
        # First case sigmas[1] = 0
        if i == 1:
            A.append([2*(hk[i - 1] + hk[i]), hk[i]] + [0]*(k - 3))
        # Second case sigmas[n+1] = 0
        elif i == k-1:
            A.append([0]*(k - 3) + [hk[i - 1], 2*(hk[i - 1] + hk[i])])
        else:
            A.append([0]*(i - 2) + [hk[i - 1], 2*(hk[i - 1] + hk[i]),
                                    hk[i]] + [0]*(k - 2 - i))
        b.append(6*(delta_yk[i]/hk[i] - delta_yk[i - 1]/hk[i - 1]))
    # Convert to numpy array
    A = np.array([np.array(a) for a in A])
    b = np.array(b)
    # Solving system linear equation
    x0 = np.zeros(b.shape)
    sigmas = relajacion(A, b, x0)
    # Append sigmas[1] = 0 and sigmas[n+1] = 0
    sigmas = np.append(0, np.append(sigmas, 0))
    # Coeficients
    a, b, c, d = [[], [], [], []]
    # Initial points
    xk = points[:, 0]
    yk = points[:, 1]
    # Compute coeficients
    for i in range(k-1):
        a.append((sigmas[i+1] - sigmas[i])/(6*hk[i]))
        b.append(sigmas[i]/2)
        c.append((yk[i+1] - yk[i])/hk[i] - (2*hk[i]*sigmas[i]+hk[i]*sigmas[i+1])/6)
        d.append(yk[i])
    # To numpy array
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    return a, b, c, d


if __name__ == '__main__':
    print('Natural Cubic Splines Interpolation method working, put your code below')
    # Example, uncomment to see
    # points = [[-5, 0], [-4.5, 0.0707], [-4, 0], [-3.5, -0.0909],
    #          [-3, 0], [-2.5, 0.1273], [-2, 0], [-1.5, -0.2122],
    #          [-1, 0], [-0.5, 0.6366], [ 0, 1], [ 0.5,  0.6366],
    #          [ 1, 0], [ 1.5, 0.2122], [ 2, 0], [ 2.5,  0.1273],
    #          [ 3, 0], [ 3.5, 0.0909], [ 4, 0], [ 4.5,  0.0707],
    #          [5, 0]]
    # print(points)

    # a, b, c, d = trazador_cubico(points)
    # print(a, b, c, d, sep='\n')