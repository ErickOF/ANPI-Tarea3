# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from trazador_cubico import *


def plot_ncsi(points, a, b, c, d):
    """
    Plotter for Natural Cubic Splines Interpolation

    @points - Initial points of NCSI
    
    @a - first coeficient
    
    @b - second coeficient
    
    @c - third coeficient
    
    @d - fourth coeficient
    """
    # Compute x values
    x = np.linspace(points[0][0], points[-1][0], 1000)
    # Interpolation polynomial
    pk = []
    # Compute interpolation polynomial
    for xk in x:
        for i in range(a.shape[0]):
            if points[i][0] <= xk <= points[i+1][0]:
                pk.append([xk, a[i]*(xk - points[i][0])**3 + b[i]*(xk - points[i][0])**2 + c[i]*(xk - points[i][0]) + d[i]])

    # Plot function
    plt.title('Natural Cubic Splines Interpolation')
    plt.plot([x for x, y in pk], [y for x, y in pk])
    plt.scatter([x for x, y in points], [y for x, y in points], c='red')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

if __name__ == '__main__':
    points = [[-5, 0], [-4.5, 0.0707], [-4, 0], [-3.5, -0.0909],
              [-3, 0], [-2.5, 0.1273], [-2, 0], [-1.5, -0.2122],
              [-1, 0], [-0.5, 0.6366], [ 0, 1], [ 0.5,  0.6366],
              [ 1, 0], [ 1.5, 0.2122], [ 2, 0], [ 2.5,  0.1273],
              [ 3, 0], [ 3.5, 0.0909], [ 4, 0], [ 4.5,  0.0707],
              [5, 0]]
    a, b, c, d = trazador_cubico(points)
    plot_ncsi(points, a, b, c, d)