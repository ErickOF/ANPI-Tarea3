import matplotlib.pyplot as plt
import numpy as np

from typing import List

from trazador_cubico import cubic_spline


def plot_ncsi(points: List[List[float]], a, b, c, d) -> None:
    """
    Plots the Natural Cubic Spline Interpolation (NCSI) for a given set of
    points.

    This function takes the coefficients of the natural cubic splines and the
    initial points to compute the interpolating polynomial and plot it along
    with the initial points.

    Parameters:
        points (List[List[float]]): List of points where each point is a list
                                    [x, y].
        a (np.ndarray): Coefficients for the cubic terms (x^3) of the spline
                        segments.
        b (np.ndarray): Coefficients for the quadratic terms (x^2) of the
                        spline segments.
        c (np.ndarray): Coefficients for the linear terms (x) of the spline
                        segments.
        d (np.ndarray): Coefficients for the constant terms of the spline
                        segments.

    Example:
        points: List[List[float]] = [[-5, 0], [-4.5, 0.0707], ..., [5, 0]]
        a, b, c, d = cubic_spline(points)
        plot_ncsi(points, a, b, c, d)
    """

    # Generate 1000 evenly spaced values between the x-coordinates of the
    # first and last points
    x: np.ndarray = np.linspace(points[0][0], points[-1][0], 1000)
    
    # Initialize a list to store the computed values of the interpolation
    # polynomial
    pk: List[float] = []

    # Compute the interpolation polynomial for each segment of the spline
    for xk in x:
        for i in range(a.shape[0]):
            # Find the interval where xk belongs
            if points[i][0] <= xk <= points[i + 1][0]:
                # Evaluate the spline polynomial at xk and append the result
                # to pk
                pk.append([xk, a[i] * (xk - points[i][0])**3 + 
                                b[i] * (xk - points[i][0])**2 + 
                                c[i] * (xk - points[i][0]) + 
                                d[i]])

    # Plot the natural cubic spline interpolation
    plt.title('Natural Cubic Splines Interpolation')
    plt.plot([x for x, _ in pk], [y for _, y in pk], label='Cubic Spline')
    
    # Plot the original points
    plt.scatter(
        [x for x, _ in points], [y for _, y in points],
        c='red',
        label='Data Points'
    )
    # Add grid, labels, and legend
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    
    # Display the plot
    plt.show()

if __name__ == '__main__':
    # Define a set of points for interpolation
    points: List[List[float]] = [[-5, 0], [-4.5,  0.0707],
                                 [-4, 0], [-3.5, -0.0909],
                                 [-3, 0], [-2.5,  0.1273],
                                 [-2, 0], [-1.5, -0.2122],
                                 [-1, 0], [-0.5,  0.6366],
                                 [ 0, 1], [ 0.5,  0.6366],
                                 [ 1, 0], [ 1.5,  0.2122],
                                 [ 2, 0], [ 2.5,  0.1273],
                                 [ 3, 0], [ 3.5,  0.0909],
                                 [ 4, 0], [ 4.5,  0.0707],
                                 [ 5, 0]]

    # Calculate the spline coefficients
    a, b, c, d = cubic_spline(points)

    # Plot the natural cubic spline interpolation
    plot_ncsi(points, a, b, c, d)
