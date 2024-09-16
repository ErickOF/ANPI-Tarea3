import numpy as np
from relajacion import relaxation
from typing import List, Tuple

def cubic_spline(
        points: List[List[float]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the coefficients for a natural cubic spline interpolation.

    A natural cubic spline is a piecewise cubic polynomial that interpolates a
    set of points such that the second derivative at the endpoints is zero
    (natural boundary conditions).
    This function calculates the coefficients for the cubic splines that pass
    through each of the given points.

    Parameters:
        points (List[List[float]]): A list of points where each point is
                                    represented as a list or tuple [x, y] to
                                    be used for interpolation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple
            containing four numpy arrays:
            - a: Coefficients for the cubic term (x^3)
            - b: Coefficients for the quadratic term (x^2)
            - c: Coefficients for the linear term (x)
            - d: Coefficients for the constant term

    Example:
        points: List[List[float]] = [[-5, 0], [-4.5, 0.0707], ..., [5, 0]]
        a, b, c, d = cubic_spline(points)
    """

    # Convert points to a numpy array for easier manipulation
    points: np.ndarray = np.array([np.array(p) for p in points])
    
    # Compute the differences in the x-coordinates (hk) between consecutive
    # points
    hk: np.ndarray = points[1:, 0] - points[:-1, 0]
    
    # Compute the differences in the y-coordinates (delta_yk) between
    # consecutive points
    delta_yk: np.ndarray = points[1:, 1] - points[:-1, 1]
    
    # Initialize matrix A and vector b to set up the linear system for spline
    # coefficients
    A: List[float] = []
    b: List[float] = []

    # Number of intervals (k) is one less than the number of points
    k: int = hk.shape[0]

    # Construct the matrix A and vector b for the linear system
    for i in range(1, k):
        # First equation (natural boundary condition at the start)
        if i == 1:
            A.append([2 * (hk[i - 1] + hk[i]), hk[i]] + [0] * (k - 3))
        # Last equation (natural boundary condition at the end)
        elif i == (k - 1):
            A.append([0] * (k - 3) + [hk[i - 1], 2 * (hk[i - 1] + hk[i])])
        # Intermediate equations
        else:
            A.append(
                [0] * (i - 2) + [hk[i - 1],
                2 * (hk[i - 1] + hk[i]),
                hk[i]] + [0] * (k - 2 - i)
            )
        
        # Right-hand side of the linear equations
        b.append(6 * (delta_yk[i] / hk[i] - delta_yk[i - 1] / hk[i - 1]))

    # Convert A and b to numpy arrays for further computation
    A: np.ndarray = np.array([np.array(a) for a in A])
    b: np.ndarray = np.array(b)
    
    # Initial guess for the relaxation method (zeros)
    x0: np.ndarray = np.zeros(b.shape)
    
    # Solve the linear system to find the 'sigmas' using the relaxation method
    sigmas: np.ndarray = relaxation(A, b, x0)
    
    # Add the natural boundary conditions: sigmas[0] = 0 and sigmas[n + 1] = 0
    sigmas: np.ndarray = np.append(0, np.append(sigmas, 0))
    
    # Initialize lists to hold the coefficients for the spline polynomials
    a: List[float] = []
    b: List[float] = []
    c: List[float] = []
    d: List[float] = []
    
    # Extract the y-coordinates from the points
    yk = points[:, 1]

    # Compute the coefficients for each spline segment
    for i in range(k):
        # Coefficient for (x - x_i)^3
        a.append((sigmas[i + 1] - sigmas[i]) / (6 * hk[i]))
        # Coefficient for (x - x_i)^2
        b.append(sigmas[i] / 2)
        # Coefficient for (x - x_i)
        c.append(
            (yk[i + 1] - yk[i]) / hk[i] -\
            (2 * hk[i] * sigmas[i] + hk[i] * sigmas[i + 1]) / 6)
        # Constant term (f(x_i))
        d.append(yk[i])

    # Convert the coefficient lists to numpy arrays and return them
    return np.array(a), np.array(b), np.array(c), np.array(d)

if __name__ == '__main__':
    print('Natural Cubic Splines Interpolation')

    # Example set of points for interpolation
    #points: List[float] = [[-5, 0], [-4.5,  0.0707],
    #                       [-4, 0], [-3.5, -0.0909],
    #                       [-3, 0], [-2.5,  0.1273],
    #                       [-2, 0], [-1.5, -0.2122],
    #                       [-1, 0], [-0.5,  0.6366],
    #                       [ 0, 1], [ 0.5,  0.6366],
    #                       [ 1, 0], [ 1.5,  0.2122],
    #                       [ 2, 0], [ 2.5,  0.1273],
    #                       [ 3, 0], [ 3.5,  0.0909],
    #                       [ 4, 0], [ 4.5,  0.0707],
    #                       [ 5, 0]]

    #print(points)

    # Perform cubic spline interpolation
    #a, b, c, d = cubic_spline(points)
    
    # Print the spline coefficients
    #print(a, b, c, d, sep='\n')
