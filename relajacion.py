import numpy as np


def relaxation(
        A: np.ndarray,
        b: np.ndarray,
        x0: np.ndarray,
        omega: float = 0.5,
        tol: float = 1e-8
    ) -> np.ndarray:
    """
    Performs the Successive Over-Relaxation (SOR) method to solve the system
    of linear equations Ax = b.

    The SOR method is an iterative technique used to solve a system of linear
    equations, especially useful when the coefficient matrix A is large and
    sparse. This method improves the convergence rate of the Gauss-Seidel
    method by introducing a relaxation factor, omega.

    Parameters:
        A (np.ndarray): An n x n matrix representing the coefficients of the
                        linear system.
        b (np.ndarray): An n-dimensional vector representing the right-hand
                        side of the equation.
        x0 (np.ndarray): An n-dimensional vector representing the initial
                         guess for the solution.
        omega (float): The relaxation factor (0 < omega < 2). Default is 0.5.
        tol (float): The tolerance for convergence. The iteration stops when
                     the norm of the residual is less than this value. Default
                     is 1e-8.

    Returns:
        np.ndarray: An n-dimensional vector representing the solution to the
                    system of linear equations.

    Example:
        A: np.ndarray = np.array([[4, -1, -6, 0],
                                  [-5, -4, 10, 8],
                                  [0, 9, 4, -2],
                                  [1, 0, -7, 5]])
        b: np.ndarray = np.array([2, 21, -12, -6])
        x0: np.ndarray = np.zeros(4)
        x: np.ndarray = relaxation(A, b, x0)
    """
    
    # Initialize the solution with the initial guess
    x: np.ndarray = x0.copy()
    # Calculate the initial residual (difference between Ax and b)
    residual: np.ndarray = np.linalg.norm(np.matmul(A, x) - b)

    # Iterate until the residual is less than the specified tolerance
    while residual > tol:
        # Update each component of the solution vector x
        for i in range(A.shape[0]):
            # Temporary sum for the off-diagonal elements
            sigma: float = 0.0
            
            # Compute the sum of A[i][j] * x[j] for all j except i
            for j in range(A.shape[1]):
                if j != i:
                    sigma += A[i][j] * x[j]

            # Update the i-th component of x using the relaxation formula
            x[i] = (1 - omega) * x[i] + (omega / A[i][i]) * (b[i] - sigma)

        # Recompute the residual
        residual = np.linalg.norm(np.matmul(A, x) - b)

    return x


if __name__ == '__main__':
    print('SOR method')

    # Define the coefficient matrix A and the right-hand side vector b
    #A: np.ndarray = np.array([[ 4, -1, -6,  0],
    #                          [-5, -4, 10,  8],
    #                          [ 0,  9,  4, -2],
    #                          [ 1,  0, -7,  5]])
    #b: np.ndarray = np.array([2, 21, -12, -6])
    # Initial guess for the solution
    #x0: np.ndarray = np.zeros(4)
    
    # Print the input values
    #print(A, b, x0, '', sep='\n')

    # Call the relaxation function to solve the system
    #x: np.ndarray = relaxation(A, b, x0)
    
    # Print the resulting solution
    #print(x)
