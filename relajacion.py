import numpy as np


def relajacion(A, b, x0, omega=0.5, tol=1e-8):
    """
    This is an implementation of the succesive over-relaxation method.

    Inputs:
    
      A: nxn numpy matrix
    
      b: n dimensional numpy vector
    
      omega: relaxation factor
    
      x0: an initial value
    
      tol: tolerance
    
    Returns:
    
      x: solution matrix
    """
    x = x0.copy()
    residual = np.linalg.norm(np.matmul(A, x) - b)
    while residual > tol:
        for i in range(A.shape[0]):
            sigma = 0
            for j in range(A.shape[1]):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (1 - omega) * x[i] + (omega / A[i][i]) * (b[i] - sigma)
        residual = np.linalg.norm(np.matmul(A, x) - b)
    return x


if __name__ == '__main__':
    print('SOR method working, put your code below')
    # Example, uncomment to see
    # A = np.array([[ 4, -1, -6,  0],
    #            [-5, -4, 10,  8],
    #            [ 0,  9,  4, -2],
    #            [ 1,  0, -7,  5]])
    # b = np.array([2, 21, -12, -6])
    # x0 = np.zeros(4)
    # print(A, b, x0, '', sep='\n')

    # x = relajacion(A, b, x0)
    # print(x)
