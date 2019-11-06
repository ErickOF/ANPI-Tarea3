import numpy as np


def relajacion(A, b, omega, x0, tol):
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
    x = x0[:]
    residual = np.linalg.norm(np.matmul(A, x) - b)
    while residual > tol:
        for i in range(A.shape[0]):
            sigma = 0
            for j in range(A.shape[1]):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (1 - omega) * x[i] + (omega / A[i][i]) * (b[i] - sigma)
        residual = np.linalg.norm(np.matmul(A, x) - b)
        #print('Residual: {0:10.6g}'.format(residual))
    return x


# == ESTE ES UN EJEMPLO DE USO, NO ADJUNTAR AL ENTREGABLE ==
tol = 1e-8
omega = 0.5

A = np.ones((4, 4))
A[0][0] = 4
A[0][1] = -1
A[0][2] = -6
A[0][3] = 0

A[1][0] = -5
A[1][1] = -4
A[1][2] = 10
A[1][3] = 8

A[2][0] = 0
A[2][1] = 9
A[2][2] = 4
A[2][3] = -2

A[3][0] = 1
A[3][1] = 0
A[3][2] = -7
A[3][3] = 5

b = np.ones(4)
b[0] = 2
b[1] = 21
b[2] = -12
b[3] = -6

x0 = np.zeros(4)

print(A)
print('\n')
print(b)
print('\n')
print(x0)
print('\n')

x = relajacion(A, b, omega, x0, tol)
print(x)
