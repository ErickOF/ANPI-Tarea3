# Numerical Analysis for Engineering - Assignment 3 - Successive Over-Relaxation (SOR) Method

# **Costa Rica Institute of Technology**

**CE3102:** Numerical Analysis for Engineering  

Computer Engineering

Semester II - 2019

## **Assignment 3**

### Questions

1. Implement a function for the iterative relaxation method in Octave or Python to solve a system of linear equations of the form $Ax = b$. The name of this function should be `relajacion`. Indicate which are the initial parameters and which are the output parameters.

2. Implement a function for the natural cubic spline interpolation method in Octave or Python, considering the ordered pairs $(x_0, y_0), (x_1, y_1), ..., (x_n, y_n)$. The name of this function should be `trazador_cubico`. Note: The implementation of this method requires solving a system of linear equations. For this, use the iterative relaxation method. Indicate which are the initial parameters and which are the output parameters.

3. Consider the following table, which represents ordered pairs of the form $(x_n, y_n)$:

   | n  | \(x_n\) | \(y_n\)  |
   |----|---------|----------|
   | 0  | -5      | 0        |
   | 1  | -4.5    | 0.0707   |
   | 2  | -4      | 0        |
   | 3  | -3.5    | -0.0909  |
   | 4  | -3      | 0        |
   | 5  | -2.5    | 0.1273   |
   | 6  | -2      | 0        |
   | 7  | -1.5    | -0.2122  |
   | 8  | -1      | 0        |
   | 9  | -0.5    | 0.6366   |
   | 10 | 0       | 1.0000   |
   | 11 | 0.5     | 0.6366   |
   | 12 | 1       | 0        |
   | 13 | 1.5     | -0.2122  |
   | 14 | 2       | 0        |
   | 15 | 2.5     | 0.1273   |
   | 16 | 3       | 0        |
   | 17 | 3.5     | -0.0909  |
   | 18 | 4       | 0        |
   | 19 | 4.5     | 0.0707   |
   | 20 | 5       | 0        |

Implement a script that plots the natural cubic spline passing through the above points. The name of the file should be `pregunta1`.
