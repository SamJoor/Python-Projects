import numpy as np
from copy import deepcopy

def lu_decomposition(A):
    """
    Perform LU decomposition using Gaussian elimination.
    
    Parameters:
    A (numpy.ndarray): A two-dimensional numpy array representing matrix A.
    
    Returns:
    L (numpy.ndarray): Lower triangular matrix L.
    U (numpy.ndarray): Upper triangular matrix U.
    """
    n = A.shape[0]
    U = deepcopy(A)  # Create a new array for U (deep copy of A)
    L = np.identity(n)  # Initialize L as an identity matrix

    # Loop over each row
    for i in range(n):
        # Perform Gaussian elimination to form U and store multipliers in L
        for j in range(i + 1, n):
            # Multiplier for the row
            L[j, i] = U[j, i] / U[i, i]
            # Update the U matrix rows
            U[j, i:] = U[j, i:] - L[j, i] * U[i, i:]

    return L, U

# Example usage:
A = np.array([[73, 15, 9, 12],
              [16, 27, 14, 9],
              [10, 6, 16, 5],
              [11, 8, 10, 14]])

L, U = lu_decomposition(A)

# Print L and U matrices
print("Lower triangular matrix L:")
print(L)
print("\nUpper triangular matrix U:")
print(U)

# Test the LU decomposition function with the matrix A from Step 1
A = np.array([[73, 15, 9, 12],
              [16, 27, 14, 9],
              [10, 6, 16, 5.],
              [11, 8, 10, 14]])

# Perform LU decomposition
L, U = lu_decomposition(A)

# Display the results
print("Lower triangular matrix L:")
print(L)
print("\nUpper triangular matrix U:")
print(U)

def forward_substitution(L, b):
    """
    Perform forward substitution to solve L * y = b.
    
    Parameters:
    L (numpy.ndarray): Lower triangular matrix L.
    b (numpy.ndarray): Right-hand side vector b.
    
    Returns:
    y (numpy.ndarray): Solution vector y.
    """
    n = L.shape[0]
    y = np.zeros_like(b, dtype=np.double)
    
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    return y

def backward_substitution(U, y):
    """
    Perform backward substitution to solve U * x = y.
    
    Parameters:
    U (numpy.ndarray): Upper triangular matrix U.
    y (numpy.ndarray): Solution vector from forward substitution.
    
    Returns:
    x (numpy.ndarray): Solution vector x.
    """
    n = U.shape[0]
    x = np.zeros_like(y, dtype=np.double)
    
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x

# Right-hand side vector b
b = np.array([1139.50, 645.00, 403.50, 454.75])

# Perform forward and backward substitution
y = forward_substitution(L, b)
x = backward_substitution(U, y)

# Print the solution (gift prices for builders, painters, electricians, plumbers)
print("Gift prices:")
print(x)

# New right-hand side vector b (with updated budgets)
b_new = np.array([1443.90, 828.90, 505.35, 578.50])

# Perform forward and backward substitution with the new budgets
y_new = forward_substitution(L, b_new)
x_new = backward_substitution(U, y_new)

# Print the new solution (updated gift prices)
print("Updated gift prices:")
print(x_new)

