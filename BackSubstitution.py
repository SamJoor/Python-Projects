import numpy as np
from copy import deepcopy

def back_substitution(A, b):
    # Use deep copy to avoid modification
    x = deepcopy(b)
    
    # Get number of rows
    n = len(b)
    
    # Do back-substitution
    for i in range(n-1, -1, -1):  # Loop n-1 down to 0
        x[i] = x[i] / A[i, i]  # Divide by diagonal element
        for j in range(i-1, -1, -1):  # Subtract elements above diagonal
            x[j] = x[j] - A[j, i] * x[i]
    
    return x

# Example usage 
A = np.array([[350, 310, 290, 250],
              [0, 295, 275, 235],
              [0, 0, 245, 190],
              [0, 0, 0, 150]], dtype=float)

b = np.array([61630, 45795, 26300, 6450], dtype=float)

# Call function
x = back_substitution(A, b)

# Output
print("Solution vector x:", x)

# New matrix A and vector b from Step 6
A = np.array([[-12.8, 8.2, -3.4, 5.2, -1.3, 4.2],
              [0.0, 14.1, 9.2, -5.1, 8.4, 7.1],
              [0.0, 0.0, 11.9, 4.1, -3.7, 6.9],
              [0.0, 0.0, 0.0, -16.2, 8.3, -1.7],
              [0.0, 0.0, 0.0, 0.0, -11.2, 8.3],
              [0.0, 0.0, 0.0, 0.0, 0.0, 9.9]])

b = np.array([-224.94, 128.70, 135.62, 167.37, 28.07, 76.23])

# Call the back-substitution function
x = back_substitution(A, b)

# Output the solution
print("Solution vector x:", x)


