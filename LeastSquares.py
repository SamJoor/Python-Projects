import numpy as np
from copy import deepcopy

def least_squares_qr(X, y):
    """
    Implements QR decomposition using Householder transformations to solve the least-squares problem.
    """
    # Step 1: Make deep copies of X and y to avoid changing the original data
    X = deepcopy(X)
    y = deepcopy(y)
    
    # Get the number of rows (n) and columns (k) of X
    n, k = X.shape
    
    # Loop over each column in X to perform Householder transformations
    for i in range(k):
        # Extract the part of the i-th column below the diagonal
        x_i = X[i:, i]
        
        # Calculate the norm of the vector and the alpha value for the transformation
        alpha = np.sign(X[i, i]) * np.linalg.norm(x_i)
        
        # Create vector v for the Householder reflection
        v = x_i.copy()
        v[0] += alpha
        
        # Normalize v to make it a unit vector (ensure it doesn't change the direction)
        v = v / np.linalg.norm(v)
        
        # Apply the Householder transformation to the columns of X from i onwards
        for j in range(i, k):
            X_j = X[i:, j]
            # Update X by reflecting across the Householder vector
            X[i:, j] = X_j - 2 * v * np.dot(v, X_j)
        
        # Apply the same transformation to the vector y
        y_i = y[i:]
        y[i:] = y_i - 2 * v * np.dot(v, y_i)
    
    # Solve the resulting upper triangular system R * b = Q^T * y
    R = X[:k, :k]
    b = np.linalg.solve(R, y[:k])
    
    return b

# Problem 5: Validation with Provided Data (Goal Section Validation)
print("### Problem 5: Validation with Provided Data ###")

X_test = np.array([
    [1, 0.500, 0.390],
    [1, 0.720, 0.575],
    [1, 0.430, 0.350],
    [1, 0.420, 0.325],
    [1, 0.625, 0.505]
])

y_test = np.array([0.3145, 0.9515, 0.1675, 0.1000, 0.4960])

# Calculate the least-squares solution for the validation data
b_solution_test = least_squares_qr(X_test, y_test)

# Print the solution for validation
print("Least-squares solution b (validation data):", b_solution_test)

# Problem 6: Apply Model to Full Dataset
print("\n### Problem 6: Apply Model to Full Dataset ###")

# Load the larger dataset from the CSV file, skipping the header row
data = np.loadtxt("C:\\Users\\samjo\\Downloads\\fullAbaloneData_F24.csv", delimiter=',', skiprows=1)

# Separate into design matrix X and vector y
X_full = data[:, :-1]  # All columns except the last (predictors and intercept)
y_full = data[:, -1]   # Last column is y (response variable)

# Calculate the least-squares solution for the full dataset
b_solution_full = least_squares_qr(X_full, y_full)

# Print the solution for the full dataset
print("Least-squares solution b (full dataset):", b_solution_full)
