import numpy as np
from copy import deepcopy
from matplotlib.pyplot import plot, show, subplots
import matplotlib
import matplotlib.pyplot as plt

# Force TkAgg backend for matplotlib
matplotlib.use('TkAgg')  # Ensure a working backend for displaying plots

# Enable interactive mode
plt.ion()

# helper
def drawPlot(pc1_scores, pc2_scores):
    '''
    Create an ordination of PC scores.
    
    Args:
        pc1 = []double; an array of PC scores.
        pc2 = []double; a second array of PC scores.
    
    Return
    Void, but it will display the requested ordination and pause execution until the window is closed.
    '''
    fig, ax = subplots(1, 1)
    ax.plot(pc1_scores, pc2_scores, 'ko')
    ax.axis('equal')
    fig.tight_layout()
    plt.show(block=True)  

# helper
def loadData(file_path, skiprows=1, delimiter=','):
    return np.loadtxt(file_path, skiprows=skiprows, delimiter=delimiter)

def power_iteration(A, max_iter=1000, tol=1e-8):
    # Step 1: random vector
    n, _ = A.shape
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)  # normalize initial vector

    for _ in range(max_iter):
        # Step 2: matrix vector multiplication
        Av = np.matmul(A, v)

        # Step 3: Normalize  vector
        v_new = Av / np.linalg.norm(Av)

        # Step 4: check convergange
        if np.linalg.norm(v_new - v) < tol:
            break

        v = deepcopy(v_new)  # update

    # Step 5: compute associated eigenvalue
    eigenvalue = np.dot(v.T, np.dot(A, v)) / np.dot(v.T, v)

    return eigenvalue, v

def deflation(A, v, eigenvalue):
    # outer product of eigenvector with itself
    outer_v = np.outer(v, v)
    # deflate the matrix
    A_deflated = A - eigenvalue * outer_v
    return A_deflated

def find_two_largest_eigenpairs(A, max_iter=1000, tol=1e-8):
    # Step 1: find the largest eigenvalue and corresponding eigenvector
    eigenvalue_1, v1 = power_iteration(A, max_iter=max_iter, tol=tol)

    # Step 2: deflate the matrix
    A_deflated = deflation(A, v1, eigenvalue_1)

    # Step 3: find the second largest eigenvalue and eigenvector on deflated matrix
    eigenvalue_2, v2 = power_iteration(A_deflated, max_iter=max_iter, tol=tol)

    return (eigenvalue_1, v1), (eigenvalue_2, v2)

# example usage with a sample symmetric matrix
A = np.array([[1, 0.00143, -0.31938],
              [0.00143, 1, 0.77224],
              [-0.31938, 0.77224, 1]])

# finding the two largest eigenpairs
eigenpair_1, eigenpair_2 = find_two_largest_eigenpairs(A)

print(eigenpair_1, eigenpair_2)

# function to compute PC scores and plot using helper plot function
def compute_pc_scores(data, covariance_matrix):
    # Step 1: get the largest and second-largest eigenpairs
    (eigenvalue_1, v1), (eigenvalue_2, v2) = find_two_largest_eigenpairs(covariance_matrix)

    # Step 2: project data onto the first two principal components
    pc1_scores = np.dot(data, v1)
    pc2_scores = np.dot(data, v2)

    # Step 3: print the eigenvalues
    print(f"Eigenvalue 1 (λ1): {eigenvalue_1}")
    print(f"Eigenvalue 2 (λ2): {eigenvalue_2}")

    # Step 4: use helper function to draw the plot
    drawPlot(pc1_scores, pc2_scores)

    return pc1_scores, pc2_scores

# function to process 10-variable data
def compute_pc_scores_10var(data_file, covariance_file):
    # load data and covariance matrix using the helper function
    data = loadData(data_file, skiprows=1, delimiter=',')
    covariance_matrix = loadData(covariance_file, skiprows=0, delimiter=',')

    # compute the PC scores and plot them
    pc1_scores, pc2_scores = compute_pc_scores(data, covariance_matrix)
    return pc1_scores, pc2_scores

# exsample usage for 3-variable data 
data_3var = loadData(r"C:\Users\samjo\Downloads\Lab5_ThreeVariableSugarData_F24 (1).csv", skiprows=1, delimiter=',')
cov_matrix_3var = loadData(r"C:\Users\samjo\Downloads\Lab5_ThreeVariableSugarBeanCovariance_F24 (1).csv", skiprows=0, delimiter=',')

# compute and plot the PC scores for the 3-variable dataset 
pc1_scores_3var, pc2_scores_3var = compute_pc_scores(data_3var, cov_matrix_3var)

# example usage for 10-variable data 
data_file_10var = r"C:\Users\samjo\Downloads\Lab5_FullSugarBeanData_F24.csv"
cov_matrix_file_10var = r"C:\Users\samjo\Downloads\Lab5_FullSugarBeanCovariance_F24.csv"

# compute and plot the PC scores for the 10-variable dataset 
pc1_scores_10var, pc2_scores_10var = compute_pc_scores_10var(data_file_10var, cov_matrix_file_10var)