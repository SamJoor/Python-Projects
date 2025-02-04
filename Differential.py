# Step 2 Convert Algorithm 1 into a higher-order function called solver
def solver(f_prime, initial, t):
    #Step 3
    """Approximates the solution to a differential equation using a simple algorithm.

    Args:
     f_prime (function): The mathematical function representing the differential equation.
     initial (float): The initial value of the system.
     t (int): The number of time steps to include in the approximate solution.

    Returns:
     list: An approximate solution to the differential equation.
    """
    #Step 4: Convert differential equation into a Python function.
    y = [0] * t
    y[0] = initial
    for i in range(1, t):
        y[i] = y[i - 1] + f_prime(y[i - 1])
    return y

#Step 5: Write documentation for logistic
def logistic(x):
    """
    Represents the differential equation: f'(x) = 2x * ((100 - x) / 100).

    Args: x (float): The value 

    Returns: float: The instantaneous change in the system's value based on its current value."""
    return 2 * x * ((100 - x) / 100)
#Step 6 Create an if name ==‘ main ’ block
if __name__ == '__main__':
    # Step 7 use solver to approximate a solution
    initial_value = 40
    time_steps = 20
    solution = solver(logistic, initial_value, time_steps)

    #Step 9 Print solution
    print("Approximate Solution to the Logistic Growth Model:")
    print(solution)
