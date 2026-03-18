import numpy as np
import matplotlib.pyplot as plt

def main():
    D = np.array([[0.7, 0.15, 0.05, 0.1],
                  [0.1, 0.6, 0.2, 0.05],
                  [0.1, 0.2, 0.7, 0.1],
                  [0.1, 0.05, 0.05, 0.75]])
    print("Original Matrix D:")
    print(D)
    print("\n\nColumn sums:")
    print(np.sum(D, axis=0))  # Ensure columns sum to 1
    eigenvalues_D, eigenvectors_D = np.linalg.eig(D)
    print("\n\nEigenvalues:")
    print(eigenvalues_D)
    print("\n\nEigenvectors: (will use the first eigenvector corresponding to the eigenvalue of 1)")
    print(eigenvectors_D)
    print("\n\nEigenvector normalized (also steady-state vector):")
    d_steady_state = eigenvectors_D[:, 0] / np.sum(eigenvectors_D[:, 0])  # Normalize the first eigenvector
    print(d_steady_state)
    print(f"""\nThis means there is going to be equilibrium zones of \n{d_steady_state[0] * 100:.2f}% in Core, \n{d_steady_state[1]*100:.2f}% in Inner Ring, \n{d_steady_state[2]*100:.2f}% in Suburbs, and \n{d_steady_state[3]*100:.2f}% in Exurbs.""")

    print("\n\n----------Sensitivity Analysis----------")
    newD = D.copy()
    newD[0, 2] = 0.15  # Change the value in the first row, third column to 0.15
    print("\n\nModified Matrix D:")
    print(newD)
    print("\n\nColumn sums of modified D:")
    print(np.sum(newD, axis=0))
    print("Rebalancing column 3...")
    lowerSumC3 = np.sum(newD[:, 2]) - newD[0, 2]  # Sum of column 3 without the modified value
    for i in range(3):
         if i!= 0:
              newD[i, 2] = newD[i, 2] * (1 - newD[0, 2])/lowerSumC3  # Rebalance the other values in column 3
    print("\n\nRebalanced Matrix D:")
    print(newD)
    print("\n\nColumn sums of rebalanced D:")
    print(np.sum(newD, axis=0))
    newD_eigenvalues, newD_eigenvectors = np.linalg.eig(newD)
    newD_steady_state = newD_eigenvectors[:, 0] / np.sum(newD_eigenvectors[:, 0])  # Normalize the eigenvector with eigenvalue 1
    print("\n\nNew steady-state vector after modification:")
    print(newD_steady_state)
    print(f"""\nThis means there is going to be new equilibrium zones of \n{newD_steady_state[0] * 100:.2f}% in Core, \n{newD_steady_state[1]*100:.2f}% in Inner Ring, \n{newD_steady_state[2]*100:.2f}% in Suburbs, and \n{newD_steady_state[3]*100:.2f}% in Exurbs.""")

    print("\n\n----------Rate of Convergence----------")
    pop_vector = np.array([0.25, 0.25, 0.25, 0.25])  # Initial population distribution
    all_pop_vectors = [pop_vector.copy()]  # To store population distribution at each iteration
    last_pop_vector = pop_vector.copy()
    num_iterations = 0
    years_per_iteration = 5  # Each iteration represents 5 years

    while True:
        new_pop_vector = D @ last_pop_vector  # Update population distribution
        num_iterations += 1
        all_pop_vectors.append(new_pop_vector.copy())
        if np.allclose(new_pop_vector, last_pop_vector, atol=5e-3):  # Check for convergence
            print("\n\nPopulation distribution has converged to the steady-state vector:")
            print(new_pop_vector)
            print(f"Number of iterations: {num_iterations}\nThus it took {num_iterations * years_per_iteration} years to converge (assuming accuracy within 0.5%).")
            break
        last_pop_vector = new_pop_vector

    # Plotting the population distribution over time
    x = np.arange(0, num_iterations + 1) * years_per_iteration
    all_pop_vectors = np.array(all_pop_vectors)
    y = all_pop_vectors
    plt.figure(figsize=(10, 6))
    plt.plot(x, y[:, 0], label='Core')
    plt.plot(x, y[:, 1], label='Inner Ring')
    plt.plot(x, y[:, 2], label='Suburbs')
    plt.plot(x, y[:, 3], label='Exurbs')
    plt.xlabel('Time (years)')
    plt.ylabel('Population Distribution')
    plt.title('Population Dynamics in Dallas')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
      main()
