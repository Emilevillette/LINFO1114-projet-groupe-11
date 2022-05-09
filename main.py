import numpy as np


def pageRankLinear(A, alpha, v):
    """Page Rank linear method. See consignes.pdf for details

    :param A: Adjacency matrix of a directed graph G
    :type A: np.matrix
    :param alpha: teleportation parameter 0 <= alpha <= 1 (default: 0.9)
    :type alpha: float
    :param v: personalization vector
    :type v: np.array
    :return: a vector containing the importance scores of the nodes sorted in the same ordered as the input adjacency matrix A
    :rtype: np.array
    """
    outdegree = np.zeros(len(A))

    # Determine the transition probabilities matrix from a and the out-degree of a
    P = np.zeros(shape=(len(A), len(A[0])))
    for i in range(len(A)):
        for k in range(len(A[0])):
            if A[i][k] != 0:
                outdegree[i] += A[i][k]
        for j in range(len(A[0])):
            P[i][j] = A[i][j] / outdegree[i]
    print(np.transpose(np.identity(len(A)) - alpha * P))
    print((1 - alpha) * v)
    # solve the linear system from slide 144 of chapter 10 (see Moodle, author: Marco Saerens)
    return np.array(np.linalg.solve(np.transpose(np.identity(len(A)) - alpha * P), (1 - alpha) * v))


def pageRankPower(A, alpha, v):
    """Page Rank power method. See consignes.pdf for details

    :param A: Adjacency matrix of a directed graph G
    :type A: np.matrix
    :param alpha: teleportation parameter 0 < alpha < 1 (default: 0.9)
    :type alpha: float
    :param v: personalization vector
    :type v: np.array
    :return: a vector containing the importance scores of the nodes sorted in the same ordered as the input adjacency matrix A
    :rtype: np.array
    """
    x = np.copy(v)
    outdegree = np.zeros(len(A))

    # Determine the transition probabilities matrix from a and the out-degree of a
    P = np.zeros(shape=(len(A), len(A[0])))
    for i in range(len(A)):
        for k in range(len(A[0])):
            if A[i][k] != 0:
                outdegree[i] += A[i][k]
        for j in range(len(A[0])):
            P[i][j] = A[i][j] / outdegree[i]

    # Get the Google matrix (transposed to apply power method)
    G = np.transpose(alpha * P + (1 - alpha) * np.ones(len(P)) * np.transpose(v))

    # Copy to determine when to stop iterating
    temp_x = np.copy(x)

    # Iterate on the left eigenvector until it converges
    stop_loop = False
    cnt = 0
    while not stop_loop:
        x = np.matmul(G, x)
        for i in range(len(x)):
            # We could add a tolerance parameter to the function (not specified in the instructions so not added here)
            # We believe 10e-6 is a reasonable tolerance factor
            # Stops iterating when the difference between the previous matrix and the current is below a given threshold
            if abs(x[i] - temp_x[i]) < 0.000001:
                stop_loop = True
            else:
                stop_loop = False
                break
        # print(f"OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO {cnt}")
        # np.savetxt(f"foo{cnt}.xslx", x, delimiter="    ")
        # print(x)
        # Replace temp_x with new x
        temp_x = np.copy(x)
        cnt += 1
    return np.array(x)


if __name__ == '__main__':
    # Main method as asked in the instructions (alpha = 0.9)
    adj = np.genfromtxt('Adjacency_matrix.csv', delimiter=',')
    pers = np.genfromtxt('VecteurPersonnalisation_Groupe11.csv', delimiter=',')
    print(pageRankLinear(adj, 0.9, pers))
    print(pageRankPower(adj, 0.9, pers))
