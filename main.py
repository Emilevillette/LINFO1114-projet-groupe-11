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
    P = np.zeros(shape=(len(A), len(A[0])))
    for i in range(len(A)):
        for k in range(len(A[0])):
            if A[i][k] != 0:
                outdegree[i] += A[i][k]
        for j in range(len(A[0])):
            P[i][j] = A[j][i] / outdegree[i]
    newA = np.matmul(np.transpose(np.identity(len(A)) - alpha * P), A/np.linalg.norm(A))
    B = v
    return np.linalg.solve(newA, B)



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
    return None


if __name__ == '__main__':
    adj = np.genfromtxt('Adjacency_matrix.csv', delimiter=',')
    pers = np.genfromtxt('VecteurPersonnalisation_Groupe11.csv', delimiter=',')
    print(pageRankLinear(adj, 0.9, pers))
    #print(sum(pageRankLinear(adj, 0.9, pers)))
