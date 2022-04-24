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
    return None


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
    print('Hello, world!\n')
