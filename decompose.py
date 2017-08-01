import numpy as np


def qr_mgs_decompose(matrix: np.array) -> (np.array, np.array):
    """
    For n x m matrix return Q1 and R1 components of QR decomposition using
    the modified Gram-Schmidt process, where R1 is n x n upper triangular
    and Q1 is m x n and have orthogonal columns.
    """
    n = matrix.shape[1]
    q1 = np.array(matrix, dtype='float64')
    r1 = np.zeros((n, n))
    for k in range(n):
        a_k = q1[..., k]
        r1[k,k] = np.linalg.norm(a_k)
        a_k /= r1[k, k]
        for i in range(k+1, n):
            a_i = q1[..., i]
            r1[k,i] = np.transpose(a_k) @ a_i
            a_i -= r1[k, i] * a_k
    return q1, r1


def test_diagonal():
    q, r = qr_mgs_decompose(np.diag([-1, 2, 4]))
    assert np.allclose(q, np.diag([-1, 1, 1]))
    assert np.allclose(r, np.diag([1, 2, 4]))


def test_orthogonal():
    orthogonal = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1]]).T
    q, r = qr_mgs_decompose(orthogonal)
    assert np.allclose(q, 0.5 * orthogonal)
    assert np.allclose(r, np.diag([2, 2, 2]))


def test_matrix_eq_qr_and_q_orthogonal():
    matrix = np.array([[1, 0, -1], [-2, 1, 3], [5, 4, -3], [3, 0, -4]])
    q, r = qr_mgs_decompose(matrix)
    assert np.allclose(matrix, q @ r)
    assert np.allclose(q.T @ q, np.eye(3))
