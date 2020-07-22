import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial import polynomial


def interpolate_with_order(x, y, order):
    coef = lagrange(x, y).coef
    # pad with zeros
    result = np.zeros(order + 1)
    # scipy.interpolate.lagrange returns a numpy.poly1d instance,
    # which is deprecated in numpy,
    # the new numpy.polynomial.polynomial.Polynomial class should be used,
    # coefficients have to be reversed since ploy1d and Polynomial have different constructor parameter semantics
    result[:coef.shape[0]] = coef[::-1]
    return result


def lagrange_matrix(constraints):
    ploy_len = constraints.shape[0]
    ploy_num = constraints.shape[1]
    ploy = np.array([interpolate_with_order(range(1, ploy_len + 1),
                                            constraints.T[i], ploy_len - 1) for i in range(ploy_num)])
    return ploy


witness = np.array([1, 3, 35, 9, 27, 30]).T

constraints_A = np.array([[0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 1, 0, 0, 1, 0],
                          [5, 0, 0, 0, 0, 1]])
constraints_B = np.array([[0, 1, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0]])
constraints_C = np.array([[0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 1, 0, 0, 0]])

assert np.array_equal(np.dot(constraints_A, witness) *
                      np.dot(constraints_B, witness), np.dot(constraints_C, witness))

ploy_A = lagrange_matrix(constraints_A)
ploy_B = lagrange_matrix(constraints_B)
ploy_C = lagrange_matrix(constraints_C)

t = polynomial.polymul(np.dot(witness, ploy_A), np.dot(
    witness, ploy_B)) - polynomial.Polynomial(np.dot(witness, ploy_C))

Z = polynomial.Polynomial(polynomial.polyfromroots([1, 2, 3, 4]))

print('t is', t)
print('Z is', Z)

# ploydiv is a deprecated function with ploy1d as parameters,
# so coefficients have to be reversed
h, r = np.polydiv(t.coef[::-1], Z.coef[::-1])
print('h is', h[::-1])
print('r is', r[::-1], ', which should be zero')
