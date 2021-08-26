#
import math
from validations import *


def vector_addition(v, w):
    """

    :param v: first vector
    :param w: second vector
    :return: sum of vectors
    """
    are_vec_eq_len(v, w)
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def vector_subtraction(v, w):
    """

    :param v: first vector
    :param w: second vector
    :return: sum of vectors
    """
    are_vec_eq_len(v, w)
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def multiple_vector_sum(*vectors):
    """

    :param vectors: the vectors to sum
    :return: te sum of the vectors passed
    """
    are_vec_eq_len(*vectors)
    summed = vectors[0]
    for i in range(len(vectors)-1):
        summed = vector_addition(summed, vectors[i+1])
    return summed


def product_num_vec(v, a):
    """

    :param v: the vector
    :param a: the scalar
    :return: the product of the vector with the scalar
    """
    is_vector(v)
    return [a*v_i for v_i in v]


def mean_vector(*vectors):
    """

    :param vectors: the vectors
    :return: the elementwise mean of the vectors
    """
    return product_num_vec(multiple_vector_sum(*vectors), 1/len(vectors))


def scalar_product(v, w):
    """

    :param v: first vector
    :param w: second vector
    :return: returns the dot product of the two vectors
    """
    are_vec_eq_len(v, w)
    return sum([v_i*w_i for v_i, w_i in zip(v, w)])


def sum_of_comp_sq(v):
    return scalar_product(v, v)


def vec_magnitude(v):
    return math.sqrt(sum_of_comp_sq(v))


def dist_between_vectors(v, w):
    """

    :param v: first vector
    :param w: second vector
    :return: the distance between the given vectors (coordinate points in this case)
    """
    return vec_magnitude(vector_subtraction(v, w))


# ToDo fix the errors
def tensor_shape(tensor, shape=None):
    """

    :param tensor: the tensor the shape of which to be found
    :param shape: the shape of the tensor so far in the recursion
    :return: the shape of the tensor
    """
    if shape is None:
        shape = []
    if type(tensor) != list:
        return tuple(shape)
    else:
        eq_len(*[el for el in tensor])
        shape.append(len(tensor))
        return tensor_shape(tensor[0], shape)


def row_of_matrix(matrix, i):
    """

    :param matrix: the matrix to be analyzed
    :param i: the required row
    :return: the specified row of the given matrix
    """
    return matrix[i]


def column_of_matrix(matrix, i):
    """

    :param matrix: the matrix to be analyzed
    :param i: the required column
    :return: the specified column of the given matrix
    """
    is_matrix(matrix)
    return [matrix_row[i] for matrix_row in matrix]


def build_matrix(shape: tuple, el_func):
    """

    :param shape: the shape the matrix will have
    :param el_func: the function that determines the elements of the matrix
    :return: a matrix created based of the passed function
    """
    return [[el_func(i, j) for i in range(shape[1])] for j in range(shape[0])]  # took some hints from the book


def square_sum_of_indices(i, j):
    return i**2 + j**2


def identity_matrix(i, j):
    return int(i == j)
