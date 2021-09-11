from utils import Validation as val, Number
import math
from typing import List, Tuple, Callable


def vector_addition(v: List, w: List) -> List:
    """

    :param v: first vector
    :param w: second vector
    :return: sum of vectors
    """
    if val.are_vec_eq_len(v, w):
        return [v_i + w_i for v_i, w_i in zip(v, w)]


def vector_subtraction(v: List, w: List) -> List:
    """

    :param v: first vector
    :param w: second vector
    :return: sum of vectors
    """
    if val.are_vec_eq_len(v, w):
        return [v_i - w_i for v_i, w_i in zip(v, w)]


def multiple_vector_sum(*vectors: List) -> List:
    """

    :param vectors: the vectors to sum
    :return: te sum of the vectors passed
    """
    if val.are_vec_eq_len(*vectors):
        summed = vectors[0]
        for i in range(len(vectors)-1):
            summed = vector_addition(summed, vectors[i+1])
        return summed


def product_num_vec(v: List, a: float) -> List:
    """

    :param v: the vector
    :param a: the scalar
    :return: the product of the vector with the scalar
    """
    if val.is_vector(v):
        return [a*v_i for v_i in v]


def mean_vector(*vectors: Tuple[List]) -> List:
    """

    :param vectors: the vectors
    :return: the elementwise mean of the vectors
    """
    return product_num_vec(multiple_vector_sum(*vectors), 1/len(vectors))


def scalar_product(v: List, w: List) -> Number:
    """

    :param v: first vector
    :param w: second vector
    :return: returns the dot product of the two vectors
    """
    if val.are_vec_eq_len(v, w):
        return element_sum([v_i*w_i for v_i, w_i in zip(v, w)])


def element_sum(data: List[Number]) -> Number:
    """
    basically the sum() function
    :param data: dataset to be analyzed
    :return: the sum of all the elements
    """
    if val.is_vector(data):
        total = data[0]
        for el in data[1:]:
            total += el
        return total


def sum_of_comp_sq(v: List) -> Number:
    return scalar_product(v, v)


def vec_magnitude(v: List) -> Number:
    return math.sqrt(sum_of_comp_sq(v))


def dist_between_vectors(v: List, w: List) -> Number:
    """

    :param v: first vector
    :param w: second vector
    :return: the distance between the given vectors (coordinate points in this case)
    """
    return vec_magnitude(vector_subtraction(v, w))


# ToDo fix the errors
def tensor_shape(tensor, shape=None) -> Tuple:
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
        if val.eq_len(*[el for el in tensor]):
            shape.append(len(tensor))
            return tensor_shape(tensor[0], shape)


def row_of_matrix(matrix: List[List], i: Number) -> List:
    """

    :param matrix: the matrix to be analyzed
    :param i: the required row
    :return: the specified row of the given matrix
    """
    if val.is_matrix(matrix):
        return matrix[i]
    raise Exception("the matrix passed must be a list with lists of the same length as elements")


def column_of_matrix(matrix: List[List], i: Number) -> List:
    """

    :param matrix: the matrix to be analyzed
    :param i: the required column
    :return: the specified column of the given matrix
    """
    if val.is_matrix(matrix):
        return [matrix_row[i] for matrix_row in matrix]
    raise Exception("the matrix passed must be a list with lists of the same length as elements")


def build_matrix(shape: Tuple, el_func: Callable) -> List[List]:
    """

    :param shape: the shape the matrix will have
    :param el_func: the function that determines the elements of the matrix
    :return: a matrix created based of the passed function
    """
    return [[el_func(i, j) for i in range(shape[1])] for j in range(shape[0])]  # took some hints from the book


def square_sum_of_indices(i: int, j: int) -> int:
    return i**2 + j**2


def identity_matrix(i: int, j: int) -> int:
    return int(i == j)
