#
import math
from validations import *
from collections import Counter


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
    return element_sum([v_i*w_i for v_i, w_i in zip(v, w)])


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


def element_sum(data):
    """
    basically the sum() function
    :param data: dataset to be analyzed
    :return: the sum of all the elements
    """
    is_vector(data)
    total = data[0]
    for el in data[1:]:
        total += el
    return total


def mean(data):
    """

    :param data: dataset to be analyzed
    :return: the mean of the dataset
    """
    is_vector(data)
    return element_sum(data)/len(data)


def median(data):
    """

    :param data: dataset to be analyzed
    :return: the median of the dataset
    """
    is_vector(data)
    sorted_data = sorted(data)
    mid = len(data)//2
    if len(data) % 2 == 1:
        return sorted_data[mid]
    return (sorted_data[mid+1] + sorted_data[mid])/2


def quantile(data, percentage):
    """

    :param data: dataset to be analyzed
    :param percentage: the percentile
    :return: the quantile of the dataset at the percentile
    """
    is_vector(data)
    is_number(percentage)
    assert percentage > 0 and percentage < 1, "the percentage must be in range (0, 1)"
    index = int(percentage*len(data))
    return sorted(data)[index]


def mode(data):
    """

    :param data: dataset to be analyzed
    :return: the most common value in the dataset
    """
    is_vector(data)
    counted = Counter(data)
    max_occ = max(counted.values())
    return [key for key in counted if counted[key] == max_occ]


def mean_dif(data):
    """

    :param data: dataset to be analyzed
    :return: a list containing the difference of each element of the data with the mean of the set
    """
    means = mean(data)
    return [x - means for x in data]


def variance(data):
    """

    :param data: dataset to be analyzed
    :return: the variance of the dataset
    """
    is_vector(data)
    variances = mean_dif(data)
    return sum_of_comp_sq(variances)/(len(data)-1)


def std(data):
    """

    :param data: dataset to be analyzed
    :return: the standard deviation of the sample
    """
    return math.sqrt(variance(data))


def covariance(data1, data2):
    """

    :param data1: first sample
    :param data2: second sample
    :return: the covariance of the two samples
    """
    eq_len(data1, data2)
    x, y = mean_dif(data1), mean_dif(data2)
    numerator = scalar_product(x, y)
    return numerator/(len(x)-1)


def correlation(data1, data2):
    eq_len(data1, data2)
    std1, std2 = std(data1), std(data2)
    if std1 != 0 and std2 != 0:
        return covariance(data1, data2) / (std1 * std2)
    return 0
