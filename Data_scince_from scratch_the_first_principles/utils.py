#
import math


def vector_addition(v, w):
    # returns the element-wise sum of two vectors
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def vector_subtraction(v, w):  # where v and w are vectors of the same size
    # returns the element-wise sum of two vectors
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def multiple_vector_sum(vectors):  # vectors is a list containing vectors
    # return the element-wise sum of multiple vectors
    summed = vectors[0]
    for i in range(len(vectors)-1):
        summed = vector_addition(summed, vectors[i+1])
    return summed


def product_num_vec(v, a):  # where v is the vector and a is a scalar
    # returns the product of a vector and a scalar
    return [a*v_i for v_i in v]


def mean_vector(vectors):  # vectors is a list containing vectors
    # returns the element-wise mean of multiple vectors
    return product_num_vec(multiple_vector_sum(vectors), 1/len(vectors))


def scalar_product(v, w):  # v and w are vectors of the same length
    # returns the dot/scalar/inner product of two vectors
    return sum([v_i*w_i for v_i, w_i in zip(v, w)])


def sum_of_comp_sq(v):
    # returns the sum of the squares of the elements
    return scalar_product(v, v)


def vec_magnitude(v):
    # returns the magnitude of a vector
    return math.sqrt(sum_of_comp_sq(v))


def dist_between_vectors(v, w):
    # returns the distance between the endpoints of two vectors
    return vec_magnitude(vector_subtraction(v, w))


def tensor_shape(tensor, shape=None):
    # creating a new array everytime the function is called and recursively calls itself to find the dimensionality
    # of each rank of the tensor assuming each list within a container-list is of the same dimensionality
    if shape is None:
        shape = []
    if type(tensor) != list:
        return tuple(shape)
    else:
        shape.append(len(tensor))
        return tensor_shape(tensor[0], shape)


def row_of_matrix(matrix, i):
    # returns the i-th row of the matrix
    return matrix[i]  # matrix[i] is already the i-th row of the matrix


def column_of_matrix(matrix, i):
    # returns the i-th column of the matrix
    return [matrix_row[i] for matrix_row in matrix]


def build_matrix(shape: tuple, el_func):
    return [[el_func(i, j) for i in range(shape[1])] for j in range(shape[0])]  # took some hints from the book


def square_sum_of_indices(i, j):
    return i**2 + j**2


def identity_matrix(i, j):
    return int(i == j)
