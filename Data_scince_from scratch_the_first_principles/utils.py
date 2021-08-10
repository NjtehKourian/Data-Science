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
