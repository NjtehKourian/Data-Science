
def is_int(number):
    assert isinstance(number, int), "input must be of type int"


def is_float(number):
    assert isinstance(number, float), "input must be of type float"


def is_number(number):
    assert isinstance(number, (int, float)), "input must be a number"


def is_list(obj):
    assert isinstance(obj, list), "input must be a list"


def are_lists(*objs):
    for obj in objs:
        is_list(obj)


def are_vectors(*objs):
    for obj in objs:
        is_vector(obj)


def eq_len(*objs):
    len0 = len(objs[0])
    for obj in objs[1:]:
        assert len(obj) == len0, "the vectors passed must be of equal length"


def are_vec_eq_len(*objs):
    are_vectors(*objs)
    eq_len(*objs)


def is_vector(vector):
    is_list(vector)
    for el in vector:
        is_number(el)


def is_matrix(matrix):
    is_list(matrix)
    are_vectors(*[el for el in matrix])
