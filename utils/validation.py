
class Validation:
    def __init__(self):
        pass

    @staticmethod
    def is_int(number) -> bool:
        return isinstance(number, int)

    @staticmethod
    def is_float(number) -> bool:
        return isinstance(number, float)

    @staticmethod
    def is_number(number) -> bool:
        return isinstance(number, (int,  float))

    @staticmethod
    def is_list(obj) -> bool:
        return isinstance(obj, list)

    @staticmethod
    def are_lists(*args) -> bool:
        return all(Validation.is_list(arg) for arg in args)

    @staticmethod
    def is_vector(vector) -> bool:
        return Validation.is_list(vector) and all(Validation.is_number(el) for el in vector)

    @staticmethod
    def are_vectors(*args) -> bool:
        return all(Validation.is_vector(arg) for arg in args)

    @staticmethod
    def eq_len(*args) -> bool:
        len0 = len(args[0])
        return all(len(arg) == len0 for arg in args[1:])

    @staticmethod
    def are_vec_eq_len(*args) -> bool:
        return Validation.are_vectors(*args) and Validation.eq_len(*args)

    @staticmethod
    def is_matrix(matrix) -> bool:
        # here i put my el in list and spread the list because otherwise it passes a generator object to the function
        return Validation.is_list(matrix) and Validation.are_vectors(*[el for el in matrix])
