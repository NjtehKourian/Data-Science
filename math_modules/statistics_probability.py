from utils import Validation as val, Number
from math_modules.linear_algebra import sum_of_comp_sq, scalar_product
import math
from collections import Counter
from typing import List


class Statistics:
    def __init__(self):
        pass


    @staticmethod
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
        raise Exception("the argument passed must be a list and its elements numbers.")

    @staticmethod
    def mean(data: list[Number]) -> Number:
        """

        :param data: dataset to be analyzed
        :return: the mean of the dataset
        """
        if val.is_vector(data):
            return Statistics.element_sum(data)/len(data)
        raise Exception("the argument passed must be a list and its elements numbers.")

    @staticmethod
    def median(data: List[Number]) -> Number:
        """

        :param data: dataset to be analyzed
        :return: the median of the dataset
        """
        if val.is_vector(data):
            sorted_data = sorted(data)
            mid = len(data)//2
            if len(data) % 2 == 1:
                return sorted_data[mid]
            return (sorted_data[mid+1] + sorted_data[mid])/2
        raise Exception("the argument passed must be a list and its elements numbers.")

    @staticmethod
    def quantile(data: List[Number], percentage: Number) -> Number:
        """

        :param data: dataset to be analyzed
        :param percentage: the percentile
        :return: the quantile of the dataset at the percentile
        """
        if val.is_vector(data):
            if val.is_number(percentage):
                assert percentage > 0 and percentage < 1, "the percentage must be in range (0, 1)"
                index = int(percentage*len(data))
                return sorted(data)[index]
            raise Exception("the percentage passed must be a number in range(0, 1)")
        raise Exception("the data passed must be a list and its elements numbers.")

    @staticmethod
    def mode(data: List[Number]) -> List[Number]:
        """

        :param data: dataset to be analyzed
        :return: the most common value in the dataset
        """
        if val.is_vector(data):
            counted = Counter(data)
            max_occ = max(counted.values())
            return [key for key in counted if counted[key] == max_occ]
        raise Exception("the argument passed must be a list and its elements numbers.")

    @staticmethod
    def mean_dif(data: List[Number]) -> List[Number]:
        """

        :param data: dataset to be analyzed
        :return: a list containing the difference of each element of the data with the mean of the set
        """
        means = Statistics.mean(data)
        return [x - means for x in data]

    @staticmethod
    def variance(data: List[Number]):
        """

        :param data: dataset to be analyzed
        :return: the variance of the dataset
        """
        if val.is_vector(data):
            variances = Statistics.mean_dif(data)
            return sum_of_comp_sq(variances)/(len(data)-1)
        raise Exception("the argument passed must be a list and its elements numbers.")

    @staticmethod
    def std(data):
        """

        :param data: dataset to be analyzed
        :return: the standard deviation of the sample
        """
        return math.sqrt(Statistics.variance(data))

    @staticmethod
    def covariance(data1: List[Number], data2: List[Number]) -> Number:
        """

        :param data1: first sample
        :param data2: second sample
        :return: the covariance of the two samples
        """
        if val.eq_len(data1, data2):
            x, y = Statistics.mean_dif(data1), Statistics.mean_dif(data2)
            numerator = scalar_product(x, y)
            return numerator/(len(x)-1)
        raise Exception("both lists must be of the same length. You passed lengts of {} and {}".format(len(data1), len(data2)))
    @staticmethod
    def correlation(data1: List[Number], data2: List[Number]) -> Number:
        if val.eq_len(data1, data2):
            std1, std2 = Statistics.std(data1), Statistics.std(data2)
            if std1 != 0 and std2 != 0:
                return Statistics.covariance(data1, data2) / (std1 * std2)
            return 0
        raise Exception("both lists must be of the same length. You passed lengts of {} and {}".format(len(data1), len(data2)))