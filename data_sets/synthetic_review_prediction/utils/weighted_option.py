from functools import reduce
from random import random
from typing import List, Generic, TypeVar


T = TypeVar('T')


class WeightedOption(Generic[T]):
    def __init__(self, option: T=None, weight: float=1.0):
        self.weight = weight
        self.option = option


def choose_weighted_option(options: List[WeightedOption[T]]) -> T:
    total_weights = float(sum(x.weight for x in options))

    def get_decision_boundaries() -> (float, T):
        current = 0
        for weighted_option in options:
            current += weighted_option.weight / total_weights
            yield current, weighted_option.option

    decision_point = random()

    for boundary, option in get_decision_boundaries():
        if decision_point <= boundary:
            return option


def get_average_value(options: List[WeightedOption[int]]) -> float:
    total_weights = float(sum(x.weight for x in options))

    return reduce(lambda x,y: x + y.weight * float(y.option), options, 0.0) / total_weights
