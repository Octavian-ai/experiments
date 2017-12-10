import math
from typing import List
from ..utils import WeightedOption, choose_weighted_option, get_average_value
from ..classes import PersonStylePreferenceEnum, ProductStyleEnum

class DataSetProperties(object):
    def __init__(self,
                 n_reviews: int,
                 reviews_per_product: float,
                 reviews_per_person_distribution: List[WeightedOption[int]],
                 person_styles_distribution: List[WeightedOption[PersonStylePreferenceEnum]],
                 product_styles_distribution: List[WeightedOption[ProductStyleEnum]]
                 ):
        self.product_styles_distribution = product_styles_distribution
        self.person_styles_distribution = person_styles_distribution
        self.reviews_per_person_distribution = reviews_per_person_distribution
        self.reviews_per_product = reviews_per_product
        self.n_reviews = n_reviews

    @property
    def reviews_per_person(self) -> float:
        return get_average_value(self.reviews_per_person_distribution)

    @property
    def n_people(self) -> int:
        return int(math.ceil(float(self.n_reviews) / self.reviews_per_person))

    @property
    def n_products(self) -> int:
        return int(math.ceil(float(self.n_reviews) / self.reviews_per_product))
