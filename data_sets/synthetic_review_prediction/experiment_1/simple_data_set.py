from ..meta_classes import DataSetProperties
from ..classes import *
from ..utils import choose_weighted_option
from typing import Set, List, Callable
import random


class SimpleDataSet(object):
    def __init__(self, properties: DataSetProperties, opinion_function: Callable[[Person, Product], ReviewScore]):
        self.opinion_function = opinion_function
        self.properties = properties

        self._public_products: List[Product] = list()
        self._public_people_ids: Set[PersonID] = set()
        self._public_review_ids: Set[ReviewID] = set()

    def generate_public_products(self):
        for i in range(self.properties.n_products):
            style = choose_weighted_option(self.properties.product_styles_distribution)
            product_style = ProductStyle(style)
            product = Product(ProductID.new_random(), IsGoldenFlag(False), product_style)
            self._public_products.append(product)
            yield product

    def generate_public_people(self):
        for i in range(self.properties.n_people):
            number_of_reviews = choose_weighted_option(self.properties.reviews_per_person_distribution)
            meta = PersonMetaProperties(number_of_reviews=number_of_reviews)
            style_preference = PersonStylePreference(choose_weighted_option(self.properties.person_styles_distribution))
            person = Person(PersonID.new_random(), IsGoldenFlag(False), style_preference, meta_properties=meta)
            self._public_people_ids.add(person.id)
            yield person

    def generate_reviews(self, person: Person):
        for i in range(person.meta_properties.number_of_reviews):
            product = self.pick_public_product()
            score = self.opinion_function(person, product)
            review = Review(ReviewID.new_random(), IsGoldenFlag(False), score, person.id, product.id)
            self._public_review_ids.add(review.id)
            yield review

    def pick_public_product(self) -> Product:
        return random.choice(self._public_products)

    def pick_public_person(self) -> PersonID:
        return random.choice(self._public_people_ids)
