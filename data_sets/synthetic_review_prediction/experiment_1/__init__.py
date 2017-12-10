from ..meta_classes import DataSetProperties
from ..meta_classes.data_set_properties import PersonStyleWeightDistribution
from ..utils import WeightedOption, Distribution
from ..classes import PersonStylePreferenceEnum, ProductStyleEnum

ProductStyleEnum.LIKES_A: ProductStyleEnum = ProductStyleEnum()
ProductStyleEnum.LIKES_B: ProductStyleEnum = ProductStyleEnum()

bar: PersonStylePreferenceEnum = PersonStylePreferenceEnum()
PersonStylePreferenceEnum.B: PersonStylePreferenceEnum = PersonStylePreferenceEnum()

Foo = WeightedOption[ProductStyleEnum]

data_set_properties = DataSetProperties(
    n_reviews=12000,
    reviews_per_product=75,
    reviews_per_person_distribution=[
        WeightedOption[int](1, 0.5),
        WeightedOption[int](2, 0.5)
    ],
    person_styles_distribution=PersonStyleWeightDistribution(
        WeightedOption[int](1, 0.5),
    ),
    product_styles_distribution=[
        Foo(1, 0.5),
        WeightedOption[ProductStyleEnum](bar, 0.5),
    ],
)