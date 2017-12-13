from ..meta_classes import DataSetProperties
from ..meta_classes.data_set_properties import PersonStyleWeightDistribution, PersonStyleWeight, ProductStyleWeight
from ..utils import WeightedOption, Distribution
from ..classes import PersonStylePreferenceEnum, ProductStyleEnum, Style


def create_data_set_properties() -> DataSetProperties:
    STYLE_A = Style("A")
    STYLE_B = Style("B")

    ProductStyleEnum.LIKES_A: ProductStyleEnum = ProductStyleEnum(STYLE_A)
    ProductStyleEnum.LIKES_B: ProductStyleEnum = ProductStyleEnum(STYLE_B)

    PersonStylePreferenceEnum.A: PersonStylePreferenceEnum = PersonStylePreferenceEnum(STYLE_A)
    PersonStylePreferenceEnum.B: PersonStylePreferenceEnum = PersonStylePreferenceEnum(STYLE_B)


    data_set_properties = DataSetProperties(
        n_reviews=12000,
        reviews_per_product=75,
        reviews_per_person_distribution=[
            WeightedOption[int](1, 0.5),
            WeightedOption[int](2, 0.5)
        ],
        person_styles_distribution=PersonStyleWeightDistribution([
            PersonStyleWeight(PersonStylePreferenceEnum.A, 1),
            PersonStyleWeight(PersonStylePreferenceEnum.B, 1)
        ]),
        product_styles_distribution=Distribution[ProductStyleWeight, ProductStyleEnum]([
            ProductStyleWeight(ProductStyleEnum.LIKES_A, 1),
            ProductStyleWeight(ProductStyleEnum.LIKES_B, 1)
        ])
    )

    return data_set_properties
