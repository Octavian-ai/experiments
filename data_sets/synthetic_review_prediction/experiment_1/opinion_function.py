from ..classes import Person, Product, ReviewScore


def opinion_function(person: Person, product: Product) -> ReviewScore:
    if person.style_preference.value.style == product.style.value.style:
        return ReviewScore(1.0)
    else:
        return ReviewScore(0)

