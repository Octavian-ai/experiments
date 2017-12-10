from uuid import UUID, uuid4
from .graph_edge import GraphEdge
from .nano_type import NanoType
from .person import Person
from .product import Product

class Review(GraphEdge[]):

    @classmethod
    def new_random(cls):
        return ProductID(uuid4())


class Product(GraphNode[ProductID]):
    def __init__(self,
                 product_id: ProductID,
                 is_golden: IsGoldenFlag,
                 style: ProductStyle
                 ):
        super(Product, self).__init__(product_id, is_golden)
        self.style = style


class ProductStyleEnum(object):
    def __init__(self):
        pass

    @classmethod
    def parse(cls, name: str):
        candidate = getattr(cls, name.upper())
        assert isinstance(candidate, cls)
        return candidate


ProductStyleEnum.LIKES_A: ProductStyleEnum()
ProductStyleEnum.LIKES_B: ProductStyleEnum()


class ProductStyle(NanoType[ProductStyleEnum]):
    pass