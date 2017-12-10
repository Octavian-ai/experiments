from uuid import UUID, uuid4
from .graph_node import NodeLabel, GraphNodeIdentifier, GraphNode, IsGoldenFlag
from .nano_type import NanoType, NanoID


class ProductID(GraphNodeIdentifier):
    LABEL = NodeLabel('PRODUCT')

    def __init__(self, _id: UUID):
        super().__init__(self.LABEL, _id)


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



class ProductStyle(NanoType[ProductStyleEnum]):
    pass