from uuid import UUID, uuid4
from .graph_node import NodeLabel, GraphNodeIdentifier, GraphNode, IsGoldenFlag
from .nano_type import NanoType, NanoID
from .style import Style


class ProductID(GraphNodeIdentifier):
    LABEL = NodeLabel('PRODUCT')

    def __init__(self, _id: UUID):
        super().__init__(self.LABEL, _id)


class ProductStyleEnum(object):
    def __init__(self, style: Style):
        self.style = style

    def __eq__(self, other):
        return self.style == other.style

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.style.value

    @classmethod
    def parse(cls, name: str):
        candidate = getattr(cls, name.upper())
        assert isinstance(candidate, cls)
        return candidate


class ProductStyle(NanoType[ProductStyleEnum]):
    pass


class Product(GraphNode[ProductID]):
    def __init__(self,
                 product_id: ProductID,
                 is_golden: IsGoldenFlag,
                 style: ProductStyle
                 ):
        super(Product, self).__init__(product_id, is_golden)
        self.style = style


