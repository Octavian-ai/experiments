from typing import Generic, TypeVar
from .nano_type import NanoType, NanoID
from .golden_flag import IsGoldenFlag

from uuid import UUID

T = TypeVar('T')


class NodeLabel(NanoType[str]):
    pass


class GraphNodeIdentifier(NanoID):
    def __init__(self, label: NodeLabel, _id: UUID):
        super().__init__(_id)
        self.label = label

T_gni = TypeVar('T_gni',  bound=GraphNodeIdentifier, covariant=True)


class GraphNode(Generic[T_gni]):
    def __init__(self, _id: T_gni, is_golden: IsGoldenFlag):
        self._id: T_gni = _id
        self.is_golden = is_golden

    @property
    def id(self) -> T_gni:
        return self._id

    @property
    def label(self) -> NodeLabel:
        return self._id.label
