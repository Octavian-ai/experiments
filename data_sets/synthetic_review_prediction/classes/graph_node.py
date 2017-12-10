from typing import Generic, TypeVar
from .nano_type import NanoType, NanoID

from uuid import UUID

T = TypeVar('T')


class IsGoldenFlag(object):
    def __init__(self, value: bool):
        self._value = value

    @property
    def value(self):
        return self._value


class NodeLabel(NanoType[str]):
    pass


class GraphNodeIdentifier(NanoID):
    def __init__(self, label: NodeLabel, _id: UUID):
        super().__init__(_id)
        self.label = label


T_gni = TypeVar('T_gni', GraphNodeIdentifier)


class GraphNode(Generic[T]):
    def __init__(self, _id: T_gni, is_golden: IsGoldenFlag):
        self._id = _id
        self.is_golden = is_golden

    @property
    def id(self) -> T:
        return self._id.id

    @property
    def label(self) -> NodeLabel:
        return self._id.label
