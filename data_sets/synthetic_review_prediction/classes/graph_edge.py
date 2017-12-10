from typing import Generic, TypeVar
from .nano_type import NanoType
from .graph_node import GraphNode

T_from = TypeVar('T_from', GraphNode)
T_to = TypeVar('T_to', GraphNode)


class IsGoldenFlag(object):
    def __init__(self, value: bool):
        self._value = value

    @property
    def value(self):
        return self._value


class EdgeType(NanoType[str]):
    pass


class GraphEdge(Generic[T_from, T_to]):
    def __init__(self, _from: T_from, to: T_to, is_golden: IsGoldenFlag):
        assert _from is not None
        assert to is not None

        self.is_golden = is_golden
        self._from = _from
        self._to = to
