from typing import Generic, TypeVar
from uuid import uuid4, UUID

T = TypeVar('T')


class NanoType(Generic[T]):
    def __init__(self, value: [T]):
        self._value: T = value

    @property
    def value(self) -> T:
        return self._value

    def __hash__(self):
        return self._value.__hash__()

    def __eq__(self, other):
        return self._value.__eq__()

    def __ne__(self, other):
        return self._value.__ne__()

    def __gt__(self, other):
        return self._value.__gt__()

    def __lt__(self, other):
        return self._value.__lt__()


class NanoID(NanoType[UUID]):

    @classmethod
    def new_random(cls):
        return cls(uuid4())