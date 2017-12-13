from typing import Generic, TypeVar, Iterable

T = TypeVar("T")


class DynamicEnum(Generic[T]):
    LOCKED = False

    def __init__(self, defn: T):
        self.defn = defn

    def __neo__(self):
        return self.as_one_hot()

    def __eq__(self, other):
        return self.defn == other.defn

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.defn)

    @classmethod
    def parse(cls, name: str):
        candidate = getattr(cls, name.upper())
        assert isinstance(candidate, cls)
        return candidate

    @classmethod
    def register(cls, name: str, style: T):
        if cls.LOCKED:
            raise Exception('cannot add to locked enum')
        setattr(cls, name.upper(), cls(style))

    @classmethod
    def iterate(cls) -> Iterable[T]:
        cls.LOCKED = True
        for k, v in sorted(vars(cls).items()):
            if isinstance(v, cls):
                yield v

    # TODO: memoize this
    def as_one_hot(self):
        cls = self.__class__

        idx = None
        count = 0
        for i, v in enumerate(cls.iterate()):
            if v == self:
                if idx is not None:
                    raise Exception('Same value defined multiple times in enum')
                idx = i
            count = i + 1

        assert count > 0

        return [1 if i == idx else 0 for i in range(count)]