from uuid import UUID


class QueryParams(object):
    def __init__(self, **kwargs):
        self._params = kwargs

    @property
    def query_string(self):
        return "{" + ", ".join(f"{name}: ${name}" for name in self._params) + "}"

    def union(self, other):
        return QueryParams(**self._params, **other._params)

    def __extract_neo_value(self, element):
        assert element is not None

        if hasattr(element, '__neo__'):
            return element.__neo__()
        if hasattr(element, 'value'):
            return self.__extract_neo_value(element.value)
        if isinstance(element, UUID):
            return str(element)
        return element

    @property
    def cypher_query_parameters(self):
        return {n: self.__extract_neo_value(p) for n,p in self._params.items()} if self._params else {}
