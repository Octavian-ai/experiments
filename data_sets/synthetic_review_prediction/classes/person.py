from uuid import UUID, uuid4
from .graph_node import GraphNodeIdentifier, GraphNode, NodeLabel, IsGoldenFlag
from .nano_type import NanoID, NanoType


class PersonID(GraphNodeIdentifier):
    LABEL = NodeLabel('PERSON')

    def __init__(self, _id: UUID):
        super().__init__(self.LABEL, _id)


class PersonStylePreferenceEnum(object):
    def __init__(self):
        pass


class PersonStylePreference(NanoType[PersonStylePreferenceEnum]):
    pass


class PersonMetaProperties(object):
    def __init__(self, number_of_reviews: int):
        self.number_of_reviews = number_of_reviews


class Person(GraphNode[PersonID]):
    def __init__(self,
                 person_id: PersonID,
                 is_golden: IsGoldenFlag,
                 style_preference: PersonStylePreference,
                 meta_properties: PersonMetaProperties=None
                 ):
        super(Person, self).__init__(person_id, is_golden)
        self.style_preference = style_preference
        self.meta_properties = meta_properties
