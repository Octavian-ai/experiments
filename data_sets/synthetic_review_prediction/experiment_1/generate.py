from . import data_set_properties
from ..classes import GraphNode, IsGoldenFlag
from typing import Set, AnyStr

from .simple_data_set import SimpleDataSet
from .opinion_function import opinion_function
from graph_io import SimpleNodeClient, QueryParams, CypherQuery


def run(client):

    data_set = SimpleDataSet(data_set_properties, opinion_function)

    def create_node_if_not_exists(node: GraphNode, properties: Set[AnyStr]):
        label: str = node.label.value
        _id: str = str(node.id.value)
        properties.add('id')
        properties.add('is_golden')
        properties_dict = {name: str(getattr(node, name).value) for name in properties}
        print(properties_dict)
        properties_string = "{" + ", ".join(f"{name}: ${name}" for name in properties) + "}"
        create_query = CypherQuery(f"MERGE (n:{label} {properties_string})")
        query_params = QueryParams(**properties_dict)
        result = client.execute_cypher_write(create_query, query_params)
        print("merged node", result)

    #TODO: refactor this!! Have an explicit relationship type
    def create_relationship_if_not_exists(relationship_type, is_golden: IsGoldenFlag, _from=None, _to=None):
        match = f"MATCH (from:{_from.label.value} {{ id: $from_id }}),(to:{_to.label.value} {{ id: $to_id }}) "
        merge = f"MERGE (from)-[r:{relationship_type}{{ is_golden: $is_golden }}]->(to)"

        create_query = CypherQuery(match + merge)
        query_params = QueryParams(is_golden=is_golden.value, from_id=str(_from.id.value), to_id=str(_to.id.value))

        result = client.execute_cypher_write(create_query, query_params)
        print("merged edge", result)


    for i, product in enumerate( data_set.generate_public_products()):
        create_node_if_not_exists(product, {"style"})
        if i> 10:
            break

    for i, person in enumerate(data_set.generate_public_people()):
        create_node_if_not_exists(person, {"style_preference"})

        for review in data_set.generate_reviews(person):
            create_node_if_not_exists(review, {"score"})
            create_relationship_if_not_exists("BY_PERSON", is_golden=IsGoldenFlag(False), _from=review.by_person, _to=review.id)
            create_relationship_if_not_exists("OF_PRODUCT", is_golden=IsGoldenFlag(False), _from=review.id, _to=review.of_product)


        if i> 10:
            break

