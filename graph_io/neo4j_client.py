from neo4j.v1 import GraphDatabase, Driver
from .classes import CypherQuery, QueryParams
from config import config


class NodeClient(object):
    class_singleton = None

    def __init__(self, uri, user, password):
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def execute_cypher(self, cypher: CypherQuery, query_params: QueryParams):
        execute_cypher = lambda tx: tx.run(cypher.value, **query_params.params)

        with self._driver.session() as session:
            result = session.read_transaction(execute_cypher)
            return result

    @staticmethod
    def get_client():
        if NodeClient.class_singleton is None:
            NodeClient.class_singleton = NodeClient(config.neo4j_url, config.neo4j_user, config.neo4j_password)
        return NodeClient.class_singleton

    @staticmethod
    def close_client():
        NodeClient.class_singleton._driver.close()


class SimpleNodeClient(NodeClient):
    def __init__(self):
        self.instance = None
        # deliberately not calling super

    def __enter__(self) -> NodeClient:
        self.instance = NodeClient.get_client()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        NodeClient.close_client()
        self.instance = None
        return

    def __getattr__(self, item):
        return getattr(self.instance, item)