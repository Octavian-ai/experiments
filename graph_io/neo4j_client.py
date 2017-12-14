from neo4j.v1 import GraphDatabase, Driver
from .classes import CypherQuery, QueryParams
from config import config
from lazy import lazy

class NodeClient(object):
    class_singleton = None

    def __init__(self, uri, user, password):
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self.instance = None

    @lazy
    def _session(self):
        return self._driver.session().__enter__()


    def execute_cypher(self, cypher: CypherQuery, query_params: QueryParams):
        # TODO: If you use this for writes then bad things can happen
        for x in self._session.run(cypher.value, **query_params.params):
            yield x

    def execute_cypher_write(self, cypher: CypherQuery, query_params: QueryParams):
        execute_cypher = lambda tx: tx.run(cypher.value, **query_params.params)

        session = self._session
        result = session.write_transaction(execute_cypher)
        return result

    @staticmethod
    def get_client():
        if NodeClient.class_singleton is None:
            NodeClient.class_singleton = NodeClient(config.neo4j_url, config.neo4j_user, config.neo4j_password)
        return NodeClient.class_singleton

    @staticmethod
    def close_client():
        NodeClient.class_singleton._driver.close()

    @property
    def _nuke_db_query(self):
        return CypherQuery("MATCH (n) DETACH DELETE n")

    def nuke_db(self):
        self.instance.execute_cypher(self._nuke_db_query, QueryParams())


class SimpleNodeClient(NodeClient):
    def __init__(self):
        self.instance: NodeClient = None
        # deliberately not calling super

    def __enter__(self) -> NodeClient:
        self.instance = NodeClient.get_client()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.instance._session.__exit__(None, None, None)
        NodeClient.close_client()
        self.instance = None
        return

    def __getattr__(self, item):
        return getattr(self.instance, item)

