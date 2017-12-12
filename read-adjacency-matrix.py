from graph_io import SimpleNodeClient, CypherQuery, QueryParams
import numpy as np
from uuid import UUID

query = CypherQuery(
"""
MATCH p=
    (a:PERSON {is_golden:false})
        -[:WROTE {is_golden:false}]->
    (b:REVIEW {is_golden:false})
        -[:OF {is_golden:false}]->
    (c:PRODUCT {is_golden:false})

WITH 
    a,b,c
MATCH others=
    (other_person:PERSON)
        -[:WROTE {is_golden:false}]->
    (r:REVIEW {is_golden:false})
        -[:OF {is_golden:false}]->
    (c)
WHERE other_person<>a

WITH 
    a,b,c,
    COLLECT(others) as all_others

WITH 
    a,b,c,
    REDUCE(output = [], r IN all_others | output + relationships(r)) AS flat_relationships,
    REDUCE(output = [], r IN all_others | output + nodes(r)) AS flat_nodes
RETURN 
    a.style_preference, 
    c.style, 
    b.score,
    [r in  flat_relationships | [startNode(r).id, endNode(r).id]] as edges,  
    [n in  flat_nodes | properties(n)] as nodes
LIMIT 100000
""")

def get_adjacency_dataset():
    with SimpleNodeClient() as client:
        for item in client.execute_cypher(query, QueryParams()):
            print(item)

            nodes = item['nodes']
            index_lookup = {UUID(n['id']): i for i, n in enumerate(nodes)}
            N = len(nodes)
            adjacency_matrix = np.zeros((N,N))

            for e in item['edges']:
                from_id = UUID(e[0])
                to_id = UUID(e[1])
                from_idx = index_lookup[from_id]
                to_idx = index_lookup[to_id]
                adjacency_matrix[from_idx, to_idx] = 1
                adjacency_matrix[to_idx, from_idx] = 1

                # the dict['score'] is the ground truth you are trying to predict for the given 'style_preference', the nodes & adjacency matrix contain all
                # necessary information for this node to do that. N.b. the dimensions of adjacency matrix vary! But you can pad with zeros and also trim excess
                # for now if you want.
                yield dict(style_preference=item['a.style_preference'], score=item['b.score']), nodes, adjacency_matrix

for x in get_adjacency_dataset():
    print(x)