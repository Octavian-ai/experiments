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
    (other_review:REVIEW {is_golden:false})
        -[:OF {is_golden:false}]->
    (c)
WHERE other_person<>a

WITH
    a,b,c,
    COLLECT(others) as others

RETURN 
    a.style_preference, 
    b.score,
    others
    
LIMIT 10
""")

def get_dataset():
    data = []

    with SimpleNodeClient() as client:
        for item in client.execute_cypher(query, QueryParams()):
           
            others = []
            for path in item["others"]:

                other_person = path.nodes[0]
                other_review = path.nodes[1]

                others.append([other_person.properties['style_preference'], other_review.properties['score']])

            data.append(
                [item["b.score"], item["a.style_preference"], others]
            )

    return data

           
d = get_dataset()
print(d)


# Now just run network on that data!!
