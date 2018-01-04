import graph_io
import time
from data_sets.synthetic_review_prediction import experiment_4

client = graph_io.SimpleNodeClient.get_client()

t0 = time.time()
ids = list(client.get_node_ids(experiment_4.DATASET_NAME))
print(len(ids))
print(f"{time.time()-t0}")

product_and_product_subgraph = """
		MATCH p=
			(a:PERSON {is_golden:{golden}, dataset_name:{dataset_name}}) 
				-[:WROTE {is_golden:{golden}, dataset_name:{dataset_name}}]-> 
			(review:REVIEW {is_golden:{golden}, dataset_name:{dataset_name}}) 
				-[:OF {is_golden:{golden}, dataset_name:{dataset_name}}]-> 
			(product:PRODUCT {is_golden:{golden}, dataset_name:{dataset_name}})
        WHERE review.id={id}
		WITH
			product,
			COLLECT(p) as neighbors

		RETURN 
			product,
			neighbors

"""

for i, _ in enumerate(client.execute_cypher_once_per_id(
        graph_io.CypherQuery(product_and_product_subgraph),
        graph_io.QueryParams(golden=False, dataset_name=experiment_4.DATASET_NAME),
        dataset_name=experiment_4.DATASET_NAME
)):
    if i%100 == 0:
        print(float(i)/(time.time()-t0))

print('done')
print(f"{time.time()-t0}")

client.close_client()