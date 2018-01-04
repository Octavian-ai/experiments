import graph_io
import time
from data_sets.synthetic_review_prediction import experiment_4

client = graph_io.SimpleNodeClient.get_client()

t0 = time.time()
ids = list(client.get_node_ids(experiment_4.DATASET_NAME, 200, "REVIEW"))
print(f"Number of ids: {len(ids)}")
print(f"Time to list ids: {round(time.time()-t0)}")

product_and_product_subgraph = """
		MATCH p=
			(review:REVIEW {is_golden:{golden}, dataset_name:{dataset_name}}) 
				-[*5]-
			(other)
		WHERE review.id={id}
		WITH
			review,
			COLLECT(p)[0..6] as neighbors
		RETURN 
			review,
			neighbors
"""

for i, d in enumerate(client.execute_cypher_once_per_id(
		graph_io.CypherQuery(product_and_product_subgraph),
		graph_io.QueryParams(golden=False, dataset_name=experiment_4.DATASET_NAME),
		dataset_name=experiment_4.DATASET_NAME,
		id_limit=200,
		id_type="REVIEW"
)):
	print(i, d)
	if i%100 == 0:
		print(float(i)/(time.time()-t0))

print('done')
print(f"Total time: {time.time()-t0}")

client.close_client()