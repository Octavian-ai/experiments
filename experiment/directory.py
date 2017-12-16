from data_sets.synthetic_review_prediction import EXPERIMENT_1_DATASET, EXPERIMENT_2_DATASET, EXPERIMENT_3_DATASET
from graph_io.classes import DatasetName


class ExperimentHeader(object):
	def __init__(self, doc, dataset_name: DatasetName, cypher_query, meta={}):
		# Jesus I have to spell this out?!
		# WTF are the python language devs doing?!
		self.dataset_name = dataset_name
		self.doc = doc
		self.cypher_query = cypher_query
		self.meta = meta

shared_query = {
	"product_and_product_subgraph": """
			MATCH p=
				(a:PERSON {is_golden:{golden}, dataset_name:{dataset_name}}) 
					-[:WROTE {is_golden:{golden}, dataset_name:{dataset_name}}]-> 
				(b:REVIEW {is_golden:{golden}, dataset_name:{dataset_name}}) 
					-[:OF {is_golden:{golden}, dataset_name:{dataset_name}}]-> 
				(product:PRODUCT {is_golden:{golden}, dataset_name:{dataset_name}})

			WITH
				product,
				COLLECT(p) as neighbors

			RETURN 
				product,
				neighbors

	"""

}

directory = {
	"review_from_visible_style": ExperimentHeader(
			"""
				A simple baseline experiment.

				From a person's style preference and a product's style, predict review score.

				review_score = dot(style_preference, product_style)
			""",
			EXPERIMENT_2_DATASET,
			"""MATCH p=
					(a:PERSON {is_golden:{golden}, dataset_name:{dataset_name}}) 
						-[:WROTE {is_golden:{golden}, dataset_name:{dataset_name}}]-> 
					(b:REVIEW {is_golden:{golden}, dataset_name:{dataset_name}}) 
						-[:OF {is_golden:{golden}, dataset_name:{dataset_name}}]-> 
					(c:PRODUCT {is_golden:{golden}, dataset_name:{dataset_name}})
				RETURN a.style_preference AS style_preference, c.style AS style, b.score AS score
			"""
		),


	"review_from_hidden_style_neighbor_conv": ExperimentHeader(
		"""
			A simple experiment requiring the ML system to aggregate information from a sub-graph

			Predict a person's score for a product, given a person's style preference and the product

			This needs to be able to take in the review graph for a product
			and infer the product's style based on the style_preference and scores other people gave the product.

			Plan for the network (assume 1 hot encoding for categorical variables):

			For a product (product):
				For a person (person):

					- get array of N other people's reviews: [other_person.style_preference, score] x N
					- Apply 1d_convolution output: [product_style] x N
					- Apply average across N, output: [product_style]
					- Apply softmax, output: [product_style]
					- Concat with person, output: [product_style, person.style_preference]
					- Apply dense layer, activation sigmoid, output: [score]

					- Train that!

		""",
		EXPERIMENT_2_DATASET,
		"""
			MATCH (a:PERSON) 
					-[e1:WROTE ]-> 
				(b:REVIEW {is_golden:{golden}, dataset_name:{dataset_name}}) 
					-[e2:OF ]-> 
				(c:PRODUCT),
			others=
			    (other_person:PERSON {is_golden:{golden}, dataset_name:{dataset_name}})
			        -[:WROTE {is_golden:{golden}, dataset_name:{dataset_name}}]->
			    (other_review:REVIEW {is_golden:{golden}, dataset_name:{dataset_name}})
			        -[:OF {is_golden:{golden}, dataset_name:{dataset_name}}]->
			    (c)
			WHERE other_person<>a AND other_review<>b 
			WITH
				a,b,c,
                e1,e2,
				COLLECT(others) as neighbors
			WHERE a.dataset_name={dataset_name} AND a.is_golden={golden}
            AND b.dataset_name={dataset_name} AND b.is_golden={golden}
			AND c.dataset_name={dataset_name} AND c.is_golden={golden}
			AND e1.dataset_name={dataset_name} AND e1.is_golden={golden}
			AND e2.dataset_name={dataset_name} AND e2.is_golden={golden}
            RETURN 
				a.style_preference AS style_preference,
				b.score AS score,
				neighbors

		"""
		),

	"style_from_neighbor_conv": ExperimentHeader(
		""" 
		A precursor to review_from_hidden_style_neighbor_conv

		This experiment seeks to see if we can efficiently determine a product's style
		given it's set of reviews and the style_preference of each reviewer.

		This should be easy!!

		""",
		EXPERIMENT_2_DATASET,
		shared_query["product_and_product_subgraph"]
		),

	"style_from_neighbor_rnn": ExperimentHeader(
		""" The same as style_from_neighbor_conv but using an RNN instead of convolution """,
		EXPERIMENT_2_DATASET,
		shared_query["product_and_product_subgraph"]
	)

}

default_experiment = "style_from_neighbor_rnn"


