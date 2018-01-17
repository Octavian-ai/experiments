from data_sets.synthetic_review_prediction import *

from basic_types import NanoType

from .experiment_header import ExperimentHeader

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
		""",
		float
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

		""",
		float
	),

	"review_from_all_hidden_simple_unroll": ExperimentHeader(
		"""
			# Objective

			Learn a function `score(input_person, input_product)` that gives a product review
			given a person and a product.

			## Input format

			People, reviews and products are essentially anonymous and defined by their relationship
			to each-other.

			Our network needs to take in a portion of the graph then output the predicted score.

			The graph is transformed and formatted in a consistent fashion, allowing the network
			to understand which person and product is being input.

			# Solution
	
			Allow the network to find look-a-likes by generating array of person-product-person-product-person chains

			E.g. If me and my lookalike both liked product X, then we'll agree for product Y

			This has a limitation that it can only successfully predict a score of there happens to be someone
			with the same style_preference who has reviewed a product you have also reviewed.

		""",
		EXPERIMENT_4_DATASET,
		"""
			MATCH g=(input_person:PERSON) 
					-[:WROTE]-> 
				(target_review:REVIEW {dataset_name:{dataset_name}}) 
					-[:OF]-> 
				(input_product:PRODUCT)
					<-[:OF]-
				(review1:REVIEW)
					<-[:WROTE]-
				(person2:PERSON)
					-[:WROTE]->
				(review2:REVIEW)
					-[:OF]->
				(product2:PRODUCT)
					<-[:OF]-
				(review3:REVIEW)
					<-[:WROTE]-
				(input_person)
			
			WHERE 
				input_person<>person2 
				AND input_product<>product2 

			RETURN
				target_review.score as score,
				COLLECT([1.0, review1.score, review2.score, review3.score])[0..50] as neighbors,

				// These two need to be here otherwise the query implicitly groups by score
				input_product.id,
				input_person.id

		""",
		float,
		{
			"neighbor_count":50
		}
	),

	"review_from_all_hidden_random_walks": ExperimentHeader(
		"""
			Let's try to do a RNN that operates on pieces of the graph
			Generate random walks.

			This is a great problem because it requires the network to find a specific
			shape of subgraph in order to answer the question.

			It needs to find a loop, with 1s on the review scores, like such:

			(REVIEW=1)  --> (PRODUCT) <--  (REVIEW=1) <-- (PERSON_B)
				↑											  |
				|											  ↓
			(PERSON_A) --> (THE_REVIEW) --> (PRODUCT) <-- (REVIEW=1)


			# Idea

			What if the parameters define a shape the network wants to look for?

			That's the solution to this problem and could be useful for other problems,
			particularly since the magic of neural networks lets you define a noise-resiliant
			function, and an ensemble of shapes.

			Let:

			const string_length = 9
			pattern:List[part]  = |----|-----|-----|----|  ==>  Convolve 1D with path
			path:List[part]     = (a)-->(b)-->(c)-->(d)
			part                = (type, parameter_values, is_target) | Loop | None
			target_type         = "REVIEW"

			## Algorithm

			1) For each node of type=target_type:
			1.a) Generate all paths s.t. |path| <= string_length
			1.b) If a path is cyclic it should have a 'Loop' element after the nodes
			2) Feed to network ([path, ..., path], target_review_score)
			3) Network performs 1D convolution of each path with pattern kernel (The overflow of the kernel should wrap around the input path)
			4) Network performs a 1D convolution on those outputs
			5) Network sums those values
			6) Network applies a dense layer, thus outputting y_prediction


		""",
		EXPERIMENT_4_DATASET,
		"""
			MATCH p=
				(review:REVIEW {is_golden:{golden}, dataset_name:{dataset_name}}) 
					-[*8]-
				(otherB)
			WHERE review.id={id}
			WITH
				review,
				COLLECT(p)[0..600] as neighbors
			RETURN 
				review,
				neighbors
		""",
		float,
		{
			"generate_address": False,
			"target_dropout": 0.0,
			"memory_size": 1000,
			"word_size": 4,
			"sequence_size": 600,
			"patch_width": 7,
			"patch_size": 20,
			"epochs": 20,
			"repeat_batch": 1,
			"working_width": 64,
			"id_limit": 32 * 10
		}, 
		["id_limit"]
	),

	"review_from_all_hidden_adj": ExperimentHeader(
		"""
			Try the following:
				- variable pr represents PRODUCT style vectors
				- variable pe represents PERSON preference vectors
				- x = adj matrix of PRODUCT-REVIEW-PERSON
				- y = adj matrix of same with REVIEW.score as the weights
				- Use optimizer to optimize the style/pref vectors such that: Dot(MatMul(pr, T(pe)), x) = y


		""",
		EXPERIMENT_5_DATASET,
		"""
			MATCH p=
				(person:PERSON) -->
				(review:REVIEW {is_golden:{golden}, dataset_name:{dataset_name}}) -->
				(product:PRODUCT)
			RETURN 
				person.id as person_id, review.score as score, product.id as product_id
		""",
		"adj_equals",
		{
			"product_count": 160, # total 160
			"person_count": 1200, # total 1200
			"style_width": 20,
			"epochs": 20,
			"batch_per_epoch": 10
		}
	),

	"style_from_neighbor_conv": ExperimentHeader(
		""" 
		A precursor to review_from_hidden_style_neighbor_conv

		This experiment seeks to see if we can efficiently determine a product's style
		given it's set of reviews and the style_preference of each reviewer.

		This should be easy!!

		""",
		EXPERIMENT_2_DATASET,
		shared_query["product_and_product_subgraph"],
		list,
	),

	"style_from_neighbor_rnn": ExperimentHeader(
		""" The same as style_from_neighbor_conv but using an RNN instead of convolution """,
		EXPERIMENT_2_DATASET,
		shared_query["product_and_product_subgraph"],
		list
	)

}

default_experiment = "review_from_all_hidden_adj"


