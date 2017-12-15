
from collections import namedtuple

Experiment = namedtuple('Experiment', ['doc', 'cypher_query'])

directory = {
	"review_from_visible_style": Experiment(
			"""
				A simple baseline experiment.

				From a person's style preference and a product's style, predict review score.

				review_score = dot(style_preference, product_style)
			""",
			"""MATCH p=
					(a:PERSON {is_golden:{golden}}) 
						-[:WROTE {is_golden:{golden}}]-> 
					(b:REVIEW {is_golden:{golden}}) 
						-[:OF {is_golden:{golden}}]-> 
					(c:PRODUCT {is_golden:{golden}})
				RETURN a.style_preference AS style_preference, c.style AS style, b.score AS score
			"""
		),


	"review_from_hidden_style_neighbor_conv": Experiment(
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

		"""
			MATCH p=
				(a:PERSON {is_golden:{golden}}) 
					-[:WROTE {is_golden:{golden}}]-> 
				(b:REVIEW {is_golden:{golden}}) 
					-[:OF {is_golden:{golden}}]-> 
				(c:PRODUCT {is_golden:{golden}})


			WITH 
			    a,b,c
			MATCH others=
			    (other_person:PERSON)
			        -[:WROTE {is_golden:{golden}}]->
			    (other_review:REVIEW {is_golden:{golden}})
			        -[:OF {is_golden:{golden}}]->
			    (c)
			WHERE other_person<>a

			WITH
				a,
				b,
				COLLECT(others) as neighbors

			RETURN 
				a.style_preference AS style_preference, 
				b.score AS score, 
				neighbors

		"""
		),
	"style_from_neighbor_conv": Experiment(
		""" 
		A precursor to review_from_hidden_style_neighbor_conv

		This experiment seeks to see if we can efficiently determine a product's style
		given it's set of reviews and the style_preference of each reviewer.

		This should be easy!!

		""",
		"""
			MATCH p=
				(a:PERSON {is_golden:{golden}}) 
					-[:WROTE {is_golden:{golden}}]-> 
				(b:REVIEW {is_golden:{golden}}) 
					-[:OF {is_golden:{golden}}]-> 
				(product:PRODUCT {is_golden:{golden}})

			WITH
				product,
				COLLECT(p) as neighbors

			RETURN 
				product.style,
				neighbors

		"""

		)

}

default = "review_from_visible_style"


