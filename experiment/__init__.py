
from collections import namedtuple

Experiment = namedtuple('Experiment', ['doc', 'cyper_query'])

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
				RETURN a.style_preference AS preference, c.style AS style, b.score AS score
			"""
		),


	"review_from_hidden_style": Experiment(
		"""
			A simple experiment requiring the ML system to aggregate information from a sub-graph

			Predict a person's score for a product, given a person's style preference and the product

			This needs to be able to take in the review graph for a product
			and infer the product's style based on the style_preference and scores other people gave the product.
		""",

		"""MATCH p=
				(a:PERSON {is_golden:{golden}}) 
					-[:WROTE {is_golden:{golden}}]-> 
				(b:REVIEW {is_golden:{golden}}) 
					-[:OF {is_golden:{golden}}]-> 
				(c:PRODUCT {is_golden:{golden}})

			RETURN a.style_preference AS preference, b.score AS score
		"""
	)
}

default = "review_from_visible_style"


