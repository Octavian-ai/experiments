# Review prediction

## Introduction

The aim of this experiment is to investigate the performance of
1) different NN approaches 
2) different graph representations of the same data 

on a simple synthetic prediction task.

## The Task

We model personalised recommendations as a system containing _people_, _products_ and _recommendations_. In our system every product has a _style_ and each person has a _style preference_. _People_ can make _reviews_ of products. In our system the _review score_ will be a function _Y(...)_ of the person's _style preference_ and the product's _style_. We call this function the _opinion function_ i.e.:
 
  _review_score_ = _Y(product_style, person_style_preference)_ 

We will generate data using this model. We will then use this synthetic data to investigate how effective various ML approaches on the data set  are at learning the behaviour of this system.


If necessary we can change the opinion function _Y(...)_ to increase or decrease the difficulty of the task.

## The Synthetic Data

The synthetic data for this task can be varied in various ways:

1) Change which information is hidden e.g. we could hide _product_style_, _style_preference_ or both.
1) Change the representation of the key properties e.g. reviews/styles and preferences could be boolean, categorical, continuous scalars or even multi dimensional vectors.
1) Change how the data is represented as a graph e.g. reviews could be nodes in their own right, or they could be edges with properties, product_style could be a property on a product node or product_style could be a seperate node connected to a product node by a _HAS_STYLE_ relationship (edge).
1) Add additional meaningless or semi-meaningless information to the training data. 

We will generate different data sets to qualitatively investigate different ML approaches on the same basic system.


## Evaluation Tasks

We are interested in three different evaluation tasks depending on whether the person or product is included in the training set or not:

new product == unknown at training time i.e. not in training set or validation set
new person == unknown at training time i.e. not in training set or validation set
existing product == known at training time i.e. present in training set
existing person == known at training time i.e. present in training set

The evaluation tasks we are interested in are, how well can you predict the person's review? Given:

1) new product and new person 
1) existing product and new person
1) new product and existing person
1) existing product and existing person


## Approach

Although we have a synthetic system for which we can generate more data we want to get into good habits for working with "real" data. So we will attempt to blind the ML system to the fact that we are working with synthetic data and not rely on our ability to generate more information at will. 
So it will be the responsibility of the ML part of the system to split the data into Test / Train and Validation sets. However for each data set that we generate we will keep back a small portion to make up a "hidden" or "golden" test set which is only to be used at the very end of the investigation to provide a "double check" on the performance of different models.

Because of the three different evaluation tasks it will be necessary for us to keep back three different golden test sets:

Note: Since we don't know exactly how the training/validation/test split will be carried out we will generate excessive numbers for data sets used in 'existing data' cases

1) INDEPENDENT: A completely independent data set containing 1000 reviews
1) NEW_PEOPLE: new people + their reviews of existing products containing approx 2000 reviews
1) NEW_PRODUCTS: new products + reviews of them by existing people containing approx 2000 reviews
1) EXISTING: 2000 additional reviews between existing people and products.



# The Data Sets


## Data Set 1: The simplest situation that Andrew considers to be interesting 

- All variables will be 'public' in the data set


### Product Style
- _style_ will be categorical with two mutually exclusive elements (A and B).
- The distribution of product styles will be uniform i.e. Approx 50% of products will have style A and 50% will have style B.


### Style Preference
- _style_preference_ will be categorical with two mutually exclusive elements (likes_A_dislikes_B | likes_B_dislikes_A ).
- The distribution of product styles will be uniform i.e. Approx 50% of people will like style A and 50% will like style B.


### Reviews and Opinion Function
- _review_score_ will be boolean (1 for a positive review and 0 for a negative review)
- Each person will have made either 1 or 2 reviews. The mean number of reviews-per-person will be approx 1.5 i.e. approx 50% will have made 2 reviews and 50% will have made 1 review. 

Note: having people with 0 reviews would be useless since you cannot train or validate/test using them.
 
Note: fixing the number of reviews-per-person would restrict the graph structure too much and open up the problem to approaches that we aren't interested in right now.


### Entity Ratios and Data Set Size

I basically made these up. Intuitively the reviews-per-product and reviews-per-person parameters affect how much we can infer about people/product hidden variables. I like the idea of those figures being very different so we can see how systems cope with that distinction. 

- _people_:_products_ = 50:1
- _people_:_reviews_ = 1:1.5
- _reviews_:_products_ = 75:1

Data set size: 12000 reviews / 160 products / 8000 people 

n.b. because we assign the reviews randomly some products may not have reviews, but it is relatively unlikely.

### Graph Schema

PERSON(id: <uuid>, style_preference: A|B, is_golden: True|False) -- WROTE(is_golden: True|False) -> REVIEW(id: <uuid>, score: 1|0, is_golden: True|False) -- OF(is_golden: True|False) --> PRODUCT(id: <uuid>, style: A:B, is_golden: True|False)

### Data generation algorithm

1) Instantiate all products for public data set and write to Neo, keeping an array of the ids.
1) Iteratively instantiate people, decide how many reviews that person will have made (probabilistically)
1) For each review that the person has to make randomly choose a product to review (without replacement)
1) Calculate the review score and submit the Person + their reviews to Neo
1) Read the data back out of neo and validate the entity ratios
1) Create the golden test sets: 
  - NEW_PEOPLE: create 2000/reviews_per_person new people + their reviews of randomly selected (with replacement) existing products.
  - NEW_PRODUCTS: create 2000/reviews_per_product new products, have randomly selected (with replacement) people review them.
  - EXISTING randomly pick 2000 people (with replacement) have each of them review a randomly selected (with replacement) product
  - INDEPENDENT is easy, but best to leave till last to avoid confusion - just repeat the basic data generation from scratch

 
