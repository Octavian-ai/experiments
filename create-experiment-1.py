
from graph_io import SimpleNodeClient
from data_sets.synthetic_review_prediction.experiment_1 import run

with SimpleNodeClient() as client:
    run(client)
