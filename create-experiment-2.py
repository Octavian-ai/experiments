from data_sets.synthetic_review_prediction.experiment_2 import run
from graph_io import SimpleNodeClient


with SimpleNodeClient() as client:
    run(client)
