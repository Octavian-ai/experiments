
from data_sets.synthetic_review_prediction.experiment_1 import run

if __name__ == '__main__':
    from graph_io import SimpleNodeClient
    with SimpleNodeClient() as client:
        run(client)
