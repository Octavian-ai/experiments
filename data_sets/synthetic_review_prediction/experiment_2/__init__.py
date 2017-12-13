from .configure import create_data_set_properties
from ..experiment_1.generate import run as _run


def run(client):
    return _run(client, create_data_set_properties())
