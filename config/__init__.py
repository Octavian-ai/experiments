from collections import defaultdict
from .environment import Environment

default_values = {
    'neo4j_url': 'bolt://010e4dc7-staging.databases.neo4j.io',
    'neo4j_user': 'readonly',
    'neo4j_password': 'neo4j_movies_db!'
}

overrides = defaultdict(dict)
overrides.update(**{
    'andrew': {},
    'david': {
        'neo4j_url': 'bolt://796bafef-staging.databases.neo4j.io',
        'neo4j_user': 'readonly',
        'neo4j_password': '0s3DGA6Zq'
    },
    'floydhub': {

    }
})

environment_box = Environment(None)


def set_environment(environment_name):
    environment_box.name = environment_name


def get(config_variable_name):
    return overrides[environment_box.name].get(config_variable_name, default_values[config_variable_name])


class Config(object):
    @property
    def neo4j_url(self):
        return get('neo4j_url')

    @property
    def neo4j_user(self):
        return get('neo4j_user')

    @property
    def neo4j_password(self):
        return get('neo4j_password')


config: Config = Config()

import os
set_environment(os.environ['ENVIRONMENT'])