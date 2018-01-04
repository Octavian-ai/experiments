
import hashlib
import os.path
import logging

logger = logging.getLogger(__name__)

def generate_path(experiment, prefix, suffix, extra=""):
	query = experiment.header.cypher_query
	m = hashlib.md5()

	m.update(query.encode('utf-8'))
	m.update(extra.encode('utf-8'))
	# logger.info(f"generate_path {prefix} {suffix} {query} {extra}")
	# m.update(str(experiment.header.params).encode('utf-8'))
	return os.path.join(prefix + '/' + experiment.name + '_' + m.hexdigest()  + suffix)

def generate_output_path(experiment, suffix):
	return generate_path(experiment, experiment.params.output_dir, suffix)

def generate_data_path(experiment, suffix, query_params=None):
	return generate_path(experiment, experiment.params.data_dir, suffix, str(query_params))
