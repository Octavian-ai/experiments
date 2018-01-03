
import hashlib
import os.path

def generate_path(experiment, prefix, suffix):
	query = experiment.header.cypher_query
	m = hashlib.md5()
	m.update(query.encode('utf-8'))
	# m.update(str(experiment.header.params).encode('utf-8'))
	return os.path.join(prefix + '/' + experiment.name + '_' + m.hexdigest()  + suffix)

def generate_output_path(experiment, suffix):
	return generate_path(experiment, experiment.params.output_dir, suffix)

def generate_data_path(experiment, suffix):
	return generate_path(experiment, experiment.params.data_dir, suffix)
