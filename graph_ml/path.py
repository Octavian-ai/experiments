
import experiment
import hashlib
import os.path

def generate_output_path(experiment, suffix):
	query = experiment.header.cypher_query
	m = hashlib.md5()
	m.update(query.encode('utf-8'))
	return os.path.join(experiment.params.output_dir + '/' + experiment.name + '_' + m.hexdigest()  + suffix)
