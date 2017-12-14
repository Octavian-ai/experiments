
import experiment
import hashlib
import os.path

def generate_output_path(params, suffix):
	query = experiment.directory[params.experiment].cypher_query
	m = hashlib.md5()
	m.update(query.encode('utf-8'))
	return os.path.join(params.output_dir + '/' + params.experiment + '_' + m.hexdigest()  + suffix)
