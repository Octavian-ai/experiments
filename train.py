
from graph_ml import Arguments, Train

if __name__ == '__main__':

	params = Arguments.parse()
	score = Train.run(params)

	print('Test loss:', score[0])
	print('Test accuracy:', score[1])