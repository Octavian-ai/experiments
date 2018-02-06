#!/bin/bash

floyd run  \
	--data davidmack/datasets/graph_experiments/1:/data \
	--env tensorflow-1.4 \
	--gpu \
	--tensorboard \
	--message "adj dense with dropout" \
	"ENVIRONMENT=floyd python train.py \
	 	--output-dir /output \
	 	--data-dir /data/ \
	 	--epochs 100"