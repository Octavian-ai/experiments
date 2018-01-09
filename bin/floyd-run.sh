#!/bin/bash

floyd run  \
	--data davidmack/datasets/review_from_all_hidden_random_walks/1:/data \
	--env tensorflow-1.4 \
	--gpu \
	--tensorboard \
	--message "See if NTM will train at all!" \
	"ENVIRONMENT=floyd python train.py \
	 	--output-dir /output \
	 	--data-dir /data/ \
	 	--epochs 50"