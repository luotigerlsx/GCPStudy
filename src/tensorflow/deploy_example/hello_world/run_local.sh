#!/usr/bin/env bash

MODEL_DIR=./tmp/
TRAIN_DATA=~/Downloads/data/adult.data.csv
EVAL_DATA=~/Downloads/data/adult.test.csv

gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir $MODEL_DIR \
    --verbosity debug \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
    --train-steps 1000 \
    --eval-steps 100

