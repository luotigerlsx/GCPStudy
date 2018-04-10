#!/usr/bin/env bash

JOB_NAME=hello_world_cloud_ml

TRAIN_DATA=gs://testl-bucket/data/adult.data.csv
EVAL_DATA=gs://testl-bucket/data/adult.test.csv

OUTPUT_PATH=gs://testl-bucket/hello_world_cloud_ml

REGION=us-central1

gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--runtime-version 1.4 \
--module-name trainer.task \
--package-path trainer/ \
--region $REGION \
--config ./config.yaml \
-- \
--train-files $TRAIN_DATA \
--eval-files $EVAL_DATA \
--train-steps 1000 \
--verbosity DEBUG  \
--eval-steps 100



# or use standard scale-tier

#gcloud ml-engine jobs submit training $JOB_NAME \
#--job-dir $OUTPUT_PATH \
#--runtime-version 1.4 \
#--module-name trainer.task \
#--package-path trainer/ \
#--region $REGION \
#--scale-tier `BASIC_GPU` \
#-- \
#--train-files $TRAIN_DATA \
#--eval-files $EVAL_DATA \
#--train-steps 1000 \
#--verbosity DEBUG  \
#--eval-steps 100