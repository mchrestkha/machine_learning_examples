echo "Submitting an AI Platform job..."


PROJECT_ID="mchrestkha-demo-env"
BUCKET_ID="mchrestkha-demo-env-ml-examples"
JOB_NAME=catboost_census_training_$(date +"%Y%m%d_%H%M%S")
JOB_DIR=gs://$BUCKET_ID/census/catboost_job_dir
TRAINING_PACKAGE_PATH="../trainer/"
MAIN_TRAINER_MODULE=trainer.train
REGION=us-west1
RUNTIME_VERSION=2.1
PYTHON_VERSION=3.7
SCALE_TIER=BASIC


gcloud ai-platform jobs submit training $JOB_NAME \
--job-dir $JOB_DIR \
--package-path $TRAINING_PACKAGE_PATH \
--module-name $MAIN_TRAINER_MODULE \
--region $REGION \
--runtime-version=$RUNTIME_VERSION \
--python-version=$PYTHON_VERSION \
--scale-tier $SCALE_TIER

