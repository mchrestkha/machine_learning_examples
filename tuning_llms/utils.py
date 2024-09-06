import gzip
import json
from google.cloud import storage
from datasets import load_dataset
import random
import time
import vertexai
from vertexai.preview.tuning import sft
from vertexai.evaluation import EvalTask, MetricPromptTemplateExamples
import json
import utils
import mercury as mr
import openai
import os
import pandas as pd
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig, Part
from google.cloud.exceptions import NotFound

def format_tuning_dataset(train_list, valid_list, base_instruction, train_filename, valid_filename):

    # Initialize lists to store messages for training and validation
    train_messages = []
    validation_messages = []

    # Iterate over training data and create messages for each dialogue-summary pair
    for d in train_list:
      prompts=[]
      prompts.append({"role": "user", "parts": [{"text": base_instruction + d["dialogue"]}]})
      prompts.append({"role": "model", "parts": [{"text": d["summary"]}]}) 
      train_messages.append({'contents': prompts})

    # Iterate over validation data and create messages similarly
    for d in valid_list:
      prompts=[]
      prompts.append({"role": "user", "parts": [{"text": base_instruction + d["dialogue"]}]})
      prompts.append({"role": "model", "parts": [{"text": d["summary"]}]}) 
      validation_messages.append({'contents': prompts})
    
    # Save to JSON locally
    dicts_to_jsonl(train_messages, train_filename, False)
    dicts_to_jsonl(validation_messages, valid_filename, False)

    # Print lengths of message lists and an example training message
    len(train_messages), len(validation_messages), train_messages[3]

# Delete & Overwrite files to upload to GCS
def delete_and_upload(filename):
    try:
        delete_blob("mchrestkha-sample-data",f"dialogsum/{filename}")
    except NotFound:
        pass
    upload_blob("mchrestkha-sample-data",filename,f"dialogsum/{filename}")    

    
#Submit Tuning Job
def tune_gemini(train_file, valid_file, model, model_name):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name=model_name+timestr
    vertexai.init(project="mchrestkha-sandbox", location="us-central1")

    sft_tuning_job = sft.train(
        source_model=model,
        train_dataset=train_file,
        # The following parameters are optional
        validation_dataset=valid_file,
        epochs=5,
        adapter_size=4,
        learning_rate_multiplier=1.0,
        tuned_model_display_name=model_name,
    ) 
    

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )
    
def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    generation_match_precondition = None

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to delete is aborted if the object's
    # generation number does not match your precondition.
    blob.reload()  # Fetch blob metadata to use in generation_match_precondition.
    generation_match_precondition = blob.generation

    blob.delete(if_generation_match=generation_match_precondition)

    print(f"Blob {blob_name} deleted.")


def dicts_to_jsonl(data_list: list, filename: str, compress: bool = True) -> None:
    """
    Method saves list of dicts into jsonl file.

    :param data: (list) list of dicts to be stored,
    :param filename: (str) path to the output file. If suffix .jsonl is not given then methods appends
        .jsonl suffix into the file.
    :param compress: (bool) should file be compressed into a gzip archive?
    """

    sjsonl = '.jsonl'
    sgz = '.gz'

    # Check filename

    if not filename.endswith(sjsonl):
        filename = filename + sjsonl

    # Save data
    
    if compress:
        filename = filename + sgz
        with gzip.open(filename, 'w') as compressed:
            for ddict in data_list:
                jout = json.dumps(ddict) + '\n'
                jout = jout.encode('utf-8')
                compressed.write(jout)
    else:
        with open(filename, 'w') as out:
            for ddict in data_list:
                jout = json.dumps(ddict) + '\n'
                out.write(jout)