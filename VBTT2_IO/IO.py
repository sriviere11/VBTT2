#import bamboolib as bam

# Input Output functions to save and read files

#import bamboolib as bam
import json
import configparser
import logging
import joblib
from google.api_core.exceptions import NotFound
from google.cloud import storage
import os
import os.path






def write_list(a_list, filename_json):
    print("Started writing list data into a json file")
    with open(filename_json, "w") as fp:
        json.dump(a_list, fp)
        print("Done writing data into .json file")
    upload_file_to_bucket(filename_json)


# Read list to memory
def read_list(filename_json):
    # for reading also binary mode is important
    with open(filename_json, 'rb') as fp:
        n_list = json.load(fp)
        return n_list




def read_config_file():
    # return element that is in configfile. exemple additional data
    config = configparser.ConfigParser()
    config.read('config.ini')
    version = config['DEFAULT']['version']
    additional_data = config['DEFAULT']['additional_data'].split(',')
    regressor = config['DEFAULT']['regressor']
    model = config['DEFAULT']['model']
    root_bucket=config['DEFAULT']['gcloud_root_bucket']

    # ^TNX reasury yield is the annual return investors can expect from holding a U.S. government security with a given
    # ^GSPC tracks the performance of the stocks of 500 large-cap companies in the US"
    # CL=F crude oil pricesi.get_data(result[0][0])

    return version, additional_data,regressor, model, root_bucket




#file existimport logging


def create_bucket(bucket_name):
    log = logging.getLogger()

    storage_client = storage.Client()
    if bucket_name not in [x.name for x in storage_client.list_buckets()]:
        bucket = storage_client.create_bucket(bucket_name)

        log.info("Bucket {} created".format(bucket.name))
    else:
        log.info("Bucket {} already exists".format(bucket_name))



def get_bucket_name():
    root_bucket = read_config_file()[4]
    version = read_config_file()[0]
    bucket_name=f'{root_bucket}_{version.replace(".", "")}'
    create_bucket(bucket_name)
    return bucket_name


def upload_file_to_bucket(model_file_name):
    bucket_name=get_bucket_name()
    log = logging.getLogger()
    log.warning(f'uploading {model_file_name} to {bucket_name}')
    storage_client = storage.Client()
    bucket=storage_client.bucket(bucket_name)
    blob = bucket.blob(model_file_name)
    blob.upload_from_filename(model_file_name)


def delete_blob(blob_name):
    log = logging.getLogger()
    bucket_name = get_bucket_name()
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()
    log.warning(f'old blob {blob_name} deleted from {bucket_name}\n')


def delete_then_get_model_from_bucket(model_filename):
    log = logging.getLogger()
    if (os.path.exists(model_filename)):
        os.remove(model_filename)
        log.warning(f'old file {model_filename} deleted locally\n')

    bucket_name = get_bucket_name()
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob=bucket.blob(model_filename)

    try:
        blob.download_to_filename(model_filename)
        print ('not right')
        log.warning(f'new file   {model_filename} downloaded\n')
        return True #file download ok therefore retrain not required  or fileexists locally

    except NotFound as e:
        log.warning(f'file {model_filename} not found in bucket\n')
        return False  #retrain required or file does not exist locally