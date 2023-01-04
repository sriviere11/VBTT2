#import bamboolib as bam

# Input Output functions to save and read files

#import bamboolib as bam
import json
import configparser
import joblib
from google.api_core.exceptions import NotFound
from google.cloud import storage, logging
import pandas as pd
import io





import os
import os.path


def instantiate_logging():
    # Instantiates a client
    logging_client = logging.Client()
    # The name of the log to write to
    log_name = "VBTT3"
    # Selects the log to write to
    logger = logging_client.logger(log_name)
    return logger




def write_list(a_list, filename_json):
    logger=instantiate_logging()
    logger.log_text(f"Function write_list|Started writing list data into a json file {filename_json}")
    with open(filename_json, "w") as fp:
        json.dump(a_list, fp)
        logger.log_text(f"Function write_list|Done writing list data into a json file {filename_json}")
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
    filename_json=config['DEFAULT']['filename_json']

    # ^TNX reasury yield is the annual return investors can expect from holding a U.S. government security with a given
    # ^GSPC tracks the performance of the stocks of 500 large-cap companies in the US"
    # CL=F crude oil pricesi.get_data(result[0][0])

    return version, additional_data,regressor, model, root_bucket,filename_json




def create_bucket(bucket_name):


    storage_client = storage.Client()
    logger=instantiate_logging()
    if bucket_name not in [x.name for x in storage_client.list_buckets()]:
        bucket = storage_client.create_bucket(bucket_name)
        logger.log_text("Function create_bucket|Bucket {} created".format(bucket.name))
    else:
        logger.log_text(f"Function create_bucket|Bucket {bucket_name} already exists")



def get_bucket_name():
    root_bucket = read_config_file()[4]
    version = read_config_file()[0]
    bucket_name=f'{root_bucket}_{version.replace(".", "")}'
    create_bucket(bucket_name)
    return bucket_name


def upload_file_to_bucket(model_file_name):
    logger=instantiate_logging()
    bucket_name=get_bucket_name()
    storage_client = storage.Client()
    bucket=storage_client.bucket(bucket_name)
    blob = bucket.blob(model_file_name)
    blob.upload_from_filename(model_file_name)
    logger.log_text(f"Function upload_file_to_bucket|file {model_file_name} uploaded to {bucket_name}.")


def delete_blob(blob_name):
    logger = instantiate_logging()
    bucket_name = get_bucket_name()
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()
    logger.log_text(f"Function delete_blob|old blob {blob_name} deleted from {bucket_name}\n")


def delete_then_get_model_from_bucket(model_filename):
    logger=instantiate_logging()

    if (os.path.exists(model_filename)):
        os.remove(model_filename)
        logger.log_text(f"Function delete_then_get_model_from_bucket|old file {model_filename} deleted locally\n")

    bucket_name = get_bucket_name()
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob=bucket.blob(model_filename)

    try:
        blob.download_to_filename(model_filename)
        logger.log_text(f"Function delete_then_get_model_from_bucket|new file {model_filename} downloaded\n")
        return True #file download ok therefore retrain not required  or fileexists locally

    except NotFound as e:
        logger.log_text(f"Function delete_then_get_model_from_bucket|file {model_filename} not found in bucket\n")
        os.remove(model_filename)
        return False  #retrain required or file does not exist locally


def save_dataframe_to_bucket(dataframe, filename):
    #filename include the extension
    logger=instantiate_logging()
    bucket_name = get_bucket_name()
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    bucket.blob(f'Dataframes/{filename}').upload_from_string(dataframe.to_csv(), 'text/csv')
    logger.log_text(f"Function save_dataframe_to_bucket| Dataframe {filename} saved successfuly in folder {bucket}/Dataframes.")


def read_dataframe_from_bucket(filename):
    #filename include the extension - REMEMBER to put extension when calling function
    logger=instantiate_logging()
    bucket_name = get_bucket_name()
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob=bucket.blob(f'Dataframes/{filename}')
    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data))
    df=df.drop(columns=["Unnamed: 0"])
    logger.log_text(f"Function read_dataframe_from_bucket| Dataframe {filename} read successfuly from folder {bucket}/Dataframes.")
    return df


