import os
from VBTT2_SP500.SP500 import YF_datetime, get_SP500
from VBTT2_IO.IO import  instantiate_logging,read_config_file

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/steve/Downloads/creds.json "
filename_json=read_config_file()[5]
logger=instantiate_logging()
logger.log_text(f"Module app_SP500.py| Module run just started in order to create filename {filename_json}")
get_SP500(filename_json)
logger.log_text(f"Module app_SP500.py| Module finished running and created filename {filename_json}")
