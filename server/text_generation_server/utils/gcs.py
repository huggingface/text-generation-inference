import os
from google.cloud import storage

GCS_PREFIX = "gs://"   
GCS_LOCAL_DIR = "/tmp/gcs_model"

def download_gcs_dir_to_local(gcs_dir: str, local_dir: str):
    if os.path.isdir(local_dir):
        return
    # gs://bucket_name/dir
    bucket_name = gcs_dir.split('/')[2]
    prefix = gcs_dir[len(GCS_PREFIX + bucket_name) :].strip('/') + '/'
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    if not blobs:
        raise ValueError(f"No blobs found in {gcs_dir}")
    for blob in blobs:
        if blob.name[-1] == '/':
            continue
        file_path = blob.name[len(prefix) :].strip('/')
        local_file_path = os.path.join(local_dir, file_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        logger.info(f"==> Download {gcs_dir}/{file_path} to {local_file_path}.")
        blob.download_to_filename(local_file_path)
    logger.info("Download finished.")
