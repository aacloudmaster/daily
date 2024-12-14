import logging
import boto3
from botocore.exceptions import ClientError
import os

s3 = boto3.client('s3')
response = s3.list_buckets()

file_name = 'inputs/inputs.json'
bucket = '<<with_your_bucket_name>>'
object_name = 'myinputs.json'
local_file_name = 'inputs/localinputs.json'

def get_buckets_from_s3():
    # Output the bucket names
    print('Existing buckets:')
    for bucket in response['Buckets']:
        print(f'  {bucket["Name"]}')

def upload_file_from_s3(file_name, bucket, object_name=object_name):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def download_file_from_s3(bucket, object_name,local_file_name):

    """Download a file from an S3 bucket

    :param bucket: Bucket to upload to
    :param object_name: S3 object name to download. If not specified then file_name is used
    :param local_file_name: File to download
    :return: True if file was downloaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.download_file(bucket, object_name,local_file_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

get_buckets_from_s3()
upload_file_from_s3(file_name, bucket, object_name)
download_file_from_s3(bucket, object_name,local_file_name)


