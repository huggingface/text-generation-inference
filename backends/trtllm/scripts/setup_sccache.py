from argparse import ArgumentParser

AWS_S3_CACHING_VARIABLES = {
    "AWS_ACCESS_KEY_ID": "aws_access_key_id",
    "AWS_SECRET_ACCESS_KEY": "aws_secret_access_key",
    "AWS_SESSION_TOKEN": "aws_session_token",
    "SCCACHE_REGION": "s3_region",
    "SCCACHE_BUCKET": "s3_bucket_name",
}

ALL_CACHING_STORAGE_VARIABLES = {
    "AWS_S3_CACHING_VARIABLES"
}


def setup_sccache_locally():
    from os import environ

    print("Setting up Local Caching Layer")
    for target in ALL_CACHING_STORAGE_VARIABLES:
        for envvar in globals()[target].keys():
            if envvar in environ:
                print(f"Deleted {envvar} from environment variables")
                del environ[envvar]


def setup_sccache_for_s3(s3_args):
    from os import environ

    print("Setting up AWS S3 Caching Layer")
    for envvar, field in AWS_S3_CACHING_VARIABLES.items():
        environ[envvar] = getattr(s3_args, field)


if __name__ == "__main__":
    parser = ArgumentParser("TensorRT-LLM Build Caching Setup")

    parser.add_argument("--is-gha-build", type=str, default="FALSE",
                        help="Indicate if the build is from Github Actions")
    parser.add_argument("--aws-access-key-id", "-k", type=str, required=True, help="AWS Access Key ID to use")
    parser.add_argument("--aws-secret-access-key", "-s", type=str, required=True,
                        help="AWS Secret Access Key to use")
    parser.add_argument("--aws-session-token", "-t", type=str, required=True, help="AWS Session Token to use")
    parser.add_argument("--s3-bucket-name", "-b", type=str, required=True, help="AWS target S3 Bucket")
    parser.add_argument("--s3-bucket-prefix", "-p", type=str, required=True, help="AWS target S3 Bucket folder prefix")
    parser.add_argument("--s3-region", "-r", type=str, required=True, help="AWS target S3 region")

    # Parse args
    args = parser.parse_args()
    args.is_gha_build = args.is_gha_build.lower() in {"on", "true", "1"}
    print(args)

    match args.store:
        case "s3":
            setup_sccache_for_s3(args)
        case _:
            setup_sccache_locally()
