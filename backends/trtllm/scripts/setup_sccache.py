from argparse import ArgumentParser

AWS_S3_CACHING_VARIABLES = {
    "AWS_ACCESS_KEY_ID": "aws_access_key_id",
    "AWS_SECRET_ACCESS_KEY": "aws_secret_access_key",
    "AWS_SESSION_TOKEN": "aws_session_token",
    "SCCACHE_REGION": "s3_region",
    "SCCACHE_BUCKET": "s3_bucket_name",
}

ALL_CACHING_STORAGE_VARIABLES = {"AWS_S3_CACHING_VARIABLES"}


def setup_sccache_locally():
    from os import environ

    print("Setting up Local Caching Layer")
    for target in ALL_CACHING_STORAGE_VARIABLES:
        for envvar in globals()[target].keys():
            if envvar in environ:
                print(f"Deleted {envvar} from environment variables")
                del environ[envvar]


def setup_sccache_for_s3():
    from os import environ

    print("Setting up AWS S3 Caching Layer")
    for envvar in AWS_S3_CACHING_VARIABLES.keys():
        if envvar not in environ or not environ[envvar] or len(environ[envvar]) == 0:
            print(f"Missing definition for environment variable {envvar}")


if __name__ == "__main__":
    parser = ArgumentParser("TensorRT-LLM Build Caching Setup")

    parser.add_argument(
        "--is-gha-build",
        type=str,
        default="FALSE",
        help="Indicate if the build is from Github Actions",
    )

    # Parse args
    args = parser.parse_args()
    args.is_gha_build = args.is_gha_build.lower() in {"on", "true", "1"}

    if args.is_gha_build:
        setup_sccache_for_s3()
    else:
        setup_sccache_locally()
