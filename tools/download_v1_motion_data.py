import numpy as np
import argparse
import os

# Call the download script to pull a random selection of tfrecords, assuming they don't already
# exist in the cache directory

DEFAULT_CACHE_LOC = "../cache"
DEFAULT_VERSION = "1_2_1"

# TODO - with the introduction of the 1.2.1 dataset, there is now camera and lidar data available in a
# separate tfrecord.  However, the data is named using the scenario id, not the n of 1000 format
# used in the scenario data.  To get the ID, we need to extract it before we can pull the data.


def parse_arguments():
    parser = argparse.ArgumentParser(description="Download Waymo data")
    parser.add_argument(
        "--cache", type=str, default=DEFAULT_CACHE_LOC, help="Cache directory to store downloaded tfrecords"
    )
    parser.add_argument("--version", type=str, default=DEFAULT_VERSION, help="Waymo Open Dataset Version")
    parser.add_argument("--percentage", type=float, default=0.1, help="Percentage of tfrecords to download")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for selecting tfrecords")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # there are 1000 tfrecords, numbered 0-999 with format 'training_tfrecord_XXXXX-of-01000'

    cache_directory = args.cache + "/v1/motion/" + args.version + "/scenario"
    percentage_to_download = args.percentage
    random_seed = args.seed

    if not os.path.exists(cache_directory):
        print(f"Cache directory {cache_directory} does not exist. Exiting.")
        exit(1)

    generator = np.random.default_rng(seed=random_seed)

    # select the indices of the tfrecords to download
    num_training_records = 1000
    num_to_download = int(percentage_to_download * num_training_records)
    indices = generator.choice(num_training_records, num_to_download, replace=False)
    for index in indices:
        tfrecord = f"training.tfrecord-{index:05d}-of-01000"
        if not os.path.exists(os.path.join(cache_directory, tfrecord)):
            os.system(f"echo bash download_v1_motion_scenario.sh ${args.cache} ${args.version} training {tfrecord}")
        else:
            print(f"File {tfrecord} exists in cache.")

    # BUGBUG -- Currently skipping the test dataset as MTR is not using it in their training process

    num_validation_records = 150
    num_to_download = int(percentage_to_download * num_validation_records)

    # select the indices of the tfrecords to download
    indices = generator.choice(num_validation_records, num_to_download, replace=False)
    for index in indices:
        tfrecord = f"validation.tfrecord-{index:05d}-of-00150"
        if not os.path.exists(os.path.join(cache_directory, tfrecord)):
            os.system(f"bash download_v1_motion_scenario.sh ${args.cache} ${args.version} validation {tfrecord}")
        else:
            print(f"File {tfrecord} exists in cache.")
