import argparse

def get_parsed_args():
    parser = argparse.ArgumentParser(description="ML Training Pipeline")

    parser.add_argument('--project_name', type=str, help='Project name')
    parser.add_argument('--env', type=str, help='Environment (dev/hpc)')
    parser.add_argument('--data_dir', type=str, help='Path to data directory')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')

    return parser.parse_args()
