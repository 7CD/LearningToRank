import argparse
import yaml

from src.data.dataset import download_dataset, process_dataset


def data_load(config_path):
    config = yaml.safe_load(open(config_path))
    
    url = config['data_load']['url']
    raw_dataset_path = config['data_load']['raw_dataset_path']
    processed_dataset_dir = config['data_load']['processed_dataset_dir']
    
    download_dataset(url, raw_dataset_path)
    
    process_dataset(raw_dataset_path, processed_dataset_dir)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)

