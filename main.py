from genericpath import exists
import sys
import os
import argparse
import yaml
from code.Enhance import Enhance

if __name__ == "__main__":
    # We need to modifiy it to take into account yaml file (with other infos)

    parser = argparse.ArgumentParser(description='Enhance an image')
    parser.add_argument('--config', type=str, default='config.yaml')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    input_path = config['input_path']
    name_final = config['name_final']
    output_path = config['output_path']
    temp_path = config['temp_path']
    path_model = config['path_model']
    n_jobs = config['n_jobs']
    model_type = config['model_type'].lower()
    scale = config['scale']

    os.makedirs(temp_path, exist_ok=True)

    Enhance(input_path, name_final, output_path, temp_path, path_model, n_jobs=n_jobs, model_type=model_type, scale=scale)