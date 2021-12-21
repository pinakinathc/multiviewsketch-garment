#!/bin/bash

source /vol/research/sketchcaption/miniconda/bin/activate garment

cd /vol/research/sketchcaption/adobe/multiview_garment

# python train.py --data_dir=/vol/research/sketchcaption/adobe/training_data/adobe_shirt_rendered/ --exp_name=adobe_data_new_updater
python train.py --data_dir=/vol/research/sketchcaption/adobe/training_data/siggraph15_body_up/ --exp_name=siggraph15_data_new_updater
# python train.py --data_dir=/vol/research/sketchcaption/adobe/training_data/adobe_shirt_rendered/ --exp_name=adobe_data_legacy
# python train.py --data_dir=/vol/research/sketchcaption/adobe/training_data/siggraph15_body_up/ --exp_name=siggraph15_data_legacy