# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description='Multiview Garment Modeling')

parser.add_argument('--exp_name', type=str,
    default='experiment', help='name of experiment')

# ---
# Dataset Options
# ---

parser.add_argument('--data_dir', type=str,
	default='/vol/research/sketchcaption/adobe/multiview_modeling/data/body_up/',
	help='enter path to training data')
parser.add_argument('--num_points', type=int, default=2048,
	help='enter number of sample points to train at each iteration')
parser.add_argument('--num_views', type=int, default=3,
	help='enter number of views to train the model')

# ---
# Train Options
# ---

parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--num_workers', type=int, default=7, help='dataloader # of threads used')
parser.add_argument('--device', type=int, nargs='+', default=[0], help='multi-gpu')
parser.add_argument('--output_dir', type=str, default='output/experiment', help='dir to save/load')

# ---
# Logging Options
# ---

parser.add_argument('--print_freq', type=int, default=60, help='iterations after prints status')
parser.add_argument('--save_freq', type=int, default=500, help='iterations after save model')

opts = parser.parse_args()
