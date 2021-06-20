# -*- coding: utf-8 -*-
# Reference: https://github.com/shunsukesaito/PIFu/blob/master/lib/options.py
# author: pinakinathc.me

import argparse
import os


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        g_data = parser.add_argument_group('Data')
        g_data.add_argument('--dataroot', type=str, default='./data', help='path to images (data folder)')
        g_data.add_argument('--load_size', type=int, default=256, help='load size of input image')

        # Experiment related
        g_exp = parser.add_argument_group('Experiment')
        g_exp.add_argument('--name', type=str, default='example',
            help='name of experiment. It decides where to store samples and models')
        g_exp.add_argument('--num_views', type=int, default=1, help='how many views to use for multiview')
        g_exp.add_argument('--random_multiview', action='store_true', help='select random multiview combination')
        g_exp.add_argument('--scale', type=float, default=0.7, help='scale 3D vertices')

        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
        g_train.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0 0,1,2, -1 for CPU mode')
        g_train.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        g_train.add_argument('--serial_batches', action='store_true',
            help='if true, takes images in order to make batches, otherwise takes them randomly')
        g_train.add_argument('--pin_memory', action='store_true', help='pin_memory')
        g_train.add_argument('--batch_size', type=int, default=1, help='input batch size')
        g_train.add_argument('--learning_rate', type=float, default=1e-3, help='adam learning rate')
        g_train.add_argument('--num_epoch', type=int, default=7000, help='num epoch to train')
        g_train.add_argument('--resume_epoch', type=int, default=-1, help='epoch resuming training')
        g_train.add_argument('--continue_train', action='store_true', help='continue training: load latest model')
        g_train.add_argument('--freq_plot', type=int, default=10, help='frequency of error plot')
        g_train.add_argument('--freq_save', type=int, default=100, help='frequency of save checkpoints')
        g_train.add_argument('--freq_eval', type=int, default=5000, help='frequency of evaluation')
        g_train.add_argument('--logdir', type=str, default='train_log', help='tensorboard training log')
        parser.add_argument('--num_samples', type=int, default=10000, help='number of sampled points')

        # Testing related
        g_test = parser.add_argument_group('Testing')

        # Sampling related
        g_sample = parser.add_argument_group('Sampling')
        g_sample.add_argument('--sigma', type=float, default=1.0, help='perturbation s.d. for positions')
        g_sample.add_argument('--num_sample_inout', type=int, default=5000, help='# of sampling points')

        # Model related
        g_model = parser.add_argument_group('Model')
        # General
        g_model.add_argument('--norm', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization')

        # hg filter specify
        g_model.add_argument('--num_stack', type=int, default=4, help='# of hourglass')
        g_model.add_argument('--num_hourglass', type=int, default=2, help='# of stacked layer of hourglass')
        g_model.add_argument('--skip_hourglass', action='store_true', help='skip connection in hourglass')
        g_model.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        g_model.add_argument('--hourglass_dim', type=int, default=12, help='256 | 512')

        # smplicit generator specify
        g_model.add_argument('--point_pos_size', type=int, default=3, help='input dim of sample points')
        g_model.add_argument('--n_z_cut', type=int, default=12, help='# of dimension used to represent 3D from 2D')


        # for train
        parser.add_argument('--random_trans', action='store_true', help='if random translation')
        parser.add_argument('--random_scale', action='store_true', help='if random scaling')
        parser.add_argument('--schedule', type=int, nargs='+', default=[1000],
            help='decrease learning rate at certain epochs')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule')

        # path
        parser.add_argument('--checkpoints_path', type=str, default='./checkpoints', help='path to save checkpoints')
        parser.add_argument('--results_path', type=str, default='./results', help='path to save results ply')
        parser.add_argument('--load_checkpoint_path', type=str, help='path to saved model to continue')

        # aug
        group_aug = parser.add_argument_group('aug')
        group_aug.add_argument('--aug_bri', type=float, default=0.0, help='augmentation brightness')
        group_aug.add_argument('--aug_con', type=float, default=0.0, help='augmentation contrast')
        group_aug.add_argument('--aug_sat', type=float, default=0.0, help='augmentation saturation')
        group_aug.add_argument('--aug_hue', type=float, default=0.0, help='augmentation hue')
        group_aug.add_argument('--aug_blur', type=float, default=0.0, help='augmentation blur')

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt
