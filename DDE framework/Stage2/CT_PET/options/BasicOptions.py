import argparse
import os
from utils import util
import torch


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        
        parser.add_argument('--batch_size', type=int, default=10, help='input train batch size')
        parser.add_argument('--epoch', type=int, default=15, help='number of epochs with the initial learning rate')
        parser.add_argument('--datapath', default = r'C:\Users\2418015\Downloads\final_data\128_training', help='path of the raw data')
        parser.add_argument('--testing_datapath', default=r'C:\Users\2418015\Downloads\final_data\128_testing',help='path of the raw data')
        parser.add_argument('--lr_sin', type=float, default=0.01, help='initial learning rate of net for adam')
        parser.add_argument('--lr_img', type=float, default=0.01, help='initial learning rate of net for adam')
        parser.add_argument('--step_sin', type=int, default=200, help='initial learning rate of net for adam')
        parser.add_argument('--step_img', type=int, default=200, help='initial learning rate of net for adam')

        parser.add_argument('--lamada_sin', type=float, default=0.5, help='initial learning rate of net for adam')
        parser.add_argument('--lamada_img', type=float, default=0.5, help='initial learning rate of net for adam')
        parser.add_argument('--step', type=float, default=200, help='adjust learning rate')

        parser.add_argument('--model_save_fre', type=int, default=50, help='frequency of saving model')
        #parser.add_argument('--test_fre', type=int, default=50, help='frequency of testing the model')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--task_name', type=str, default='CT-PET_Stage2_3', help='the current task name')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
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

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, 'model_parameter_list')
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{train_opt.txt}')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)

        self.opt = opt
        return self.opt

