import argparse
import os


class BaseOptions:
    def __init__(self, save_conf):
        self._parser = argparse.ArgumentParser()
        self._initialized = False
        self._opt = None
        self._save_conf = save_conf

    def is_train(self):
        raise NotImplementedError()

    def initialize(self):
        self._parser.add_argument('--dataset', type=str, help='Path to the dataset')
        self._parser.add_argument('--batch_size', type=int, default=2, help='Input batch size')
        self._parser.add_argument('--image_size', type=int, default=1200, help='Input image size')
        self._parser.add_argument('--p2_image_size', type=int, default=800, help='Input image size')
        self._parser.add_argument('--ref_box_height', type=int, default=32, help='Reference letter box height')
        self._parser.add_argument('--cuda', action='store_true', help="Whether to use GPU")
        self._parser.add_argument('--p1_arch', type=str, default='mobinet')
        self._parser.add_argument('--p2_arch', type=str, default='resnet50')
        self._parser.add_argument('--n_threads_train', default=8, type=int, help='# threads for loading data')
        self._parser.add_argument('--n_threads_test', default=8, type=int, help='# threads for loading data')
        self._parser.add_argument('--group', type=str, default='experiment',
                                  help='name of the experiment. It decides where to store samples and models')
        self._parser.add_argument('--name', type=str, default='experiment_1',
                                  help='name of the experiment. It decides where to store samples and models')
        self._parser.add_argument('--mode', type=str, default='first_twin',
                                  help='The training mode', choices=['first_twin', 'second_twin', 'testing'])
        self._parser.add_argument('--wb_mode', type=str, default='offline', help='Wandb sync mode')
        self._parser.add_argument('--wb_entity', type=str, default='glmanhtu', help='Wandb entity name')
        self._parser.add_argument('--wb_project', type=str, default='homer-competition', help='Wandb project')
        self._parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self._parser.add_argument('--dropout', type=float, default=0.5, help="Default dropout")
        self._parser.add_argument('--merge_iou_threshold', type=float, default=0.3, help="Threshold for merging boxes")

        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        # set is train or test
        self._opt.is_train = self.is_train()

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        if self._save_conf:
            self._save(args)

        return self._opt

    @staticmethod
    def _print(args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        print(expr_dir)
        if self.is_train and not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        else:
            assert os.path.exists(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
