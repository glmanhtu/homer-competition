from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def is_train(self):
        return True

    def __init__(self, save_conf=True):
        super().__init__(save_conf)

    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--optimizer', type=str, default='Adam')
        self._parser.add_argument('--resume', action='store_true', help="Resume training")
        self._parser.add_argument('--save_freq_iter', type=int, default=10,
                                  help='save the training losses to the summary writer every # iterations')
        self._parser.add_argument('--lr', type=float, default=4e-4, help="The initial learning rate")

        self._parser.add_argument('--lr_policy', type=str, default='none', choices=['step', 'none'])
        self._parser.add_argument('--lr_decay_epochs', type=int, default=100,
                                  help='reduce the lr to 0.1*lr for every # epochs')
        self._parser.add_argument('--n_epochs_per_eval', type=int, default=3,
                                  help='Run eval every n training epochs')
        self._parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
        self._parser.add_argument('--nepochs', type=int, default=1000)
        self._parser.add_argument('--early_stop', type=int, default=20)
