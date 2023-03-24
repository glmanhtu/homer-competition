from options.train_options import TrainOptions


class CrossValOptions(TrainOptions):
    def is_train(self):
        return True

    def __init__(self, save_conf=True):
        super().__init__(save_conf)

    def initialize(self):
        TrainOptions.initialize(self)
        self._parser.add_argument('--k_fold', type=int, default=5)
