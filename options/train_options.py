from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def is_train(self):
        return True

    def __init__(self, save_conf=True):
        super().__init__(save_conf)

    def initialize(self):
        BaseOptions.initialize(self)

