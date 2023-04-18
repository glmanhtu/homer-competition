from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def is_train(self):
        return True

    def __init__(self):
        super().__init__(save_conf=False)

    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--pretrained_model_path', default = '', type=str,  help='the model to be evaluated')
        self._parser.add_argument('--prediction_name', required=True, type=str,  help='Path to save the predictions')
        self._parser.add_argument('--k_fold', type=int, default=5)

