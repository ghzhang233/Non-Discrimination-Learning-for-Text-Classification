import argparse


class Config:
    def __init__(self):
        self.name_model = "default_model_name"
        self.use_weights = False
        self.round = 10
        self.not_use_cudnn = False
        self.use_supplementation = False

    def set_params_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--name_model', type=str, help='name of the model', default="default_model_name")
        parser.add_argument('--use_weights', action="store_true", help='whether use weights', default=False)
        parser.add_argument('--round', type=int, help='number of run', default=10)
        parser.add_argument('--not_use_cudnn', action="store_true", help='not use cudnn', default=False)
        parser.add_argument('--use_supplementation', action="store_true", help='whether use supple', default=False)
        args = parser.parse_args()
        self.name_model = args.name_model
        self.use_weights = args.use_weights
        self.round = args.round
        self.not_use_cudnn = args.not_use_cudnn
        self.use_supplementation = args.use_supplementation
