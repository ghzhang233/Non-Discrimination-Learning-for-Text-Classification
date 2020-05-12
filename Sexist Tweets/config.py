import argparse


class Config:
    def __init__(self):
        self.name_model = "default_model_name"
        self.name_dataset = "st"
        self.use_weights = False
        self.round = 10
        self.use_augmentation = False

    def set_params_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--name_model', type=str, help='name of the model', default="default_model_name")
        parser.add_argument('--name_dataset', type=str, help='name of the dataset', default="st")
        parser.add_argument('--use_weights', action="store_true", help='whether use weights', default=False)
        parser.add_argument('--round', type=int, help='number of run', default=10)
        parser.add_argument('--use_augmentation', action="store_true", help='whether use augmentation', default=False)
        args = parser.parse_args()
        self.name_model = args.name_model
        self.name_dataset = args.name_dataset
        self.use_weights = args.use_weights
        self.round = args.round
        self.use_augmentation = args.use_augmentation
