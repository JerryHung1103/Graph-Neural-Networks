import warnings
warnings.filterwarnings("ignore")
from hyper_parameters import HYPERPARAMETERS
from train import run_one_training
run_one_training([HYPERPARAMETERS])