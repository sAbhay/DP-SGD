from sys import path
path.append('/juice/scr/sabhay/DP-SGD/fine_tune')

from src.training.image_classification import experiment

if __name__ == "__main__":
    experiment.run_experiment()