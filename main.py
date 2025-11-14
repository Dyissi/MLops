# main.py
import argparse
import os
from model_pipeline import prepare_data, train_model, evaluate_model

parser = argparse.ArgumentParser(description="ML Pipeline")
parser.add_argument('--prepare', action='store_true', help="Prepare and save data")
parser.add_argument('--train', action='store_true', help="Train the model")
parser.add_argument('--evaluate', action='store_true', help="Evaluate the model")
parser.add_argument('--all', action='store_true', help="Run prepare → train → evaluate")
args = parser.parse_args()

if args.prepare or args.all:
    print("Preparing data...")
    prepare_data()

if args.train or args.all:
    import os
    required = ['X_train.pkl', 'y_train.pkl', 'X_test.pkl', 'y_test.pkl']
    if not all(os.path.exists(f) for f in required):
        print("Error: Run --prepare first!")
    else:
        print("Training model...")
        train_model()  # loads data from .pkl files

if args.evaluate or args.all:
    if not os.path.exists('xgboost_model.json'):
        print("Error: Run --train first!")
    else:
        print("Evaluating model...")
        evaluate_model()  # loads model + data

