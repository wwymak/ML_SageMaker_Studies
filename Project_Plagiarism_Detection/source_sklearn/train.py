from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    parser.add_argument('--n-estimators', type=int, default=10)
    parser.add_argument('--min-samples-leaf', type=int, default=1)
    parser.add_argument('--min-samples-split', type=int, default=2)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--max-leaf-nodes', type=int, default=-1)

    # args holds all passed-in arguments
    args = parser.parse_args()
    print('here!!!')
    print(args.max_leaf_nodes)
    print('here!!!')

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)


    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]

    ## TODO: Define a model
    model = RandomForestClassifier(n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
                                   min_samples_split=args.min_samples_split,max_depth=args.max_depth,
                                   max_leaf_nodes=args.max_leaf_nodes, n_jobs=-1)
    cv_results = cross_validate(RandomForestClassifier(), train_x, train_y, cv=stratified_k_fold,
                                scoring=('recall', 'f1', 'roc_auc'), return_train_score=True)
    print(cv_results)
    
    model.fit(train_x, train_y)
    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))