import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import helperFunctions as hf

# CONSTANTS
curr = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(curr, "../data/")
VERBOSE = False

# list of models to investigate
models = [
    LogisticRegression(solver='liblinear'),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    BernoulliNB(),
    GaussianNB(),
    MLPClassifier()
]

if (__name__=="__main__"):
    # change these filenames as necessary
    mgw_file_pos = DATA_FOLDER + "positive_UAK42_47304_MGW.txt"
    roll_file_pos = DATA_FOLDER + "positive_UAK42_47304_Roll.txt"
    proT_file_pos = DATA_FOLDER + "positive_UAK42_47304_ProT.txt"
    helT_file_pos = DATA_FOLDER + "positive_UAK42_47304_HelT.txt"

    mgw_file_neg = DATA_FOLDER + "negative_UAK42_47304_MGW.txt"
    roll_file_neg = DATA_FOLDER + "negative_UAK42_47304_Roll.txt"
    proT_file_neg = DATA_FOLDER + "negative_UAK42_47304_ProT.txt"
    helT_file_neg = DATA_FOLDER + "negative_UAK42_47304_HelT.txt"

    # load DNA physical properties and create a features matrix
    all_pos = hf.getFeatsAveraged(mgw_file_pos, roll_file_pos, proT_file_pos, helT_file_pos)
    all_neg = hf.getFeatsAveraged(mgw_file_neg, roll_file_neg, proT_file_neg, helT_file_neg)
    X = pd.concat([all_pos, all_neg], axis=0)

    # create targets matrix
    targets_pos = np.ones((all_pos.shape[0],))
    targets_neg = np.zeros((all_neg.shape[0],))
    y = np.concatenate((targets_pos, targets_neg), axis=0)

    if VERBOSE:
        print("Features:")
        print(f"Pos samples: {all_pos.shape}")
        print(f"Neg samples: {all_neg.shape}")
        print(f"    Final X: {X.shape}\n")

        print("Targets:")
        print(f"Pos targets: {targets_pos.shape}")
        print(f"Neg targets: {targets_neg.shape}")
        print(f"    Final y: {y.shape}")

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X.values, 
                                                        y, 
                                                        train_size=0.8, 
                                                        test_size=0.2)
    if VERBOSE:
        print(f"X_train.shape: {X_train.shape}")
        print(f"X_test.shape:  {X_test.shape}")
        print(f"y_train.shape: {y_train.shape}")
        print(f"y_test.shape:  {y_test.shape}")

    # loop through models and evaluate them
    for model in models:
        # fit model
        model.fit(X_train, y_train)

        # make a prediction
        y_pred = model.predict(X_test)

        # evaluate
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        if VERBOSE:
            print(f"Accuracy : {acc*100:.3f}%")
            print(f"F1 Score : {f1*100:.3f}%")
            print(f"Precision: {pre*100:.3f}%")
            print(f"Recall   : {recall*100:.3f}%\n")

            hf.plot_confusion_matrix(y_test, y_pred)

