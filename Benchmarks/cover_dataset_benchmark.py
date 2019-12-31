# coding=utf-8
"""Benchmark accelerated gradient boosting."""

import os
import sys
from sklearn import datasets, model_selection, metrics, tree, svm, linear_model, neighbors

TEST_DIR = os.path.dirname(__file__)
SRC_DIR = '../Source'
sys.path.insert(0, os.path.abspath(os.path.join(TEST_DIR, SRC_DIR)))

import AcceleratedGradientBoosting


if __name__ == '__main__':
    for dataset, n_classes in zip([datasets.fetch_covtype], [7]):
        print(dataset)
        print("Loading dataset...")
        x_data, y_data = dataset(return_X_y=True)

        print("Spliting data")
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data)

        print("Performing base-learner comparison...")
        for base_learner, base_learner_params in zip([tree.DecisionTreeRegressor,
                                                      svm.LinearSVR,
                                                      linear_model.LinearRegression,
                                                      neighbors.KNeighborsRegressor],
                                                     [{'max_leaf_nodes': 5},
                                                      {},
                                                      {},
                                                      {}]):
            print(base_learner, base_learner_params)
            agb = AcceleratedGradientBoosting.AcceleratedGradientBoosting(iterations=10,
                                                                          n_classes=n_classes,
                                                                          base_learner=base_learner)
            agb.fit(x_train, y_train, verbose=True)

            print("Testing:")
            print(metrics.accuracy_score(y_test, agb.predict(x_test)))
            print("Training:")
            print(metrics.accuracy_score(y_train, agb.predict(x_train)))
