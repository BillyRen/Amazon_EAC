from __future__ import division

import numpy as np
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVC

from DataIO import load_data, save_results

SEED = 42

def main():
    """
    Fit models and make predictions.
    We'll use one-hot encoding to transform our categorical features
    into binary features.
    y and X will be numpy array objects.
    """
    # The classifier we'll use
    #model = linear_model.LogisticRegression(C=1)
    #model = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0, loss='ls')
    #model = ExtraTreesRegressor(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
    model = SVC(kernel='linear', probability=True)

    # === load data in memory === #
    print "loading data"
    y, X = load_data('train.csv')
    y_test, X_test = load_data('test.csv', use_labels=False)

    # === one-hot encoding === #
    # we want to encode the category IDs encountered both in
    # the training and the test set, so we fit the encoder on both
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    X_test = encoder.transform(X_test)
    #X_test = X_test.toarray()

    # if you want to create new features, you'll need to compute them
    # before the encoding, and append them to your dataset after

    # === training & metrics === #
    mean_auc = 0.0
    n = 10  # repeat the CV procedure 10 times to get more precise results
    for i in range(n):
        # for each iteration, randomly hold out 20% of the data as CV set
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size=.20, random_state=i*SEED)

        # if you want to perform feature selection / hyperparameter
        # optimization, this is where you want to do it

        # train model and make predictions
        model.fit(X_train, y_train) 
        preds = model.predict_proba(X_cv)[:, 1]

        # compute AUC metric for this CV fold
        fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
        roc_auc = metrics.auc(fpr, tpr)
        print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
        mean_auc += roc_auc

    print "Mean AUC: %f" % (mean_auc/n)

    # === Predictions === #
    # When making predictions, retrain the model on the whole training set
    model.fit(X, y)
    preds = model.predict_proba(X_test)[:, 1]
    filename = raw_input("Enter name for submission file: ")
    save_results(preds, "submission/"+ filename + ".csv")

if __name__ == '__main__':
    main()