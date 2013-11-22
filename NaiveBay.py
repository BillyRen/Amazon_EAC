__author__ = 'Ren Chengbin'
__email__ = 'renchengbin@outlook.com'
__date__ = '11-22-2013'

"""
Naive Bayesian Approach for Amazon EAC
"""

SEED = 37

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics, cross_validation
from DataIO import load_data, save_results

def main():
    # === load data in memory === #
    print "loading data"
    y, X = load_data('train.csv')
    y_test, X_test = load_data('test.csv', use_labels=False)

    # === choose NB model === #
    clf = GaussianNB()

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
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_cv)[:, 1]

        # compute AUC metric for this CV fold
        fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
        roc_auc = metrics.auc(fpr, tpr)
        print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
        mean_auc += roc_auc

    print "Mean AUC: %f" % (mean_auc/n)

    # === predict === #
    clf.fit(X, y)
    preds = clf.predict_proba(X_test)[:, 1]
    filename = raw_input("Enter name for submission file: ")
    save_results(preds, "submission/"+ filename + ".csv")

if __name__ == '__main__':
    main()