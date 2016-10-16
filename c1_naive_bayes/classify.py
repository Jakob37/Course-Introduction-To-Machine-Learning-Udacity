def NBAccuracy(features_train, labels_train, features_test, labels_test):

    from sklearn.naive_bayes import GaussianNB

    # Create classifier, fit to train data
    clf = GaussianNB()
    clf.fit(features_train, labels_train)

    # Generate predicted labels based on test data
    pred = clf.predict(features_test)

    # Evaluate accuracy of predictions vs. real labels
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(labels_test, pred)

    return accuracy
