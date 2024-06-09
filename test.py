import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import graphviz
from ucimlrepo import fetch_ucirepo

from C45 import C45Classifier
# Import your C45 classifier here, if it's in another file
# from your_module import C45Classifier


def test_c45_classifier():
    # Load the Iris dataset
    # fetch dataset
    statlog_german_credit_data = fetch_ucirepo(id=144)

    # data (as pandas dataframes)
    X = statlog_german_credit_data.data.features
    y = statlog_german_credit_data.data.targets

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y['class'], test_size=0.2, random_state=42)
    print(X_train.head())

    # Initialize and train the classifier
    classifier = C45Classifier(min_samples_leaf=4, max_depth=8)
    classifier.fit(X_train, y_train)
    print(classifier.tree.depth())

    # Predict the labels for the testing set
    predictions = classifier.predict(X_test)

    # Evaluate the classifier
    classifier.evaluate(X_test, y_test)


    # Optionally generate a tree diagram (if Graphviz is installed and configured)
    try:
        classifier.generate_tree_diagram(graphviz, "tree_diagram_test_5")
    except Exception as e:
        print("Could not generate tree diagram:", e)

    # Print the rules derived from the tree
    # classifier.print_rules()

if __name__ == "__main__":
    test_c45_classifier()
