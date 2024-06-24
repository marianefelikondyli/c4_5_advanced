import pandas as pd
from sklearn.model_selection import train_test_split
import graphviz

from C45 import C45Classifier
# Import your C45 classifier here, if it's in another file
# from your_module import C45Classifier


def test_c45_classifier():

    df = pd.read_csv("/kaggle/input/compas/compas.csv")

    df.dropna(inplace=True)

    X = df.drop(columns=["two_year_recid", "Unnamed: 0"])
    y = df["two_year_recid"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.head())

    max_depths = [3, 4, 5, 6, 7, 8]
    min_sample_leafs = [1, 2, 3, 4]
    max_splits = [2]

    max_accuracy = 0

    for depth in max_depths:
        for min_sample_leaf in min_sample_leafs:
            for max_split in max_splits:
                # Initialize and train the classifier
                classifier = C45Classifier(min_samples_leaf=min_sample_leaf, max_depth=depth, max_splits=max_split)
                classifier.fit(X_train, y_train)
                # print(classifier.tree.depth())

                # Predict the labels for the testing set
                # predictions = classifier.predict(X_test)

                accuracy = classifier.evaluate(X_test, y_test)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    best_depth = depth
                    best_sample = min_sample_leaf
                    best_split = max_splits
                    best_classifier = classifier

                # Evaluate the classifier
                print(f"Depth : {depth}, sample_leaf: {min_sample_leaf}, max_splits: {max_split}: Accuracy: {accuracy}")
        try:
            classifier.generate_tree_diagram(graphviz, f"compas_c4_5_depth_{depth}")
        except Exception as e:
            print("Could not generate tree diagram:", e)

    print(f"Best depth:{best_depth}, sample_leaf: {best_sample}, splits: {best_split}: Accuracy: {max_accuracy}")
    # Optionally generate a tree diagram (if Graphviz is installed and configured)
    try:
        best_classifier.generate_tree_diagram(graphviz, "compas_c4_5")
    except Exception as e:
        print("Could not generate tree diagram:", e)

    # Print the rules derived from the tree
    best_classifier.print_rules()


if __name__ == "__main__":
    test_c45_classifier()
