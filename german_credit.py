import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import graphviz
from ucimlrepo import fetch_ucirepo
from sklearn.utils import resample

from C45 import C45Classifier
# Import your C45 classifier here, if it's in another file
# from your_module import C45Classifier


def test_c45_classifier():
    df = pd.read_csv("german_credit.csv")

    df.dropna(inplace=True)

    # Separate majority and minority classes
    df_majority = df[df["class"] == df["class"].value_counts().idxmax()]
    df_minority = df[df["class"] == df["class"].value_counts().idxmin()]

    # Downsample majority class
    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=len(df_minority),  # to match minority class
                                       random_state=42)  # reproducible results

    # Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    # Display new class counts
    print(df_balanced["class"].value_counts())


    X = df_balanced.drop(columns=["class", "Unnamed: 0"])
    y = df_balanced["class"]

    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #print(X_train.head())

    print(y.value_counts())


    max_depths = [3, 4, 5, 6, 7, 8]
    max_splits = [2, 3]

    max_accuracy = 0

    for depth in max_depths:
        depth_accuracy = 0
        for max_split in max_splits:
            # Initialize and train the classifier
            classifier = C45Classifier(max_depth=depth)
            classifier.fit(X, y)
            #print(classifier.tree.depth())

            # Predict the labels for the testing set
            #predictions = classifier.predict(X_test)

            # accuracy = classifier.evaluate(X, y)
            # if accuracy > max_accuracy:
            #     max_accuracy = accuracy
            #     best_depth = depth
            #     best_split = max_splits
            #     best_classifier = classifier
            #
            # if accuracy > depth_accuracy:
            #     best_for_depth = classifier

            # Evaluate the classifier
            print(f"Depth : {depth}, max_splits: {max_split}: ")
        try:
            classifier.generate_tree_diagram(graphviz, f"german_credit_d_{depth}")
        except Exception as e:
            print("Could not generate tree diagram:", e)

    print(f"Best depth:{best_depth}, splits: {best_split}: Accuracy: {max_accuracy}")
    # Optionally generate a tree diagram (if Graphviz is installed and configured)
    try:
        best_classifier.generate_tree_diagram(graphviz, "german_credit_c4_best")
    except Exception as e:
        print("Could not generate tree diagram:", e)

    # Print the rules derived from the tree
    best_classifier.print_rules()

if __name__ == "__main__":
    test_c45_classifier()
