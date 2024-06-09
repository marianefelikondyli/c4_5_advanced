from C45 import C45Classifier
import pandas as pd
import numpy as np
import graphviz

# Sample dataset
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

# Instantiate the C45Classifier
classifier = C45Classifier(max_depth=5)

# Separate features and labels
features = data.drop('PlayTennis', axis=1)
labels = data['PlayTennis']

# Fit the model
classifier.fit(features, labels)

# Create a test set
test_data = pd.DataFrame({
    'Outlook': ['Sunny', 'Rain', 'Overcast'],
    'Temperature': ['Hot', 'Cool', 'Mild'],
    'Humidity': ['High', 'Normal', 'High'],
    'Windy': [False, True, True]
})

# Predict using the test set
predictions = classifier.predict(test_data)
print("Predictions:", predictions)

# Evaluate the model
# Creating a dummy test set with labels to demonstrate evaluation
test_data_with_labels = pd.DataFrame({
    'Outlook': ['Sunny', 'Rain', 'Overcast'],
    'Temperature': ['Hot', 'Cool', 'Mild'],
    'Humidity': ['High', 'Normal', 'High'],
    'Windy': [False, True, True],
    'PlayTennis': ['No', 'Yes', 'Yes']
})
x_test = test_data_with_labels.drop('PlayTennis', axis=1)
y_test = test_data_with_labels['PlayTennis']
classifier.evaluate(x_test, y_test)

# Print decision rules
print("Decision Rules:")
classifier.print_rules()

# Generate a visual representation of the decision tree
classifier.generate_tree_diagram(graphviz, "decision_tree")

# Print summary
classifier.summary()

