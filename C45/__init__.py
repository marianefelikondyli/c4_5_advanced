import math
import pandas as pd
import numpy as np


def is_numeric_attribute(data, attribute_index):
    # This assumes `data` is a DataFrame
    return isinstance(data[0][attribute_index], int) or isinstance(data[0][attribute_index], float)


class _DecisionNode:
    def __init__(self, attribute):
        # Initialize decision node with the given attribute
        self.attribute = attribute
        self.children = {}  # Store children of decision node

    def depth(self):
        # Calculate the depth of the decision node
        if len(self.children) == 0:
            return 1
        else:
            max_depth = 0
            for child in self.children.values():
                if isinstance(child, _DecisionNode):
                    child_depth = child.depth()
                    if child_depth > max_depth:
                        max_depth = child_depth
            return max_depth + 1

    def add_child(self, value, node):
        # Add a child to the decision node with the given attribute value
        self.children[value] = node

    def count_leaves(self):
        if len(self.children) == 0:
            return 1
        else:
            count = 0
            for child in self.children.values():
                if isinstance(child, _DecisionNode):
                    count += child.count_leaves()
                else:
                    count += 1
            return count


class _LeafNode:
    def __init__(self, label, weight):
        # Initialize leaf node with the class label and weight
        self.label = label
        self.weight = weight


class C45Classifier:
    def __init__(self, min_samples_leaf=1, max_depth=None):
        self.tree = None
        self.attributes = None
        self.data = None
        self.weight = 1
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

    def find_best_threshold(self, data, attribute_index, weights, total_entropy):
        # Extract the specific column from data
        column_data = [row[attribute_index] for row in data]
        # Obtain sorted unique values from the column
        unique_values = sorted(set(column_data))

        best_threshold = None
        best_gain = -float('inf')

        for i in range(1, len(unique_values)):
            threshold = (unique_values[i - 1] + unique_values[i]) / 2
            split_info = 0.0
            attribute_entropy = 0.0

            # Create subsets based on the threshold
            for subset in ["left", "right"]:
                if subset == "left":
                    subset_indices = [index for index, value in enumerate(column_data) if value <= threshold]
                else:
                    subset_indices = [index for index, value in enumerate(column_data) if value > threshold]

                subset_data = [data[index] for index in subset_indices]
                subset_weights = [weights[index] for index in subset_indices]

                # Calculate entropy if subset is not empty
                if subset_data:
                    subset_entropy = self.__calculate_entropy(subset_data, subset_weights)
                    subset_probability = sum(subset_weights) / sum(weights)
                    attribute_entropy += subset_probability * subset_entropy
                    if subset_probability > 0:
                        split_info -= subset_probability * math.log2(subset_probability)

            # Calculate gain and gain ratio
            gain = total_entropy - attribute_entropy
            gain_ratio = gain / split_info if split_info != 0 else 0

            # Check if this threshold provides a better gain ratio
            if gain_ratio > best_gain:
                best_gain = gain_ratio
                best_threshold = threshold

        return best_threshold, best_gain

    def __calculate_entropy(self, data, weights):
        # Calculate entropy of the given dataset
        class_counts = {}  # Count occurrences of each class label
        total_weight = 0.0  # Calculate total data weight

        for i, record in enumerate(data):
            label = record[-1]  # Get class label of each data
            weight = weights[i]  # Get weight of each data

            if label not in class_counts:
                class_counts[label] = 0.0
            class_counts[label] += weight
            total_weight += weight

        entropy = 0.0

        for count in class_counts.values():
            probability = count / total_weight  # Calculate probability of each class label
            entropy -= probability * math.log2(probability)  # Calculate entropy contribution of each label

        return entropy

    def __split_data(self, data, attribute_index, attribute_value, weights):
        # Split dataset based on the given attribute value
        split_data = []  # Store matching data subsets
        split_weights = []  # Store matching weight subsets

        for i, record in enumerate(data):
            if record[attribute_index] == attribute_value:
                split_data.append(record[:attribute_index] + record[attribute_index+1:])
                split_weights.append(weights[i])

        return split_data, split_weights

    def __select_best_attribute_c50(self, data, attributes, weights):
        # Select the best attribute to split the dataset using the C5.0 algorithm
        total_entropy = self.__calculate_entropy(data, weights)
        best_attribute = None
        best_gain_ratio = 0.0  # Can be information gain or gain ratio based on your implementation
        best_threshold = None  # Only applicable for numeric attributes

        for attribute_index in range(len(attributes)):
            if is_numeric_attribute(data, attribute_index):
                threshold, measure = self.find_best_threshold(data, attribute_index, weights, total_entropy)
                if measure > best_gain_ratio:
                    best_gain_ratio = measure
                    best_attribute = attribute_index
                    best_threshold = threshold
            else:
                split_info = 0.0
                attribute_values = set([record[attribute_index] for record in data])
                attribute_entropy = 0.0

                for value in attribute_values:
                    subset, subset_weights = self.__split_data(data, attribute_index, value, weights)
                    subset_entropy = self.__calculate_entropy(subset, subset_weights)
                    subset_probability = sum(subset_weights) / sum(weights)
                    attribute_entropy += subset_probability * subset_entropy
                    split_info -= subset_probability * math.log2(subset_probability)
                    gain = total_entropy - attribute_entropy
                    if split_info != 0.0:
                        gain_ratio = gain / split_info
                    else:
                        gain_ratio = 0.0

                    if gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_attribute = attribute_index
                        best_threshold = None
        return best_attribute, best_threshold

    def __majority_class(self, data, weights):
        # Determine the majority class in the dataset
        class_counts = {}

        for i, record in enumerate(data):
            label = record[-1]
            weight = weights[i]

            if label not in class_counts:
                class_counts[label] = 0.0
            class_counts[label] += weight

        majority_class = None
        max_count = 0.0

        for label, count in class_counts.items():
            if count > max_count:
                max_count = count
                majority_class = label

        return majority_class

    def __build_decision_tree(self, data, attributes, weights, depth=0):
        # Base case: check if maximum depth has been reached
        if self.max_depth is not None and depth >= self.max_depth:
            return _LeafNode(self.__majority_class(data, weights), sum(weights))

        # Check if the current node has enough samples
        if sum(weights) < self.min_samples_leaf:
            return _LeafNode(self.__majority_class(data, weights), sum(weights))

        # Existing check for homogeneity
        class_labels = set([record[-1] for record in data])
        if len(class_labels) == 1:
            return _LeafNode(class_labels.pop(), sum(weights))

        # Check if any attributes are left to split on
        if not attributes:
            return _LeafNode(self.__majority_class(data, weights), sum(weights))

        # Select the best attribute and threshold to split the dataset
        best_attribute, best_threshold = self.__select_best_attribute_c50(data, attributes, weights)
        if best_attribute is None:
            return _LeafNode(self.__majority_class(data, weights), sum(weights))

        best_attribute_name = attributes[best_attribute]
        tree = _DecisionNode(best_attribute_name)
        new_attributes = attributes[:best_attribute] + attributes[best_attribute + 1:]

        # Handle splits based on the best attribute and threshold found
        if best_threshold is not None:
            left_data = [rec for rec in data if rec[best_attribute] <= best_threshold]
            right_data = [rec for rec in data if rec[best_attribute] > best_threshold]
            left_weights = [weights[i] for i, rec in enumerate(data) if rec[best_attribute] <= best_threshold]
            right_weights = [weights[i] for i, rec in enumerate(data) if rec[best_attribute] > best_threshold]

            # Recursively build the tree for each subset, only if they meet the min_samples_leaf criterion
            if sum(left_weights) >= self.min_samples_leaf:
                tree.add_child("<= " + str(best_threshold),
                               self.__build_decision_tree(left_data, new_attributes, left_weights, depth + 1))
            else:
                tree.add_child("<= " + str(best_threshold),
                               _LeafNode(self.__majority_class(left_data, left_weights), sum(left_weights)))

            if sum(right_weights) >= self.min_samples_leaf:
                tree.add_child("> " + str(best_threshold),
                               self.__build_decision_tree(right_data, new_attributes, right_weights, depth + 1))
            else:
                tree.add_child("> " + str(best_threshold),
                               _LeafNode(self.__majority_class(right_data, right_weights), sum(right_weights)))
        else:
            for value in set(rec[best_attribute] for rec in data):
                subset = [rec for rec in data if rec[best_attribute] == value]
                subset_weights = [weights[i] for i, rec in enumerate(data) if rec[best_attribute] == value]

                if sum(subset_weights) >= self.min_samples_leaf:
                    tree.add_child(value, self.__build_decision_tree(subset, new_attributes, subset_weights, depth + 1))
                else:
                    tree.add_child(value, _LeafNode(self.__majority_class(subset, subset_weights), sum(subset_weights)))

        return tree

    def __make_tree(self, data, attributes, weights):
        # Make decision tree using the given dataset, attributes, and weights
        return self.__build_decision_tree(data, attributes, weights)

    def __train(self, data, weight=1):
        self.weight = weight
        # Train decision tree using the given dataset
        self.attributes = data.columns.tolist()[:-1]  # Get attributes from dataset columns
        weights = [self.weight] * len(data)  # Initialize weights with same value for each data
        self.tree = self.__make_tree(data.values.tolist(), self.attributes, weights)
        self.data = data

    def __classify(self, tree=None, instance=[]):
        if self.tree is None:
            raise Exception('Decision tree has not been trained yet!')
        # Classify instance using the decision tree
        if tree is None:
            tree = self.tree

        if isinstance(tree, _LeafNode):
            return tree.label

        attribute = tree.attribute
        attribute_index = self.attributes.index(attribute)
        attribute_values = instance[attribute_index]

        if attribute_values in tree.children:
            child_node = tree.children[attribute_values]
            return self.__classify(child_node, instance)
        else:
            class_labels = []
            for child_node in tree.children.values():
                if isinstance(child_node, _LeafNode):
                    class_labels.append(child_node.label)
            if len(class_labels) == 0:
                return self.__majority_class(self.data.values.tolist(), [1.0] * len(self.data))
            majority_class = max(set(class_labels))
            return majority_class

    def fit(self, data, label, weight=1):
        # Train decision tree using the given dataset
        if isinstance(data, pd.DataFrame):
            data = pd.concat([data, label], axis=1)
        else:
            data = pd.DataFrame(np.c_[data, label])
        self.__train(data, weight)

    def predict(self, data):
        # Predict class label for each data in the dataset
        if isinstance(data, pd.DataFrame):
            data = data.values.tolist()
        elif isinstance(data, list) and isinstance(data[0], dict):
            data = [list(d.values()) for d in data]

        if len(data[0]) != len(self.attributes):
            raise Exception('Number of variables in data and attributes do not match!')

        return [self.__classify(None, record) for record in data]

    def evaluate(self, x_test, y_test):
        # Mengevaluasi performa pohon keputusan menggunakan akurasi
        y_pred = self.predict(x_test)
        # print type y_test
        if isinstance(y_test, pd.Series):
            y_test = y_test.values.tolist()

    #     print every acc of each class
        acc = {}
        true_pred = 0
        real_acc ={}
        for i in range(len(y_test)):
            if y_test[i] not in real_acc:
                real_acc[y_test[i]] = 0
            real_acc[y_test[i]] += 1
            if y_test[i] == y_pred[i]:
                if y_test[i] not in acc:
                    acc[y_test[i]] = 0
                acc[y_test[i]] += 1
                true_pred += 1
        for key in acc:
            acc[key] /= real_acc[key]
    #     mean acc
        total_acc = true_pred / len(y_test)
        print("Evaluation result: ")
        print("Total accuracy: ", total_acc)
        for key in acc:
            print("Accuracy ", key, ": ", acc[key])

    def generate_tree_diagram(self, graphviz, filename):
        # Generate decision tree diagram using graphviz module
        dot = graphviz.Digraph()

        def build_tree(node, parent_node=None, edge_label=None):
            if isinstance(node, _DecisionNode):
                current_node_label = str(node.attribute)
                dot.node(str(id(node)), label=current_node_label)

                if parent_node:
                    dot.edge(str(id(parent_node)), str(id(node)), label=str(edge_label))

                for value, child_node in node.children.items():
                    build_tree(child_node, node, str(value))
            elif isinstance(node, _LeafNode):
                current_node_label = f"Class: {str(node.label)}, Weight: {str(node.weight)}"
                dot.node(str(id(node)), label=current_node_label, shape="box")

                if parent_node:
                    dot.edge(str(id(parent_node)), str(id(node)), label=str(edge_label))

        build_tree(self.tree)
        dot.format = 'png'
        return dot.render(filename, view=False)

    def print_rules(self, tree=None, rule=''):
        if self.tree is None:
            raise Exception('Decision tree has not been trained yet!')
        # Print rules generated by the decision tree
        if tree is None:
            tree = self.tree
        if rule != '':
            rule += ' AND '
        if isinstance(tree, _LeafNode):
            print(rule[:-3] + ' => ' + str(tree.label))
            return

        attribute = tree.attribute
        for value, child_node in tree.children.items():
            self.print_rules(child_node, rule + attribute + ' = ' + str(value))

    def rules(self):
        rules = []

        def build_rules(node, parent_node=None, edge_label=None, rule=''):
            if isinstance(node, _DecisionNode):
                current_node_label = node.attribute
                if parent_node:
                    rule += f" AND {current_node_label} = {edge_label}"
                for value, child_node in node.children.items():
                    build_rules(child_node, node, value, rule)
            elif isinstance(node, _LeafNode):
                current_node_label = f"Class: {node.label}, Weight: {node.weight}"
                if parent_node:
                    rule += f" => {current_node_label}"
                rules.append(rule[5:])
        build_rules(self.tree)
        return rules

    def summary(self):
        # Print summary
        print("Decision Tree Classifier Summary")
        print("================================")
        print("Number of Instances   : ", len(self.data))
        print("Number of Attributes  : ", len(self.attributes))
        print("Number of Leaves      : ", self.tree.count_leaves())
        print("Number of Rules       : ", len(self.rules()))
        print("Tree Depth            : ", self.tree.depth())
