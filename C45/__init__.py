import math
import pandas as pd
import numpy as np
from itertools import combinations


def is_numeric_attribute(data, attribute_index):
    # This assumes `data` is a DataFrame
    return (isinstance(data[0][attribute_index], int) and not isinstance(data[0][attribute_index], bool)) or isinstance(data[0][attribute_index], float)


def generate_unique_groupings(lst):
    '''
    Function to return the combinations for unique values
    :param lst:
    :return:
    '''
    def recursive_groupings(elements):
        if not elements:
            return [[]]
        result = []
        for i in range(1, len(elements) + 1):
            for combination in combinations(elements, i):
                remaining = [e for e in elements if e not in combination]
                for sub_grouping in recursive_groupings(remaining):
                    # Sort inner lists to ensure uniqueness
                    sorted_combination = sorted([list(combination)] + sub_grouping)
                    result.append(sorted_combination)
        return result

    # Start the recursive grouping process
    all_groupings = recursive_groupings(lst)

    # Filter out groupings that don't use all elements
    valid_groupings = [grouping for grouping in all_groupings if sum(len(group) for group in grouping) == len(lst)]

    # Use a set to remove duplicates
    unique_groupings = []
    seen = set()
    for grouping in valid_groupings:
        # Convert grouping to a tuple of tuples so it can be added to a set
        grouping_tuple = tuple(tuple(sorted(group)) for group in grouping)
        if grouping_tuple not in seen:
            seen.add(grouping_tuple)
            unique_groupings.append(grouping)

    return unique_groupings


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
        best_gain_ratio = -float('inf')

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
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_threshold = threshold

        return best_threshold, best_gain_ratio

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
                split_data.append(record)
                split_weights.append(weights[i])
        # print(split_data)
        # print(split_weights)
        return split_data, split_weights

    def __split_data_categorical(self, data, attribute_index, grouping, weights):
        split_data = {tuple(group): [] for group in grouping}
        split_weights = {tuple(group): [] for group in grouping}

        for i, record in enumerate(data):
            found = False
            for group in grouping:
                if record[attribute_index] in group:
                    split_data[tuple(group)].append(record)
                    split_weights[tuple(group)].append(weights[i])
                    found = True
                    break
            if not found:
                split_data[('Other',)].append(record)
                split_weights[('Other',)].append(weights[i])

        return split_data, split_weights

    def __select_best_attribute_c50(self, data, attributes, weights):
        total_entropy = self.__calculate_entropy(data, weights)
        best_attribute = None
        best_gain_ratio = 0.0
        best_threshold = None
        best_grouping = None
        print(attributes)
        for attribute_index in range(len(attributes)):
            if is_numeric_attribute(data, attribute_index):
                threshold, measure = self.find_best_threshold(data, attribute_index, weights, total_entropy)
                if measure > best_gain_ratio:
                    best_gain_ratio = measure
                    best_attribute = attribute_index
                    best_threshold = threshold
                    best_grouping = None
            else:
                attribute_values = list(set([record[attribute_index] for record in data]))
                unique_groupings = generate_unique_groupings(attribute_values)

                for grouping in unique_groupings:
                    split_data, split_weights = self.__split_data_categorical(data, attribute_index, grouping, weights)
                    attribute_entropy = 0.0
                    split_info = 0.0

                    for group in grouping:
                        group_data = split_data[tuple(group)]
                        group_weights = split_weights[tuple(group)]
                        if group_data:
                            subset_entropy = self.__calculate_entropy(group_data, group_weights)
                            subset_probability = sum(group_weights) / sum(weights)
                            attribute_entropy += subset_probability * subset_entropy
                            if subset_probability > 0:
                                split_info -= subset_probability * math.log2(subset_probability)

                    gain = total_entropy - attribute_entropy
                    gain_ratio = gain / split_info if split_info != 0 else 0

                    if gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_attribute = attribute_index
                        best_threshold = None
                        best_grouping = grouping
            print(best_grouping)
        print(best_attribute)
        return best_attribute, best_threshold, best_grouping

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
        if self.max_depth is not None and depth >= self.max_depth:
            return _LeafNode(self.__majority_class(data, weights), sum(weights))

        if sum(weights) < self.min_samples_leaf:
            return _LeafNode(self.__majority_class(data, weights), sum(weights))

        class_labels = set([record[-1] for record in data])
        if len(class_labels) == 1:
            return _LeafNode(class_labels.pop(), sum(weights))

        if len(attributes) == 1:
            return _LeafNode(self.__majority_class(data, weights), sum(weights))

        best_attribute, best_threshold, best_grouping = self.__select_best_attribute_c50(data, attributes, weights)

        if best_attribute is None:
            return _LeafNode(self.__majority_class(data, weights), sum(weights))

        best_attribute_name = attributes[best_attribute]

        tree = _DecisionNode(best_attribute_name)
        # new_attributes = attributes[:best_attribute] + attributes[best_attribute + 1:]

        if best_threshold is not None:
            # left_data = [rec for rec in data if rec[best_attribute] <= best_threshold]
            # right_data = [rec for rec in data if rec[best_attribute] > best_threshold]
            # left_weights = [weights[i] for i, rec in enumerate(data) if rec[best_attribute] <= best_threshold]
            # right_weights = [weights[i] for i, rec in enumerate(data) if rec[best_attribute] > best_threshold]
            #
            # if sum(left_weights) >= self.min_samples_leaf:
            #     tree.add_child("<= " + str(best_threshold),
            #                    self.__build_decision_tree(left_data, new_attributes, left_weights, depth + 1))
            # else:
            #     tree.add_child("<= " + str(best_threshold),
            #                    _LeafNode(self.__majority_class(left_data, left_weights), sum(left_weights)))
            #
            # if sum(right_weights) >= self.min_samples_leaf:
            #     tree.add_child("> " + str(best_threshold),
            #                    self.__build_decision_tree(right_data, new_attributes, right_weights, depth + 1))
            # else:
            #     tree.add_child("> " + str(best_threshold),
            #                    _LeafNode(self.__majority_class(right_data, right_weights), sum(right_weights)))
            pass
        else:
            split_data, split_weights = self.__split_data_categorical(data, best_attribute, best_grouping, weights)
            for group in best_grouping:
                subset = split_data[tuple(group)]
                subset_weights = split_weights[tuple(group)]
                value = " + ".join(map(str, group))

                if sum(subset_weights) >= self.min_samples_leaf:
                    tree.add_child(value, self.__build_decision_tree(subset, attributes, subset_weights, depth + 1))
                else:
                    tree.add_child(value, _LeafNode(self.__majority_class(subset, subset_weights), sum(subset_weights)))
            if 'Other' in split_data:
                subset = split_data[('Other',)]
                subset_weights = split_weights[('Other',)]
                if sum(subset_weights) >= self.min_samples_leaf:
                    tree.add_child('Other', self.__build_decision_tree(subset, attributes, subset_weights, depth + 1))
                else:
                    tree.add_child('Other', _LeafNode(self.__majority_class(subset, subset_weights), sum(subset_weights)))

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
        dot = graphviz.Digraph()

        def build_tree(node, parent_node=None, edge_label=None):
            if isinstance(node, _DecisionNode):
                current_node_label = str(node.attribute)
                dot.node(str(id(node)), label=current_node_label)

                if parent_node:
                    dot.edge(str(id(parent_node)), str(id(node)), label=str(edge_label))

                for value, child_node in node.children.items():
                    if " + " in value:
                        value = "Other"
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
