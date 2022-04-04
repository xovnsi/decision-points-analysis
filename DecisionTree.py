from decision_tree_utils import class_entropy, get_split_gain, get_total_threshold, extract_rule
from Nodes import DecisionNode, LeafNode
from typing import Union
import pandas as pd


class DecisionTree(object):
    """ Implements a decision tree with C4.5 algorithm """
    def __init__(self, attributes_map):
        self._nodes = set()
        self._root_node = None
        # attributes map is a disctionary contianing the type of each attribute in the data.
        # Must be on of ['categorical', 'boolean', 'continuous']
        for attr_name in attributes_map.keys():
            if not attributes_map[attr_name] in ['continuous', 'categorical', 'boolean']:
                raise Exception('Attribute type not supported')
        self._attributes_map = attributes_map

    def delete_node(self, node) -> None:
        """ Removes a node from the tree's set of nodes and disconnects it from its parent node """
#        if node.get_parent_node() is None:
#            breakpoint()
        parent_node = node.get_parent_node()
        parent_node.delete_child(node)
        self._nodes.remove(node)

    def add_node(self, node, parent_node) -> None:
        """ Add a node to the tree's set of nodes and connects it to its parent node """
        node.set_parent_node(parent_node)
        self._nodes.add(node)
        if not parent_node is None:
            parent_node.add_child(node)
        elif node.get_label() == 'root':
            self._root_node = node
        else:
            raise Exception('Parent label not present in the tree')

    def _predict(self, row_in, node) -> str:
        """ Recursively traverse the tree (given the data in "row_in") until a leaf node and predicts the correspondent class """
        attribute = node.get_attribute().split(':')[0]
        child = node.get_child(row_in[attribute])
        if isinstance(child, LeafNode):
            return child.predict_class()
        else:
            return self._predict(row_in, child)

    def predict(self, data_in):
        """ Strating from the root, predicts the class corresponding to the features contained in "data_in" """
        attribute = self._root_node.get_attribute().split(":")[0]
        preds = list()
        # data_in is a pandas DataFrame
        #breakpoint()
        for index, row in data_in.iterrows():
            child = self._root_node.get_child(row[attribute])
            if isinstance(child, LeafNode):
                preds.append(child.predict_class())
            else:
                preds.append(self._predict(row, child))
        return preds

    def get_split(self, data_in) -> Union[float, float, str]:
        """ Compute the best split of the input data """
        #breakpoint()
        max_gain_ratio = None
        threshold_max_gain_ratio = None
        split_attribute = None
        # if there is only the target column or the aren't data the split doesn't exist 
        if len(data_in['target'].unique()) > 1 and len(data_in) > 0:
            max_gain_ratio = 0
            for column in data_in.columns:
                # gain ratio and threshold (if exist) for every feature 
                if not column in ['target', 'weight'] and len(data_in[column].unique()) > 1:
                    gain_ratio, threshold = get_split_gain(data_in[[column, 'target']], 
                        self._attributes_map[column])
                    # keep the best split
                    if gain_ratio > max_gain_ratio:
                        max_gain_ratio = gain_ratio
                        threshold_max_gain_ratio = threshold
                        split_attribute = column
        return max_gain_ratio, threshold_max_gain_ratio, split_attribute

    def split_node(self, node, data_in) -> None:
        """ Recurseviley split a node based on "data_in" until some conditions are met and a leaves nodes are added to the tree """ 
        #breakpoint()
        # categorical and boolean arguments can be selected only one time in a "line of succession"
        if not ('<' in node.get_label() or '>' in node.get_label()) and not node.get_label() == 'root': 
            data_in = data_in.copy(deep=True) 
            data_in = data_in.drop(columns=[node.get_label().split()[0]])
        max_gain_ratio, local_threshold, split_attribute = self.get_split(data_in)
        #breakpoint()
        # compute error predicting the most frequent class without splitting
        node_errors = data_in['target'].value_counts().sum() - data_in['target'].value_counts().max()
        # if split attribute does not exist then is a leaf 
        if not split_attribute is None:
            child_errors = self.compute_split_error(data_in[[split_attribute, 'target']], local_threshold)
            # if child errors are greater the actual error of the node than the split is useless
            if child_errors > node_errors:
                # the node (default type "DecisionNode") is "transformed" in a leaf node ("LeafNode" type)
                parent_node = node.get_parent_node()
                self.delete_node(node)
                node = LeafNode(dict(data_in['target'].value_counts()), node.get_label())
                self.add_node(node, parent_node)
            else:
                # compute global threshold from local one
                #breakpoint()
                threshold = get_total_threshold(data_in[split_attribute], local_threshold)
                # if the attribute with the greatest gain is continuous than the spit is binary
                if self._attributes_map[split_attribute] == 'continuous':
                    node.set_attribute('{}:{}'.format(split_attribute, threshold), 'continuous')
                    # create DecisionNode, recursion and add node
                    low_split_node = DecisionNode('{} <= {}'.format(split_attribute, threshold))
                    self.add_node(low_split_node, node)
                    data_known = data_in[data_in[split_attribute] != '?']
                    data_unknown = data_in[data_in[split_attribute] == '?']
                    weight_unknown = len(data_known[data_known[split_attribute] <= threshold]) / len(data_known)
                    data_unknown['weight'] = [weight_unknown] * len(data_unknown)
                    new_data_low = pd.concat([data_known[data_known[split_attribute] <= threshold], data_unknown], ignore_index=True)
                    self.split_node(low_split_node, new_data_low)
                    high_split_node = DecisionNode('{} > {}'.format(split_attribute, threshold))
                    self.add_node(high_split_node, node)
                    weight_unknown = len(data_known[data_known[split_attribute] > threshold]) / len(data_known)
                    data_unknown['weight'] = [weight_unknown] * len(data_unknown)
                    new_data_high = pd.concat([data_known[data_known[split_attribute] > threshold], data_unknown], ignore_index=True)
                    self.split_node(high_split_node, new_data_high)
                else:
                    # if the attribute is categorical or boolean than there is a node for every possible attribute value
                    node.set_attribute(split_attribute, self._attributes_map[split_attribute])
                    data_known = data_in[data_in[split_attribute] != '?']
                    data_unknown = data_in[data_in[split_attribute] == '?']
                    for attr_value in data_known[split_attribute].unique():
                        # create DecisionNode, recursion and add node
                        child_node = DecisionNode('{} = {}'.format(split_attribute, attr_value))
                        self.add_node(child_node, node)
                        weight_unknown = len(data_known[data_known[split_attribute] == attr_value]) / len(data_known)
                        data_unknown['weight'] = [weight_unknown] * len(data_unknown)
                        new_data = pd.concat([data_known[data_known[split_attribute] == attr_value], data_unknown], ignore_index=True)
                        self.split_node(child_node, new_data)
        else:
            # the node (default type "DecisionNode") is "transformed" in a leaf node ("LeafNode" type)
            parent_node = node.get_parent_node()
#            if parent_node is None:
#                breakpoint()
            self.delete_node(node)
            node = LeafNode(dict(data_in['target'].value_counts()), node.get_label())
            self.add_node(node, parent_node)

    def compute_split_error(self, data_in, threshold) -> int:
        """ Computes the error made by the split if predicting the most frequent class for every child born after it """
        attr_name = [column for column in data_in.columns if column != 'target'][0]
        attr_type = self._attributes_map[attr_name]
        # if continuous type the split is binary given by th threshold
        if attr_type == 'continuous':
            split_left = data_in[data_in[attr_name] <= threshold]
            # pandas function to count the occurnces of the different value of target
            values_count = split_left['target'].value_counts()
            # errors given by the difference between the sum of all occurrences and the most frequent
            errors_left = values_count.sum() - values_count.max()
            split_right = data_in[data_in[attr_name] > threshold]
            values_count = split_right['target'].value_counts()
            errors_right = values_count.sum() - values_count.max()
            total_child_error = errors_left + errors_right
        # if categorical or boolean, there is a child for every possible attribute value
        else:
            total_child_error = 0
            for attr_value in data_in[attr_name].unique():
                split = data_in[data_in[attr_name] == attr_value]
                values_count = split['target'].value_counts()
                total_child_error += values_count.sum() - values_count.max()
        return total_child_error

    def fit(self, data_in) -> None:
        """ Fits the tree on "data_in" """
        root_node = DecisionNode('root')
        self.add_node(root_node, None)
        # add weight to dataset in order to handle unknown values
        data_in['weight'] = [1] * len(data_in)
        data_in = data_in.fillna('?')
        self.split_node(self._root_node, data_in)

    def get_leaves_nodes(self):
        """ Returns a list of the leaves nodes """
        return [node for node in self._nodes if isinstance(node, LeafNode)]
        
    def extract_rules(self): 
        """ Returns a dictionary containing rulws extracted from the tree, already in logical form """
        rules = dict()
        # starting from the leaves going back to the root
        for leaf_node in self.get_leaves_nodes():
            rule = ""
            if not leaf_node._label_class in rules.keys():
                rules[leaf_node._label_class] = list()
            rule = "{} && {}".format(leaf_node.get_label(), extract_rule(leaf_node.get_parent_node(), rule))
            rule = "&&".join(rule.split("&&")[:-1])
            rules[leaf_node._label_class].append(rule)
        for target_class in rules.keys():
            rules[target_class] = "|| ".join(rules[target_class])
        return rules

