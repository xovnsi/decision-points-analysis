from decision_tree_utils import class_entropy, get_split_gain, get_total_threshold, extract_rules_from_leaf
from Nodes import DecisionNode, LeafNode
from typing import Union
import pandas as pd
import numpy as np
import scipy.stats as stats
from operator import itemgetter
from tqdm import tqdm
import multiprocessing


class DecisionTree(object):
    """ Implements a decision tree with C4.5 algorithm """

    def __init__(self, attributes_map, max_depth=20):
        self._nodes = set()
        self._root_node = None
        self.max_depth = max_depth
        # attributes map is a disctionary contianing the type of each attribute in the data.
        # Must be one of ['categorical', 'boolean', 'continuous']
        for attr_name in attributes_map.keys():
            if not attributes_map[attr_name] in ['continuous', 'categorical', 'boolean']:
                raise Exception('Attribute type not supported')
        self._attributes_map = attributes_map

    def delete_node(self, node) -> None:
        """ Removes a node from the tree's set of nodes and disconnects it from its parent node """
        parent_node = node.get_parent_node()
        if node.get_parent_node() is None:
            raise Exception("Can't delete node {}. Parent node not found".format(node._label))
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
            raise Exception("Can't add node {}. Parent label not present in the tree".format(node._label))

    def _predict(self, row_in, node, predictions_dict) -> dict:
        """ Recursively traverse the tree (given the data in "row_in") until a leaf node and returns the class distribution """
        if node is None:
            raise Exception("Can't traverse the tree. Node is None")
        attribute = node.get_attribute().split(':')[0]
        # In the attribute is known, only the correspondent child is explored.
        if row_in[attribute] != '?':
            child = node.get_child(row_in[attribute])
            if child is None:
                raise Exception("Can't find child with attribute '{}'".format(attribute))
            if isinstance(child, LeafNode):
                for target in child._classes.keys():
                    if not target in predictions_dict.keys():
                        predictions_dict[target] = [(child._classes[target], sum(child._classes.values()))]
                    else:
                        predictions_dict[target].append((child._classes[target], sum(child._classes.values()))) # info about the leaf classes
                    predictions_dict['total_sum'] += child._classes[target] # total sum is needed for the final probability computation
                return predictions_dict
            else:
                return self._predict(row_in, child, predictions_dict)
        else:
            # In case of unknown attribute, the prediction is spread on every child node.
            for child in node.get_childs():
                if isinstance(child, LeafNode):
                    for target in child._classes.keys():
                        if not target in predictions_dict.keys():
                            predictions_dict[target] = [(child._classes[target], sum(child._classes.values()))]
                        else:
                            predictions_dict[target].append((child._classes[target], sum(child._classes.values()))) # info about the leaf classes
                        predictions_dict['total_sum'] += child._classes[target]
                else:
                    predictions_dict = self._predict(row_in, child, predictions_dict)
            return predictions_dict

    def predict(self, data_in, distribution=False):
        """ Starting from the root, predicts the class corresponding to the features contained in "data_in" """
        attribute = self._root_node.get_attribute().split(":")[0]
        data_in = data_in.fillna('?')
        preds = list()
        # data_in is a pandas DataFrame
        for index, row in data_in.iterrows():
            # Every class will have a list of tuple corresponding to every leaf reached (in case of unknown attribute) 
            #predictions_dict = {k: [] for k in data_in['target'].unique()}
            predictions_dict = {'total_sum': 0}
            #predictions_dict['total_sum'] = 0
            # if attribute is known, only the correspondent child is explored
            if row[attribute] != '?':
                child = self._root_node.get_child(row[attribute])
                if child is None:
                    raise Exception("Can't find child with attribute '{}'".format(attribute))
                if isinstance(child, LeafNode):
                    for target in child._classes.keys():
                        if not target in predictions_dict.keys():
                            predictions_dict[target] = [(child._classes[target], sum(child._classes.values()))]
                        else:
                            predictions_dict[target].append((child._classes[target], sum(child._classes.values()))) # info about the leaf classes
                        predictions_dict['total_sum'] += child._classes[target] # needed for the final probability computation
                    preds.append(predictions_dict)
                else:
                    preds.append(self._predict(row, child, predictions_dict))
            else:
                # if attribute is unknown, the case is spreaded to every child node
                for child in self._root_node.get_childs():
                    if isinstance(child, LeafNode):
                        for target in child._classes.keys():
                            if not target in predictions_dict.keys():
                                predictions_dict[target] = [(child._classes[target], sum(child._classes.values()))]
                            else:
                                predictions_dict[target].append((child._classes[target], sum(child._classes.values()))) # info about the leaf classes
                            predictions_dict['total_sum'] += child._classes[target] # needed for the final probability computation
                    else:
                        # recursive part
                        predictions_dict = self._predict(row, child, predictions_dict)
                preds.append(predictions_dict)
        # probability distribution computation for every prediction required
        out_preds = list() 
        out_distr = list()
        for pred in preds:
            pred_distribution = {k: [] for k in pred.keys() if k != 'total_sum'} 
            for target in pred_distribution.keys():
                cond_prob = 0
                # for every leaf selected in the previous part, containing the considered target class
                for conditional in pred[target]:
                    cond_prob += conditional[0] / pred['total_sum']
                pred_distribution[target] = np.round(cond_prob, 4)
            # select the class with max probability
            out_preds.append(max(pred_distribution, key=pred_distribution.get))
            out_distr.append(pred_distribution)
        # return also the distribution or just the prediction
        if distribution:
            out_func = (out_preds, out_distr)
        else:
            out_func = out_preds

        return out_func

    def get_split(self, data_in) -> Union[float, float, str]:
        """ Compute the best split of the input data """
        max_gain_ratio = None
        threshold_max_gain_ratio = None
        split_attribute = None
        # if there is only the target column or the aren't data the split doesn't exist 
        if len(data_in['target'].unique()) > 1 and len(data_in) > 0:
            # in order the split to be chosen, its information gain must be at least equal to the mean of all the tests considered 
            tests_examined = {'gain_ratio': list(), 'info_gain': list(),
                    'threshold': list(), 'attribute': list(), 'not_near_trivial_subset': list()}
            for column in data_in.columns:
                # gain ratio and threshold (if exist) for every feature 
                if not column in ['target', 'weight'] and len(data_in[column].unique()) > 1:
                    gain_ratio, info_gain, threshold, are_there_at_least_two = get_split_gain(data_in[[column, 'target', 'weight']], 
                        self._attributes_map[column])
                    tests_examined['gain_ratio'].append(gain_ratio)
                    tests_examined['info_gain'].append(info_gain)
                    tests_examined['threshold'].append(threshold)
                    tests_examined['attribute'].append(column)
                    tests_examined['not_near_trivial_subset'].append(are_there_at_least_two)
            #breakpoint()
            # select the best split
            tests_examined = pd.DataFrame.from_dict(tests_examined)
            mean_info_gain = tests_examined['info_gain'].mean()
            # The best split must have at the least two subset with at least two cases 
            # TODO the above condition should be user dependent
            select_max_gain_ratio = tests_examined[(tests_examined['info_gain'] >= mean_info_gain) & (tests_examined['not_near_trivial_subset'])]
            if len(select_max_gain_ratio) != 0:
                max_gain_ratio_idx = select_max_gain_ratio['gain_ratio'].idxmax()
                max_gain_ratio = select_max_gain_ratio.loc[max_gain_ratio_idx, 'gain_ratio']
                max_gain_ratio_threshold = select_max_gain_ratio.loc[max_gain_ratio_idx, 'threshold']
                split_attribute = select_max_gain_ratio.loc[max_gain_ratio_idx, 'attribute']
            elif len(tests_examined[tests_examined['not_near_trivial_subset']]) != 0:
                select_max_gain_ratio = tests_examined[tests_examined['not_near_trivial_subset']]   # Otherwise 'select_max_gain_ratio' computed before is empty
                max_gain_ratio_idx = select_max_gain_ratio['gain_ratio'].idxmax()
                max_gain_ratio = select_max_gain_ratio.loc[max_gain_ratio_idx, 'gain_ratio']
                max_gain_ratio_threshold = select_max_gain_ratio.loc[max_gain_ratio_idx, 'threshold']
                split_attribute = select_max_gain_ratio.loc[max_gain_ratio_idx, 'attribute']
            else:
                max_gain_ratio = None
                max_gain_ratio_threshold = None
                split_attribute = None
        else:
            max_gain_ratio = None
            max_gain_ratio_threshold = None
            split_attribute = None
            
        return max_gain_ratio, max_gain_ratio_threshold, split_attribute

    def split_node(self, node, data_in, data_total) -> None:
        """ Recurseviley split a node based on "data_in" until some conditions are met and a leaves nodes are added to the tree """ 
        # categorical and boolean arguments can be selected only one time in a "line of succession"
        if not ('<' in node.get_label() or '>' in node.get_label()) and not node.get_label() == 'root': 
            data_in = data_in.copy(deep=True) 
            data_in = data_in.drop(columns=[node.get_label().split()[0]])
        max_gain_ratio, local_threshold, split_attribute = self.get_split(data_in)
        #breakpoint()
        # compute error predicting the most frequent class without splitting
        node_errors = data_in['target'].value_counts().sum() - data_in['target'].value_counts().max()
        # compute percentage
        # TODO directly compute percentage
        node_errors = node_errors / len(data_in)
        # if split attribute does not exist then is a leaf 
        if split_attribute is not None and node.get_level() < self.max_depth:
            child_errors = self.compute_split_error(data_in[[split_attribute, 'target']], local_threshold)
            # compute percentage
            # TODO directly compute percentage
            child_errors = child_errors / len(data_in)
            # if child errors are greater the actual error of the node than the split is useless
            if child_errors >= node_errors:
                # the node (default type "DecisionNode") is "transformed" in a leaf node ("LeafNode" type)
                parent_node = node.get_parent_node()
                if parent_node is None and node.get_label() == 'root':
                    print("Childs error percentage is higher than the root one. Can't find a suitable split of the root node.")
                elif parent_node is None:
                    raise Exception("Can't transform DecisionNode {} in LeafNode: no parent found".format(node.get_label()))
                else:
                #breakpoint()
                    self.delete_node(node)
                    node = LeafNode(dict(data_in.groupby('target')['weight'].sum().round(4)), node.get_label(), node.get_level())
                    self.add_node(node, parent_node)
            else:
                # if the attribute with the greatest gain is continuous than the split is binary
                if self._attributes_map[split_attribute] == 'continuous':
                    # compute global threshold (on the complete dataset) from local one
                    threshold = get_total_threshold(data_total[split_attribute], local_threshold)
                    node.set_attribute('{}:{}'.format(split_attribute, threshold), 'continuous')
                    # create DecisionNode, recursion and add node
                    # Low split
                    low_split_node = DecisionNode('{} <= {}'.format(split_attribute, float(threshold)), node.get_level())
                    self.add_node(low_split_node, node)
                    # the split is computed on the known data and then weighted on unknown ones
                    data_known = data_in[data_in[split_attribute] != '?']
                    data_unknown = data_in[data_in[split_attribute] == '?']
                    weight_unknown = len(data_known[data_known[split_attribute] <= threshold]) / len(data_known)
                    new_weight = (np.array([weight_unknown] * len(data_unknown)) * np.array(data_unknown['weight'].copy(deep=True))).tolist()
                    new_data_unknown = data_unknown.copy(deep=True)
                    new_data_unknown.loc[:, ['weight']] = new_weight
                    # concat the unknown data to the known ones, weighted, and pass to the next split
                    new_data_low = pd.concat([data_known[data_known[split_attribute] <= threshold], new_data_unknown], ignore_index=True)
                    self.split_node(low_split_node, new_data_low, data_total)
                    # High split
                    high_split_node = DecisionNode('{} > {}'.format(split_attribute, float(threshold)), node.get_level())
                    self.add_node(high_split_node, node)
                    weight_unknown = len(data_known[data_known[split_attribute] > threshold]) / len(data_known)
                    new_weight = (np.array([weight_unknown] * len(data_unknown)) * np.array(data_unknown['weight'].copy(deep=True))).tolist()
                    new_data_unknown = data_unknown.copy(deep=True)
                    new_data_unknown.loc[:, ['weight']] = new_weight
                    new_data_high = pd.concat([data_known[data_known[split_attribute] > threshold], new_data_unknown], ignore_index=True)
                    self.split_node(high_split_node, new_data_high, data_total)
                else:
                    # if the attribute is categorical or boolean than there is a node for every possible attribute value
                    node.set_attribute(split_attribute, self._attributes_map[split_attribute])
                    data_known = data_in[data_in[split_attribute] != '?']
                    data_unknown = data_in[data_in[split_attribute] == '?']
                    for attr_value in data_known[split_attribute].unique():
                        # create DecisionNode, recursion and add node
                        child_node = DecisionNode('{} = {}'.format(split_attribute, attr_value), node.get_level())
                        self.add_node(child_node, node)
                        # the split is computed on the known data and then weighted on unknown ones
                        weight_unknown = len(data_known[data_known[split_attribute] == attr_value]) / len(data_known)
                        new_weight = (np.array([weight_unknown] * len(data_unknown)) * np.array(data_unknown['weight'].copy(deep=True))).tolist()
                        new_data_unknown = data_unknown.copy(deep=True)
                        new_data_unknown.loc[:, ['weight']] = new_weight
                        # concat the unknown data to the known ones, weighted, and pass to the next split
                        new_data = pd.concat([data_known[data_known[split_attribute] == attr_value], new_data_unknown], ignore_index=True)
                        self.split_node(child_node, new_data, data_total)
        else:
            # the node (default type "DecisionNode") is "transformed" in a leaf node ("LeafNode" type)
            parent_node = node.get_parent_node()
            if parent_node is None and node.get_label() == 'root':
                print("The data are not feasible for fitting a tree. Can't find a suitable split of the root node.")
            elif parent_node is None:
                raise Exception("Can't transform DecisionNode {} in LeafNode: no parent found".format(node.get_label()))
            else:
                self.delete_node(node)
                # the final number of class contained is the sum of the weights of every row with that specific target
                node = LeafNode(dict(data_in.groupby('target')['weight'].sum().round(4)), node.get_label(), node.get_level())
                self.add_node(node, parent_node)

    def compute_split_error(self, data_in, threshold) -> int:
        """ Computes the error made by the split if predicting the most frequent class for every child born after it """
        attr_name = [column for column in data_in.columns if column != 'target'][0]
        attr_type = self._attributes_map[attr_name]
        # if continuous type the split is binary given by th threshold
        if attr_type == 'continuous':
            data_in_unknown = data_in[data_in[attr_name] != '?'].copy()
            data_in_unknown.loc[:, attr_name] = data_in_unknown[attr_name].astype(float).copy()
            #breakpoint()
            split_left = data_in_unknown[data_in_unknown[attr_name] <= threshold].copy()
            # pandas function to count the occurnces of the different value of target
            values_count = split_left['target'].value_counts()
            # errors given by the difference between the sum of all occurrences and the most frequent
            errors_left = values_count.sum() - values_count.max()
            split_right = data_in_unknown[data_in_unknown[attr_name] > threshold].copy()
            values_count = split_right['target'].value_counts()
            errors_right = values_count.sum() - values_count.max()
            total_child_error = errors_left + errors_right
        # if categorical or boolean, there is a child for every possible attribute value
        else:
            total_child_error = 0
            for attr_value in data_in[attr_name].unique():
                split = data_in[data_in[attr_name] == attr_value].copy()
                values_count = split['target'].value_counts()
                total_child_error += values_count.sum() - values_count.max()
        return total_child_error

    def fit(self, data_in) -> None:
        """ Fits the tree on "data_in" """
        root_node = DecisionNode('root', 0)
        self.add_node(root_node, None)
        # add weight to dataset in order to handle unknown values
        data_in['weight'] = [1] * len(data_in)
        data_in = data_in.fillna('?')
        self.split_node(self._root_node, data_in, data_in)

    def get_leaves_nodes(self):
        """ Returns a list of the leaves nodes """
        return [node for node in self._nodes if isinstance(node, LeafNode)]

    def get_nodes(self):
        return self._nodes

    def extract_rules(self) -> dict:
        """ Extracts the rules from the tree, one for each target transition.

        For each leaf node, puts in conjunction all the conditions in the path from the root to the leaf node.
        Then, for each target class, put the conjunctive rules in disjunction.
        """

        rules = dict()
        leaf_nodes = self.get_leaves_nodes()
        for leaf_node in leaf_nodes:
            vertical_rules = extract_rules_from_leaf(leaf_node)

            vertical_rules = ' && '.join(vertical_rules)

            if leaf_node._label_class not in rules.keys():
                rules[leaf_node._label_class] = set()
            rules[leaf_node._label_class].add(vertical_rules)

        for target_class in rules.keys():
            rules[target_class] = ' || '.join(rules[target_class])

        return rules

    def extract_rules_with_pruning(self, data_in) -> dict:
        """ Extracts the rules from the tree, one for each target transition.

        For each leaf node, takes the list of conditions from the root to the leaf, simplifies it if possible, and puts
        them in conjunction, adding the resulting rule to the dictionary at the corresponding target class key.
        Finally, all the rules related to different leaves with the same target class are put in disjunction.
        """

        # Starting with a p_value threshold for the Fisher's Exact Test (for rule pruning) of 0.01, create the rules
        # dictionary. If for some target all the rules have been pruned, repeat the process increasing the threshold.
        p_threshold = 0.01
        keep_rule = set()
        while p_threshold <= 1.0:
            rules = dict()

            leaves = self.get_leaves_nodes()
            inputs = [(leaf, keep_rule, p_threshold, data_in) for leaf in leaves]
            print("Starting multiprocessing rules pruning on {} leaves...".format(str(len(leaves))))
            with multiprocessing.Pool() as pool:
                result = list(tqdm(pool.imap(self._simplify_rule_multiprocess, inputs), total=len(leaves)))

            for (vertical_rules, leaf_class) in result:
                # Create the set corresponding to the target class in the rules dictionary, if not already present
                if leaf_class not in rules.keys():
                    rules[leaf_class] = set()

                # If the resulting list is composed by at least one rule, put them in conjunction and add the result to
                # the dictionary of rules, at the corresponding class label
                if len(vertical_rules) > 0:
                    vertical_rules = " && ".join(vertical_rules)
                    rules[leaf_class].add(vertical_rules)

            # Put the rules for the same target class in disjunction. If there are no rules for some target class (they
            # have been pruned) then set the 'empty_rule' variable to True.
            empty_rule = False
            for target_class in rules.keys():
                if len(rules[target_class]) == 0:
                    empty_rule = True
                    break
                else:
                    rules[target_class] = " || ".join(rules[target_class])

            # If 'empty_rule' is True, then increase the threshold and repeat the process. Otherwise, if two target
            # transitions have the same rule (because the original vertical rule has been pruned "too much"), repeat the
            # process but avoid simplifying the rule that originated the problem. This is done only if the 'new' rules
            # to be avoided are not all already present in 'keep_rule', otherwise it means that the process is looping.
            # If that happens, simply increase the threshold and repeat the process.
            # Otherwise, return the dictionary.
            if empty_rule:
                # TODO maybe increase more each time? This is precise but it may take long since the cap is 1
                keep_rule = set()
                p_threshold = round(p_threshold + 0.01, 2)
            elif len(rules.values()) != len(set(rules.values())):
                rules_to_add = [r for r in set(rules.values()) if list(rules.values()).count(r) > 1]
                if not all([r in keep_rule for r in rules_to_add]):
                    keep_rule.update(rules_to_add)
                else:
                    keep_rule = set()
                    p_threshold = round(p_threshold + 0.01, 2)
            else:
                break

        return rules

    def _simplify_rule_multiprocess(self, input):
        leaf_node, kr, p_threshold, data_in = input
        vertical_rules = extract_rules_from_leaf(leaf_node)

        # Simplify the list of rules, if possible (and if vertical_rules does not contain rules in keep_rule)
        if not any([r in vertical_rules for r in kr]):
            vertical_rules = self._simplify_rule(vertical_rules, leaf_node._label_class, p_threshold, data_in)

        return vertical_rules, leaf_node._label_class

    def _simplify_rule(self, vertical_rules, leaf_class, p_threshold, data_in) -> list:
        """ Simplifies the list of rules from the root to a leaf node.

        Given the list of vertical rules for a leaf, i.e. the list of rules from the root to the leaf node,
        drops the irrelevant rules recursively applying a Fisher's Exact Test and returns the remaining ones.
        In principle, all the rules could be removed: in that case, the result would be an empty list.
        Method taken from "Simplifying Decision Trees" by J.R. Quinlan (1986).
        """

        rules_candidates_remove = list()
        # For every rule in vertical_rules, check if it could be removed from vertical_rules.
        # This is true if the related p-value returned by the Fisher's Exact Test is higher than the threshold.
        # Indeed, a rule is considered relevant for the classification only if the null hypothesis (i.e. the two
        # variables - table rows and columns - are independent) can be rejected at the threshold*100% level or better.
        for rule in vertical_rules:
            other_rules = vertical_rules[:]
            other_rules.remove(rule)
            table = self._create_fisher_table(rule, other_rules, leaf_class, data_in)
            (_, p_value) = stats.fisher_exact(table)
            if p_value > p_threshold:
                rules_candidates_remove.append((rule, p_value))

        # Among the candidates rules, remove the one with the highest p-value (the most irrelevant)
        if len(rules_candidates_remove) > 0:
            rule_to_remove = max(rules_candidates_remove, key=itemgetter(1))[0]
            vertical_rules.remove(rule_to_remove)
            # Then, recurse the process on the remaining rules
            self._simplify_rule(vertical_rules, leaf_class, p_threshold, data_in)

        return vertical_rules

    def _create_fisher_table(self, rule, other_rules, leaf_class, data_in) -> pd.DataFrame:
        """ Creates a 2x2 table to be used for the Fisher's Exact Test.

        Given a rule from the list of rules from the root to the leaf node, the other rules from that list, the leaf
        class and the training set, creates a 2x2 table containing the number of training examples that satisfy the
        other rules divided according to the satisfaction of the excluded rule and the belonging to target class.
        Missing values are not taken into account.
        """

        # Create a query string with all the rules in "other_rules" in conjunction (if there are other rules)
        # Get the examples in the training set that satisfy all the rules in other_rules in conjunction
        if len(other_rules) > 0:
            query_other = ""
            for r in other_rules:
                r_attr, r_comp, r_value = r.split(' ')
                query_other += r_attr
                if r_comp == '=':
                    query_other += ' == '
                else:
                    query_other += ' ' + r_comp + ' '
                if data_in.dtypes[r_attr] in ['float64', 'bool']:
                    query_other += r_value
                else:
                    query_other += '"' + r_value + '"'
                if r != other_rules[-1]:
                    query_other += ' & '
            examples_satisfy_other = data_in.query(query_other)
        else:
            examples_satisfy_other = data_in.copy()

        # Create a query with the excluded rule
        rule_attr, rule_comp, rule_value = rule.split(' ')
        query_rule = rule_attr
        if rule_comp == '=':
            query_rule += ' == '
        else:
            query_rule += ' ' + rule_comp + ' '
        if data_in.dtypes[rule_attr] in ['float64', 'bool']:
            query_rule += rule_value
        else:
            query_rule += '"' + rule_value + '"'

        # Get the examples in the training set that satisfy the excluded rule
        examples_satisfy_other_and_rule = examples_satisfy_other.query(query_rule)

        # Get the examples in the training set that satisfy the other_rules in conjunction but not the excluded rule
        examples_satisfy_other_but_not_rule = examples_satisfy_other[
            ~examples_satisfy_other.apply(tuple, 1).isin(examples_satisfy_other_and_rule.apply(tuple, 1))]

        # Create the table which contains, for every target class and the satisfaction of the excluded rule,
        # the corresponding number of examples in the training set
        table = {k1: {k2: 0 for k2 in [leaf_class, 'not '+leaf_class]} for k1 in ['satisfies rule', 'does not satisfy rule']}

        count_other_and_rule = examples_satisfy_other_and_rule.groupby('target').count().iloc[:, 0]
        count_other_but_not_rule = examples_satisfy_other_but_not_rule.groupby('target').count().iloc[:, 0]

        for idx, value in count_other_and_rule.items():
            if idx == leaf_class:
                table['satisfies rule'][leaf_class] = value
            else:
                table['satisfies rule']['not '+leaf_class] = value
        for idx, value in count_other_but_not_rule.items():
            if idx == leaf_class:
                table['does not satisfy rule'][leaf_class] = value
            else:
                table['does not satisfy rule']['not '+leaf_class] = value

        table_df = pd.DataFrame.from_dict(table, orient='index')
        return table_df
