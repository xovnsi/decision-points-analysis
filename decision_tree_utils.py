import numpy as np

def extract_rule(node, rule):
    """ Recursively adds rule for every node visited until the root is found """
    if node.get_label() == 'root':
        return rule
    else:
        return "{} && {}".format(node.get_label(), extract_rule(node.get_parent_node(), rule))

def get_total_threshold(data, local_threshold):
    data = data[data != '?'].astype(float)
    return data[data.le(local_threshold)].max()

def class_entropy(data):
    ops = data.groupby('target')['weight'].sum() / data['weight'].sum()
    return - np.sum(ops * np.log2(ops))

def get_split_gain(data_in, attr_type):
    attr_name = [col for col in data_in.columns if col != 'target'][0]
    split_gain = class_entropy(data_in[data_in[attr_name] != '?'][['target', 'weight']])
    split_info = 0
    local_threshold = None
    are_there_at_least_two = False
    #breakpoint()
    if attr_type in ['categorical', 'boolean']:
        data_counts = data_in[attr_name].value_counts()
        total_count = len(data_in)
        known_count = len(data_in[data_in[attr_name] != '?'])
        freq_known = known_count / total_count
        for attr_value in data_in[attr_name].unique():
            if not attr_value == '?':
                freq_attr = data_counts[attr_value] / known_count
                split_gain -= freq_attr * class_entropy(data_in[data_in[attr_name] == attr_value][['target', 'weight']])
                split_info += - freq_attr * np.log2(freq_attr)
            else:
                split_info += -(1 - freq_known) * np.log2(1 - freq_known)
        gain_ratio = (freq_known * split_gain) / split_info
        info_gain = split_gain
        # check also if at least two of the subset contain at least two cases, to avoid near-trivial splits
        len_subsets = list(data_in[attr_name].value_counts())
        are_there_at_least_two = len([True for len_subset in len_subsets if len_subset >= 2]) >= 2
    elif attr_type == 'continuous':
        freq_known = len(data_in[data_in[attr_name] != '?']) / len(data_in)
        data_in = data_in[data_in[attr_name] != '?']
        data_in_sorted = data_in[attr_name].sort_values()
        thresholds = data_in_sorted.unique()[1:] - (np.diff(data_in_sorted.unique()) / 2)
        max_gain = 0
        split_gain_max = 0
        for threshold in thresholds:
            freq_attr = data_in[data_in[attr_name] <= threshold][attr_name].count() / len(data_in)
            class_entropy_low = class_entropy(data_in[data_in[attr_name] <= threshold][['target', 'weight']])
            class_entropy_high = class_entropy(data_in[data_in[attr_name] > threshold][['target', 'weight']])
            split_gain_threshold = split_gain - freq_attr * class_entropy_low - (1 - freq_attr) * class_entropy_high 
            split_info = - freq_attr * np.log2(freq_attr) - (1 - freq_attr) * np.log2(1 - freq_attr) 
            if freq_known < 1.0:
                split_info += - (1 - freq_known) * np.log2(1 - freq_known) 
            gain_ratio_temp = (freq_known * split_gain_threshold) / split_info
            if gain_ratio_temp > max_gain:
                split_gain_max = split_gain_threshold
                local_threshold = threshold
                max_gain = gain_ratio_temp
                len_subsets = [len(data_in[data_in[attr_name] <= threshold]), 
                        len(data_in[data_in[attr_name] > threshold])]
                are_there_at_least_two = len([True for len_subset in len_subsets if len_subset >= 2]) >= 2
        gain_ratio = max_gain
        info_gain = split_gain_max
            
    return gain_ratio, info_gain, local_threshold, are_there_at_least_two
