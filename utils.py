import numpy as np
import pandas as pd
from sklearn.tree import export_text

def print_matrix(matrix, row_names, col_names):
    row_cols_name = "   "
    for col_name in col_names:
        row_cols_name = "{}     {}".format(row_cols_name, col_name)
    print(row_cols_name)
    for i, row_name in enumerate(row_names):
        row = row_name
        for j, row_col in enumerate(col_names):
            if j == 0:
                row = "{}   {}".format(row, matrix[i, j])
            else:
                row = "{}     {}".format(row, matrix[i, j])
        print(row)

def get_map_place_to_events(net, vertex_in_loops) -> dict:
    """ Gets a mapping of decision point and their target transitions

    Given a Petri Net in the implementation of Pm4Py library and elements belonging
    to whatever loop in it, computes the target transtion for every decision point 
    (i.e. a place with more than one out arcs). If a target is a silent transition,
    the next not silent trnasitions are taken as added targets.
    """
    # initialize
    places = dict()
    for place in net.places:
        if len(place.out_arcs) >= 2:
            # dictionary containing for every decision point target categories 
            places[place.name] = dict()
            # loop for out arcs
            for arc in place.out_arcs:
                # check if silent
                if not arc.target.label is None:
                    places[place.name][arc.target.name] = arc.target.label
                else:
                    # search for next not silent from the next places
                    silent_out_arcs = arc.target.out_arcs
                    next_not_silent = []
                    for silent_out_arc in silent_out_arcs:
                        next_place_silent = silent_out_arc.target
                        next_not_silent = get_next_not_silent(next_place_silent, next_not_silent, vertex_in_loops)
                    places[place.name][arc.target.name] = next_not_silent
    return places

def get_out_place_loop(net, vertex_in_loops) -> list:
    """ Gets the name of places that are output of a loop

    Given a net and places and transitions belonging to whatever loop in it,
    computes places that are output in one of them. A place is an output if at least one output
    arc is coming from a transition outside the loop.
    """
    # initialize
    out_places = list()
    for place in net.places:
        if place.name in vertex_in_loops:
            out_trans = set([arc.target.name for arc in place.out_arcs])
            # check if all the output transitions are contained in a loop
            if not out_trans.issubset(vertex_in_loops):
                out_places.append(place.name)
    return out_places

def get_in_place_loop(net, vertex_in_loops) -> list:
    """ Gets the name of places that are input of a loop

    Given a net and places and transitions belonging to whatever loop in it,
    computes places that are input in one of them. A place is an input if at least one input
    arc is coming from a transition outside the loop.
    """
    # initialize
    in_places = list()
    for place in net.places:
        if place.name in vertex_in_loops:
            in_trans = set([arc.source.name for arc in place.in_arcs])
            # check if all the input transitions are contained in a loop
            if not in_trans.issubset(vertex_in_loops):
                in_places.append(place.name)
    return in_places

def get_place_from_transition(places_map, transition, vertex_in_loops, last_event, in_transitions_loops, out_places_loops, in_places_loops, trans_from_event) -> list:
    """ Computes the places that decision points of a certain transition

    Given a lot of input, for every decision point checks different conditions in order to decide if it has to be added to the list of 
    decision points for that transition. The place is considered a valid candidate if 
    I) the transition considered and the previous are both in a loop and the place belong to a loop
    II) the transition considered and the previous are both not in a loop
    III) the transition considered is in loop, the last is not and the place is an input place of the loop
    IV) the last transition is in the loop and the place is an output place of the loop
    V) one transition and the place are outside the loop (deals with invisible activities before the loop)
    """
    # TODO simplify conditions and input arguments
    is_transition_in_loop = trans_from_event in vertex_in_loops and trans_from_event in in_transitions_loops
    places = list() 
    for place in places_map.keys():
        is_out_place = place in out_places_loops
        is_last_event_in_loop = last_event in vertex_in_loops
        are_both_in_loop = is_last_event_in_loop and trans_from_event in vertex_in_loops
        are_both_not_in_loop = not is_last_event_in_loop and not trans_from_event in vertex_in_loops
        for trans in places_map[place].keys():
            if transition in places_map[place][trans] and ((are_both_in_loop and place in vertex_in_loops) or (not are_both_in_loop and place in in_places_loops) or are_both_not_in_loop or (is_last_event_in_loop and is_out_place) or (not are_both_in_loop and not place in vertex_in_loops)):
                places.append((place, trans))
    return places

def get_attributes_from_event(event) -> dict:
    """ Return the attributes of an event

    Given an attribute of an event in an event log, returns a dictionary
    containing the attribute of an event, numeric if it is possible
    """
    # TODO specify what are numeric/categorical attributes in order not to try every time (also maybe a code is not numeric)
    # intialize
    attributes = dict()
    for attribute in event.keys():
        if not isinstance(event[attribute], bool):
            try:
                attributes[attribute] = [float(event[attribute])]
            except:
                attributes[attribute] = [event[attribute]]
        else:
            attributes[attribute] = [event[attribute]]
    return attributes

def get_next_not_silent(place, not_silent, vertex_in_loops) -> list:
    """ Recursively compute the first not silent transition connected to a place

    Given a place and a list of not silent transition (i.e. without label) computes
    the next not silent transitions in order to correctly characterize the path through
    the considered place. The algorithm stops in presence of a joint-node (if not in a loop) 
    or when all of the output transitions are not silent. If at least one transition is 
    silent, the algorithm computes recursively the next not silent.
    """
    # first stop condition
    if len(place.in_arcs) > 1 and not place.name in vertex_in_loops:
        return not_silent
    out_arcs_label = [arc.target.label for arc in place.out_arcs]
    # second stop condition
    if not None in out_arcs_label:
        not_silent.extend(out_arcs_label)
        return not_silent
    for out_arc in place.out_arcs:
        # add not silent
        if not out_arc.target.label is None:
            not_silent.extend(out_arc.target.label)
        else:
            # recursive part if is silent
            for out_arc_inn in out_arc.target.out_arcs:
                not_silent = get_next_not_silent(out_arc_inn.target, not_silent, vertex_in_loops)
    return not_silent

def extract_rules(dt, feature_names) -> dict:
    """ Returns a readable version of a decision tree rules

    Given a sklearn decision tree object and the name of the features used, interprets
    the output text of export_text (sklearn function) in order to give the user a more
    human readable version of the rules defining the decision tree behavior. The output 
    is a dictionary using as key the target class (leaves) of the decision tree.
    """
    #breakpoint()
    text_rules = export_text(dt)
    for feature_name in feature_names.keys():
            text_rules = text_rules.replace(feature_names[feature_name], feature_name) 
    text_rules = text_rules.split('\n')[:-1]
    extracted_rules = dict()
    one_complete_pass = ""
    #breakpoint()
    tree_level = 0
    for text_rule in text_rules:
        single_rule = text_rule.split('|')[1:]
        if '---' in single_rule[0]:
            one_complete_pass = single_rule[0].split('--- ')[1]
        else:
            if 'class' in text_rule:
                label_name = text_rule.split(': ')[-1]
                if label_name in extracted_rules.keys():
                    extracted_rules[label_name].append(one_complete_pass)
                else:
                    extracted_rules[label_name] = list()
                    extracted_rules[label_name].append(one_complete_pass)
                reset_level_rule = one_complete_pass.split(' & ')
                if len(reset_level_rule) > 1:
                    one_complete_pass = reset_level_rule[:-1][0]
            else:
                #breakpoint()
                single_rule = text_rule.split('|--- ')[1]
                one_complete_pass = "{} & {}".format(one_complete_pass, single_rule)
                tree_level += 1
    return extracted_rules

def get_feature_names(dataset):
    if not isinstance(dataset, pd.DataFrame):
        raise Exception("Not a dataset object")
    features = dict()
    for index, feature in enumerate(dataset.drop(columns=['target']).columns):
        if not feature == 'target':
            features[feature] = "feature_{}".format(index)
    return features


def get_map_transitions_events(net) -> dict:
    """ Compute a map of transitions name and events

    Given a Petri Net in the implementation of Pm4Py library, the function creates
    a dictionary containing for every event the corresponding transition name
    """
    # initialize
    map_trans_events = dict()
    for trans in net.transitions:
        map_trans_events[trans.label] = trans.name
    map_trans_events['None'] = 'None'
    return map_trans_events

def get_input_adjacency_matrix(places_list, transitions_list) -> np.ndarray:
    """ Computes the input adjacency matrix

    Given a list of places and transitions, computes the input adjacency matrix
    which is the transposition of the third quadrant of the total adjacency matrix.
    The name input is related to the transitions, from a place to a transition.
    """
    # initialize
    adj = np.zeros((len(transitions_list), len(places_list)))
    for i, trans in enumerate(transitions_list):
        for j, place in enumerate(places_list):
            # get the name of all the arcs exiting from the place
            in_arcs_name = [arc.target.name for arc in place.out_arcs]
            if trans.name in in_arcs_name:
                adj[i, j] = 1
    return adj

def get_output_adjacency_matrix(places_list, transitions_list):
    """ Computes the output adjacency matrix

    Given a list of places and transitions, computes the output adjacency matrix 
    which is the second quadrant of the total adjacency matrix.
    The name output is related to the transitions, from a transition to a place.
    """
    # initialize
    adj = np.zeros((len(transitions_list), len(places_list)))
    for i, trans in enumerate(transitions_list):
        for j, place in enumerate(places_list):
            # get the name of all the arcs exiting from the transition
            out_arcs_name = [arc.source.name for arc in place.in_arcs]
            if trans.name in out_arcs_name:
                adj[i, j] = 1
    return adj

def get_vertex_names_from_c_matrix(c_matrix, row_names, columns_names):
    vertex_in_loop = set()
    for i, row in enumerate(row_names):
        for j, col in enumerate(columns_names):
            if c_matrix[i, j] == True:
                #breakpoint()
                vertex_in_loop.add(row)
                vertex_in_loop.add(col)
    #breakpoint()
    return vertex_in_loop

def detect_loops(net) -> set:
    """ Detects loops in a Petri Net

    Given a Petri Net in the implementation of Pm4Py library, the code look for every 
    place/transition belonging to a loop of whatever length from the minimum to the maximum. 
    The algorithm used for loop detection is taken from - Davidrajuh: `Detecting Existence of
    cycle in Petri Nets'. Advances in Intelligent Systems and Computing (2017)
    """
    # get list of transitions and places names in the net
    places_names = [place.name for place in list(net.places)]
    trans_names = [tran.name for tran in list(net.transitions)]
    # compute input and output adjacency matrix (input and output considering transitions)
    in_adj = get_input_adjacency_matrix(list(net.places), list(net.transitions))
    out_adj = get_output_adjacency_matrix(list(net.places), list(net.transitions))
    # initialize adjacency matrix quadrants, cycle counter and output set
    old_quadrant_1 = np.zeros((len(trans_names), len(trans_names)))
    old_quadrant_2 = out_adj
    old_quadrant_3 = in_adj.T
    old_quadrant_4 = np.zeros((len(places_names), len(places_names)))
    cont_r = 1
    places_trans_in_loop = set()
    # Every iteration detect loop of length 2*cont_r, due to the bipartite nature of the net
    while (cont_r <= len(places_names) + len(trans_names)):
        cont_r += 1
        # initialize new A and C quadrants
        new_quadrant_1 = np.zeros((len(trans_names), len(trans_names)))
        new_quadrant_2 = np.zeros((len(trans_names), len(places_names)))
        new_quadrant_3 = np.zeros((len(places_names), len(trans_names)))
        new_quadrant_4 = np.zeros((len(places_names), len(places_names)))
        c_quadrant_1 = np.zeros((len(trans_names), len(trans_names)), dtype=bool)
        c_quadrant_2 = np.zeros((len(trans_names), len(places_names)), dtype=bool)
        c_quadrant_3 = np.zeros((len(places_names), len(trans_names)), dtype=bool)
        c_quadrant_4 = np.zeros((len(places_names), len(places_names)), dtype=bool)
        # compute new A and C quadrants
        if cont_r % 2 == 1:
            new_quadrant_2 = np.dot(out_adj, old_quadrant_4)
            new_quadrant_3 = np.dot(in_adj.T, old_quadrant_1)
            new_quadrant_2T = new_quadrant_2.T
            new_quadrant_3T = new_quadrant_3.T
            c_quadrant_2 = np.logical_and(new_quadrant_2, new_quadrant_3T)
            c_quadrant_3 = np.logical_and(new_quadrant_3, new_quadrant_2T)
            vert_loop = get_vertex_names_from_c_matrix(c_quadrant_2, trans_names, places_names)
            places_trans_in_loop = places_trans_in_loop.union(vert_loop)
            vert_loop = get_vertex_names_from_c_matrix(c_quadrant_3, places_names, trans_names)
            places_trans_in_loop = places_trans_in_loop.union(vert_loop)
        else:
            new_quadrant_1 = np.dot(out_adj, old_quadrant_3)
            new_quadrant_4 = np.dot(in_adj.T, old_quadrant_2)
            new_quadrant_1T = new_quadrant_1.T
            new_quadrant_4T = new_quadrant_4.T
            c_quadrant_1 = np.logical_and(new_quadrant_1, new_quadrant_1T)
            c_quadrant_4 = np.logical_and(new_quadrant_4, new_quadrant_4T)
            vert_loop = get_vertex_names_from_c_matrix(c_quadrant_1, trans_names, trans_names)
            places_trans_in_loop = places_trans_in_loop.union(vert_loop)
            vert_loop = get_vertex_names_from_c_matrix(c_quadrant_4, places_names, places_names)
            places_trans_in_loop = places_trans_in_loop.union(vert_loop)
        # update old quadrants
        old_quadrant_1 = new_quadrant_1
        old_quadrant_2 = new_quadrant_2
        old_quadrant_3 = new_quadrant_3
        old_quadrant_4 = new_quadrant_4

    return places_trans_in_loop
