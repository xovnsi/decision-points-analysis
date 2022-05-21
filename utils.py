import math
import os
import subprocess
from operator import itemgetter

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


def get_map_place_to_events(net, loops) -> dict:
    """ Gets a mapping of decision point and their target transitions

    Given a Petri Net in the implementation of Pm4Py library and the loops inside,
    computes the target transtion for every decision point 
    (i.e. a place with more than one out arcs). If a target is a silent transition,
    the next not silent trnasitions are taken as added targets, following rules regarding loops if present.
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
                    next_not_silent = set()
                    for silent_out_arc in silent_out_arcs:
                        loop_name = 'None'
                        is_input = False
                        is_output = False
                        for loop in loops:
                            if loop.is_vertex_in_loop(place.name):
                                next_place_silent = silent_out_arc.target
                                next_not_silent = get_next_not_silent(next_place_silent, next_not_silent, loops,
                                                                      [place.name], loop.name)
                            else:
                                next_place_silent = silent_out_arc.target
                                next_not_silent = get_next_not_silent(next_place_silent, next_not_silent, loops,
                                                                      [place.name], 'None')
                    # remove not silent transitions impossible to reach without activating a loop
                    next_not_silent_to_be_removed = set()
                    for not_silent in next_not_silent:
                        if not check_if_reachable_without_loops(loops, arc.target, not_silent, False):
                            next_not_silent_to_be_removed.add(not_silent)
                    # remove all transitions inside the loop if the place is an output place of the loop
                    for loop in loops:
                        if loop.is_vertex_output_loop(place.name):
                            next_not_silent_to_be_removed = next_not_silent_to_be_removed.union(set(loop.events))
                    places[place.name][arc.target.name] = next_not_silent.difference(next_not_silent_to_be_removed)
    return places


def get_next_not_silent(place, not_silent, loops, start_places, loop_name_start) -> list:
    """ Recursively compute the first not silent transition connected to a place

    Given a place and a list of not silent transition (i.e. without label) computes
    the next not silent transitions in order to correctly characterize the path through
    the considered place. The algorithm stops in presence of a joint-node (if not in a loop) 
    or when all of the output transitions are not silent. If at least one transition is 
    silent, the algorithm computes recursively the next not silent.
    """
    # first stop condition: joint node
    loop_name = 'None'
    if place.name == 'sink':
        return not_silent
    for loop in loops:
        if loop.is_vertex_in_loop(place.name):
            if loop_name == 'None':
                loop_name = loop.name
                is_input = loop.is_vertex_input_loop(place.name)
                is_output = loop.is_vertex_output_loop(place.name)
            else:
                loop_length = loop_name.split("_")[1]
                if int(loop_length) > int(loop.name.split("_")[1]):
                    if not is_input and not is_output:
                        loop_name = loop.name
                        is_input = loop.is_vertex_input_loop(place.name)
                        is_output = loop.is_vertex_output_loop(place.name)
    if loop_name == 'None':
        is_input = False
        is_output = False
    # TODO check it the arcs of the skip are coming from the same place
    is_input_a_skip = len(place.in_arcs) == 2 and len(
        [arc.source.name for arc in place.in_arcs if arc.source.label is None]) == 1
    if (len(place.in_arcs) > 1 and not (
            is_input or is_input_a_skip)) or place.name in start_places:  # or (is_output and loop_name == loop_name_start):
        return not_silent
    out_arcs_label = [arc.target.label for arc in place.out_arcs]
    # second stop condition: all not silent outputs
    if not None in out_arcs_label:
        not_silent = not_silent.union(set(out_arcs_label))
        return not_silent
    for out_arc in place.out_arcs:
        # add activity if not silent
        if not out_arc.target.label is None:
            not_silent.add(out_arc.target.label)
        else:
            # recursive part if is silent
            for out_arc_inn in out_arc.target.out_arcs:
                added_start_place = False
                for loop in loops:
                    if loop.is_vertex_input_loop(out_arc_inn.target.name) and int(loop.name.split("_")[1]) != int(
                            loop_length) and not out_arc_inn.target.name in start_places:
                        start_places.append(out_arc_inn.target.name)
                        added_start_place = True
                # if the place is an input to another loop (nested loops), we propagate inside the other loop too
                if added_start_place:
                    for out_arc_inner_loop in out_arc_inn.target.out_arcs:
                        if not out_arc_inner_loop.target.label is None:
                            not_silent.add(out_arc_inner_loop.target.label)
                        else:
                            for next_out_arcs in out_arc_inner_loop.target.out_arcs:
                                next_place_silent = next_out_arcs.target
                                not_silent = get_next_not_silent(next_place_silent, not_silent, loops, start_places,
                                                                 loop_name_start)
                else:
                    not_silent = get_next_not_silent(out_arc_inn.target, not_silent, loops, start_places,
                                                     loop_name_start)
    return not_silent


def get_place_from_event(places_map, event, dp_list) -> list:
    """ Returns the places that are decision points of a certain transition

    Given the dictionary mapping every decision point with its reference event(s), 
    returns the list of decision point referred to the input event
    """

    places = list()
    for place in dp_list:
        for trans in places_map[place].keys():
            if event in places_map[place][trans]:
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


def extract_rules(dt, feature_names) -> dict:
    """ Returns a readable version of a decision tree rules

    Given a sklearn decision tree object and the name of the features used, interprets
    the output text of export_text (sklearn function) in order to give the user a more
    human readable version of the rules defining the decision tree behavior. The output 
    is a dictionary using as key the target class (leaves) of the decision tree.
    """
    text_rules = export_text(dt)
    for feature_name in feature_names.keys():
        text_rules = text_rules.replace(feature_names[feature_name], feature_name)
    text_rules = text_rules.split('\n')[:-1]
    extracted_rules = dict()
    one_complete_pass = ""
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
                single_rule = text_rule.split('|--- ')[1]
                one_complete_pass = "{} & {}".format(one_complete_pass, single_rule)
                tree_level += 1
    return extracted_rules


def get_feature_names(dataset) -> dict:
    """
    Returns a dictionary containing the numerated name of the features
    """
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


def get_map_events_transitions(net) -> dict:
    """ Compute a map of event name and transitions name

    Given a Petri Net in the implementation of Pm4Py library, the function creates
    a dictionary containing for every transition the corresponding event name
    """
    # initialize
    map_events_trans = dict()
    for trans in net.transitions:
        if not trans.label is None:
            map_events_trans[trans.name] = trans.label
    return map_events_trans


def check_if_reachable_without_loops(loops, start_trans, end_trans, reachable) -> bool:
    """ Check if a transition is reachable from another one without using loops

    Given a list of Loop ojects, recursively check if a transition is reachable from another one without looping 
    and using only invisible transitions. This means that if a place is an output place for a loop, the algorithm
    chooses to follow the path exiting the loop
    """
    for out_arc in start_trans.out_arcs:
        if out_arc.target.name == "sink":
            return reachable
        else:
            is_a_loop_output = False
            loop_name = 'None'
            # check if it is an output
            for loop in loops:
                if loop.is_vertex_output_loop(out_arc.target.name):
                    is_a_loop_output = True
                    loop_name = loop.name
                    loop_selected = loop
            for out_arc_inn in out_arc.target.out_arcs:
                # if it is invisible and not output
                if out_arc_inn.target.label is None and not is_a_loop_output:
                    reachable = check_if_reachable_without_loops(loops, out_arc_inn.target, end_trans, reachable)
                # if it is invisible and the next vertex is not in the same loop
                elif out_arc_inn.target.label is None and is_a_loop_output and not out_arc_inn.target.name in loop_selected.vertex:
                    reachable = check_if_reachable_without_loops(loops, out_arc_inn.target, end_trans, reachable)
                else:
                    if out_arc_inn.target.label == end_trans:
                        reachable = True
                if reachable:
                    break
        if reachable:
            break
    return reachable


def update_places_map_dp_list_if_looping(net, dp_list, places_map, loops, event_sequence, number_of_loops,
                                         trans_events_map) -> [dict, list]:
    """ Updates the map of transitions related to a decision point in case a loop is active

    Given an event sequence, checks if there are active loops (i.e. the sequence is reproducible only
    passing two times from an input place. In case of acitve loops, in what part of the loop is located the current activity 
    and assigns to the decision points the transitions that identify if the path passed through that decision point.
    For example if a loop is:
           -   A    -           -   B    -
    dp0 - |          | - dp1 - |          | - dp2 - 
    |       - silent -           - silent -        |
    |                                              |
    ------------------- silent --------------------
    and the sequence is A - B - A, it means that the decision points to be added at the moment of the second A are:
    - from A to the start (forward): dp2
    - from the start to A (backward): dp0, dp1
    """
    transition_sequence = [trans_events_map[event] for event in event_sequence]
    loop_name = 'None'
    for loop in loops:
        loop.check_if_loop_is_active(net, transition_sequence)
        if loop.is_active():
            # update the dictionary only if the loop is actually "looping°
            # (i.e. the number of cycle is growing) -> because a decision point can be 
            # chosen only one time per cycle
            loop.count_number_of_loops(net, transition_sequence.copy())
            if number_of_loops[loop.name] < loop.number_of_loops:
                number_of_loops[loop.name] = loop.number_of_loops
                # update decision points in the forward path
                for dp in loop.dp_forward.keys():
                    if not dp in dp_list:
                        dp_list.append(dp)
                    loop_reachable = loop.dp_forward.copy()
                    for trans in places_map[dp].keys():
                        if loop.is_vertex_in_loop(trans):
                            if trans in loop.dp_forward[dp].keys():
                                if event_sequence[-1] in loop.dp_forward[dp][trans]:
                                    # update only if the decision point is connected to an invisible activity 
                                    # (i.e. related to a set of activities not just one)
                                    if isinstance(places_map[dp][trans], set):
                                        places_map[dp][trans] = places_map[dp][trans].difference(loop.events)
                                        places_map[dp][trans] = places_map[dp][trans].union(loop_reachable[dp][trans])
                # update decision points in the backward path
                for dp in loop.dp_backward.keys():
                    if not dp in dp_list:
                        dp_list.append(dp)
                    loop_reachable = loop.dp_backward.copy()
                    for trans in places_map[dp].keys():
                        if loop.is_vertex_in_loop(trans):
                            if trans in loop.dp_backward[dp].keys():
                                if event_sequence[-1] in loop.dp_backward[dp][trans]:
                                    # update only if the decision point is connected to an invisible activity 
                                    # (i.e. related to a set of activities not just one)
                                    if isinstance(places_map[dp][trans], set):
                                        places_map[dp][trans] = places_map[dp][trans].difference(loop.events)
                                        places_map[dp][trans] = places_map[dp][trans].union(loop_reachable[dp][trans])
    return places_map, dp_list


def update_dp_list(places_from_event, dp_list) -> list:
    """ Updates the list of decision points if related with the event

    Removes the decision points related to the event in input because a decision point
    can be chosen only once
    """
    for place, trans_name in places_from_event:
        if place in dp_list:
            dp_list.remove(place)
    return dp_list


def get_all_dp_from_event_to_sink(event, loops, places) -> list:
    """ Returns all the decision points from the event to the sink, passing through invisible transitions

    Given an event, recursively explore the net collecting decision points and related invisible transitions
    that need to be passed in order to reach the sink place. If a loop is encountered, the algorithm choose to 
    exit at the first output seen.
    """
    for out_arc in event.out_arcs:
        if out_arc.target.name == 'sink':
            return places
        else:
            is_output = False
            for loop in loops:
                if loop.is_vertex_output_loop(out_arc.target.name):
                    is_output = True
                    loop_selected = loop
            places_length_before = len(places)
            for out_arc_inn in out_arc.target.out_arcs:
                # if invisible transition
                if out_arc_inn.target.label is None:
                    # if is not an output of the considered loop
                    if not is_output or (is_output and not loop_selected.is_vertex_in_loop(out_arc_inn.target.name)):
                        # if it is a decision point append to the list and recurse
                        if len(out_arc.target.out_arcs) > 1:
                            places.append((out_arc.target.name, out_arc_inn.target.name))
                        places = get_all_dp_from_event_to_sink(out_arc_inn.target, loops, places)
                # if anything changes, means that the cycle stopped: remove the last tuple appended
                if len(places) == places_length_before and (out_arc.target.name, out_arc_inn.target.name) in places:
                    places.remove((out_arc.target.name, out_arc_inn.target.name))
    return places


def discovering_branching_conditions(dataset, attributes_map) -> dict:
    """ Alternative method for discovering branching conditions, using Daikon invariant detector

    It uses the existing version of Daikon, since it supports csv files (after a conversion).
    It only supports binary decision points. It returns a dictionary containing the discovered rule for each branch.
    Method taken from "Discovering Branching Conditions from Business Process Execution Logs" by Massimiliano de Leoni,
    Marlon Dumas, and Luciano Garcia-Banuelos. In particular, only the CD+IG+LV approach is implemented.
    """

    # TODO now I'm using only continuous variables, since it seems that Daikon works better with them
    # Keeping only continuous variables (and the target column) and discarding rows with missing values
    feature_names = [c for c in dataset.columns if c != 'target']
    dataset = dataset.drop(columns=[a for a in feature_names if attributes_map[a] != 'continuous']).dropna()

    # Enriching the dataset with latent variables (all combinations of variables using +, -, *, /)
    feature_names = [c for c in dataset.columns if c != 'target']
    for i, c1 in enumerate(feature_names):
        # Non-commutative operations are computed also switching the operands
        for c2 in feature_names[i+1:]:
            dataset[c1 + '_plus_' + c2] = dataset[c1] + dataset[c2]
            dataset[c1 + '_minus_' + c2] = dataset[c1] - dataset[c2]
            dataset[c2 + '_minus_' + c1] = dataset[c2] - dataset[c1]
            dataset[c1 + '_times_' + c2] = dataset[c1] * dataset[c2]
            if 0 not in dataset[c2]:    # to avoid division by zero
                dataset[c1 + '_div_by_' + c2] = dataset[c1] / dataset[c2]
            if 0 not in dataset[c1]:    # to avoid division by zero
                dataset[c2 + '_div_by_' + c1] = dataset[c2] / dataset[c1]

    # Splitting the dataset according to the target value
    gb = dataset.groupby('target')
    target_datasets = [x for _, x in gb]

    # This check is done because this approach only works with binary decision points, and we do not have only those
    if len(target_datasets) == 2:
        # Extracting the invariants using Daikon
        invariants_1 = _get_daikon_invariants(target_datasets[0])
        invariants_2 = _get_daikon_invariants(target_datasets[1])

        # Building the conjunctive expression for each branch of the decision point
        conj_expr_1 = _build_conj_expr(target_datasets[0], target_datasets[1], invariants_1)
        conj_expr_2 = _build_conj_expr(target_datasets[0], target_datasets[1], invariants_2)

        # Adjusting the conditions to ensure that one is the negation of the other
        conj_expr_1, conj_expr_2 = _adjust_conditions(target_datasets[0], target_datasets[1], conj_expr_1, conj_expr_2)

        # Rewriting the operands for the latent variables (i.e. '_plus_' becomes '+' etc.)
        conj_expr_1, conj_expr_2 = _clean_latent_variables(conj_expr_1, conj_expr_2)

        return {list(gb.groups.keys())[0]: conj_expr_1, list(gb.groups.keys())[1]: conj_expr_2}


def _get_daikon_invariants(dataset) -> list:
    """ Extracting the invariants from a set of observation instances related to a branch of a decision point

    After exporting the DataFrame as a csv file, a Perl script is launched to create the input files for Daikon in the
    proper format. Then, Daikon is called to discover the invariants. Finally, the extracted invariants are cleaned
    to be used later and returned as a list.
    """

    dataset.drop(columns=['target']).to_csv(path_or_buf='dataset.csv', index=False)
    subprocess.run(['perl', 'daikon-5.8.10/scripts/convertcsv.pl', 'dataset.csv'])
    subprocess.run(['java', '-cp', 'daikon-5.8.10/daikon.jar', 'daikon.Daikon', '--nohierarchy', '-o', 'invariants.inv',
                    '--no_text_output', '--noversion', '--omit_from_output', 'r', 'dataset.dtrace', 'dataset.decls'])
    inv = subprocess.run(['java', '-cp', 'daikon-5.8.10/daikon.jar', 'daikon.PrintInvariants',
                          'invariants.inv'], capture_output=True, text=True)
    invariants = []
    for line in inv.stdout.splitlines():
        if not any(x in line for x in ["===", "aprogram.point:::POINT", "one of"]):
            invariants.append(line)

    for file_name in ['dataset.csv', 'dataset.dtrace', 'dataset.decls', 'invariants.inv']:
        try:
            os.remove(file_name)
        except FileNotFoundError:
            continue

    return invariants


def _build_conj_expr(set_1, set_2, invariants) -> str | None:
    """ Builds a conjunctive expression starting from the invariants found.

    The resulting conjunctive expression is built using a greedy approach. The first atom selected is the one with the
    highest information gain. Then, iteratively, the atom with the highest information gain is added, provided that the
    resulting conjunctive expression, adding that atom, increases the information gain.
    The resulting conjunctive expression is then returned as a string.
    """

    if len(invariants) == 0:
        return None
    else:
        atom_information_gains = []
        for inv in invariants:
            atom_information_gains.append((inv, _compute_information_gain(set_1, set_2, [inv])))

        element_with_max_gain = max(atom_information_gains, key=itemgetter(1))
        resulting_expr = [element_with_max_gain[0]]
        atom_information_gains.remove(element_with_max_gain)

        while len(atom_information_gains) > 0:
            element_with_max_gain = max(atom_information_gains, key=itemgetter(1))
            new_predicate = resulting_expr + [element_with_max_gain[0]]
            if _compute_information_gain(set_1, set_2, new_predicate) > _compute_information_gain(set_1, set_2, resulting_expr):
                resulting_expr.append(element_with_max_gain[0])
            atom_information_gains.remove(element_with_max_gain)

        return ' && '.join(resulting_expr)


def _compute_entropy(set_1, set_2) -> float:
    """ Computes the entropy of the two sets of observation instances """

    size_1 = len(set_1)
    size_2 = len(set_2)
    if size_1 == 0 or size_2 == 0:
        return 0
    else:
        fraction_1 = size_1 / (size_1 + size_2)
        fraction_2 = size_2 / (size_1 + size_2)
        return - (fraction_1 * math.log2(fraction_1)) - (fraction_2 * math.log2(fraction_2))


def _compute_information_gain(set_1, set_2, predicate) -> float:
    """ Computes the information gain of predicate given the two sets of observation instances. """

    predicate_1, predicate_2 = predicate.copy(), predicate.copy()

    predicate_1 = ' & '.join(predicate_1)
    predicate_2 = ' & '.join(predicate_2)

    size_1 = len(set_1)
    size_2 = len(set_2)
    set_1_pred = set_1.query(predicate_1)
    size_1_pred = len(set_1_pred)
    set_1_not_pred = set_1[~set_1.apply(tuple, 1).isin(set_1_pred.apply(tuple, 1))]
    size_1_not_pred = len(set_1_not_pred)
    set_2_pred = set_2.query(predicate_2)
    size_2_pred = len(set_2_pred)
    set_2_not_pred = set_2[~set_2.apply(tuple, 1).isin(set_2_pred.apply(tuple, 1))]
    size_2_not_pred = len(set_2_not_pred)

    fraction_1 = ((size_1_pred + size_2_pred) * _compute_entropy(set_1_pred, set_2_pred)) / (size_1 + size_2)
    fraction_2 = ((size_1_not_pred + size_2_not_pred) * _compute_entropy(set_1_not_pred, set_2_not_pred)) / (
            size_1 + size_2)

    return _compute_entropy(set_1, set_2) - fraction_1 - fraction_2


def _adjust_conditions(set_1, set_2, conj_expr_1, conj_expr_2) -> (str, str):
    """ Adjusts the two conjunctive expression so that one is the negation of the other.

    If one of them is empty, then it is set to the negation of the other. If they are both non-empty, then the one with
    the lower information gain is set to the negation of the other.
    """

    if conj_expr_1 is None and conj_expr_2 is not None:
        conj_expr_1 = _negate_expr(conj_expr_2)
    elif conj_expr_1 is not None and conj_expr_2 is None:
        conj_expr_2 = _negate_expr(conj_expr_1)
    elif conj_expr_1 is None and conj_expr_2 is None:
        return 'None', 'None'
    else:
        expr_1_list = conj_expr_1.split(' && ')
        expr_2_list = conj_expr_2.split(' && ')
        info_gain_expr_1 = _compute_information_gain(set_1, set_2, expr_1_list)
        info_gain_expr_2 = _compute_information_gain(set_1, set_2, expr_2_list)

        if info_gain_expr_1 > info_gain_expr_2:
            conj_expr_2 = _negate_expr(conj_expr_1)
        else:
            conj_expr_1 = _negate_expr(conj_expr_2)

    return conj_expr_1, conj_expr_2


def _negate_expr(expr) -> str:
    """ Returns the negation of an expression.

    If the expression contains multiple atoms in conjunction, it places 'not' before the original expression.
    Otherwise, it negates the operand of the expression.
    """

    if ' && ' in expr:
        return 'not (' + expr + ')'
    else:
        ops = [' == ', ' != ', ' > ',  ' < ', ' >= ', ' <= ']
        neg_ops = [' != ', ' == ', ' <= ', ' >= ', ' < ', ' > ']

        for i, o in enumerate(ops):
            if o in expr:
                split = expr.split(o)
                return neg_ops[i].join(split)


def _clean_latent_variables(expr1, expr2) -> (str, str):
    """ Rewrites the dummy names for latent variables using the corresponding operands. """

    expr_list = [expr1, expr2]
    ops = ['_plus_', '_minus_', '_times_', '_div_by_']
    ops_symb = [' + ', ' - ', ' * ', ' / ']
    for i, e in enumerate(expr_list):
        for j, o in enumerate(ops):
            if o in e:
                e = e.replace(o, ops_symb[j])
        expr_list[i] = e

    return expr_list
