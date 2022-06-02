import numpy as np
import pandas as pd
from sklearn.tree import export_text
from pm4py.objects.petri_net.obj import PetriNet

from DecisionTree import DecisionTree
from decision_tree_utils import extract_rules_from_leaf


def are_activities_parallel(first_activity, second_activity, parallel_branches) -> bool:
    """ Check if two activities are parallel in the net. 

    The dictionary 'parallel_branches' is derived from the net simplification algorithm."""
    are_parallel = False
    for split in parallel_branches.keys():
        for branch in parallel_branches[split].keys():
            if first_activity in parallel_branches[split][branch] and not second_activity in parallel_branches[split][branch]:
                for other_branch in parallel_branches[split].keys():
                    if not other_branch == branch:
                        if second_activity in parallel_branches[split][other_branch]:
                            are_parallel = True
    return are_parallel

def get_decision_points_and_targets(sequence, loops, net, parallel_branches) -> dict: 
    """ Returns a dictionsy containing decision points and their targets.

    Starting from the last activity in the sequence, the algorithm selects the previous not 
    parallel activity. Exploring the net backward, it returns all the decision points with their targets
    encountered on the path leading to the last activity in the sequence.
    """
    # search the first not parallel activity before the last in the sequence
    #breakpoint()
    current_act_name = sequence[-1]
    previous_sequence = sequence[:-1]
    previous_sequence.reverse()
    previous_act_name = None
    for previous in previous_sequence:
        # problems if the loop has some parts that cannot be simplified
        # TODO change the simplifying algo to be sure that all the parallel parts are recognized
        parallel = are_activities_parallel(current_act_name, previous, parallel_branches)
        if not parallel:
            previous_act_name = previous
            break
    if previous_act_name is None:
        raise Exception("Can't find the previous not parallel activity")
    # keep the transition objects
    #breakpoint()
    current_act = [trans for trans in net.transitions if trans.name == current_act_name][0]
    previous_act = [trans for trans in net.transitions if trans.name == previous_act_name][0]
    loops_selected = list()
    reachability = dict()
    # check if the two activities are contained in the same loop and save it if exists
    for loop in loops:
        if loop.is_node_in_loop_complete_net(current_act.name) and loop.is_node_in_loop_complete_net(previous):
            loops_selected.append(loop)
            # check if the last activity is reacahble from the previous one (if not it means that the loop is active) 
            reachability[loop.name] = loop.check_if_reachable(previous_act, current_act.name, False)
    dp_dict = dict()
    dp_dict, _, _ = get_dp_to_previous_event(previous_act_name, current_act, loops, loops_selected, dp_dict, reachability, dict())

    return dp_dict

def get_dp_to_previous_event(previous, current, loops, common_loops, decision_points, reachability, passed_inv_act) -> [dict, bool, bool]:
    """ Recursively explores the net and saves the decision points encountered with the targets.
    
    The backward recursion is allowed only if there is an invisbile activity and the target activity has not been reached.
    When a loop input is encountered, if the two activities are in the same loop and the 'current' is reachable from 
    the 'previous' the algorithm stops. If the two activities are not in the same loop, the algorithm chooses to exit from the loop
    otherwise if the two activities are in the same loop but 'current' is not reachable from 'previous' the algorithm chooses to remain in the
    loop (i.e. it goes back)
    """
    #breakpoint()
    for in_arc in current.in_arcs:
        # setting previous_reached to False because we want to explore ALL the possible paths
        previous_reached = False
        not_found = False
        # check if the previous is in the node inputs
        inner_in_arcs_names = [inner_in_arc.source.name for inner_in_arc in in_arc.source.in_arcs if not inner_in_arc.source.label is None]
        inv_act_names = [inner_in_arc.source.name for inner_in_arc in in_arc.source.in_arcs if inner_in_arc.source.label is None]
        # recurse if the target activity is not in the inputs of the place and there is at least one invisible activity
        if not previous in inner_in_arcs_names and len(inv_act_names) > 0:
            for inner_in_arc in in_arc.source.in_arcs:
                # inner_in_arc.source is a transition
                not_found = False
                previous_reached = False
                is_activity_in_some_loop = False
                # check if we are in a loop and if it is one of the loops containing both the activities
                for loop in loops:
                    go_on = False
                    stop_recursion = False
                    # check the conditions for the recursion
                    # for loops that contain both 'current' and 'previous'
                    if loop in common_loops:
                        # 1) the node is an input of the loop, the next (backward) activity is also in the loop AND the last activitiy in the initial sequence is NOT reachable from 'previous'
                        if loop.is_node_in_loop_complete_net(inner_in_arc.source.name) and loop.is_input_node_complete_net(in_arc.source.name) and not reachability[loop.name]:
                            go_on = True
                        # if we reached an input of the loop and the last activity in the initial sequence was reachable, the algorithm stops
                        elif loop.is_input_node_complete_net(in_arc.source.name) and reachability[loop.name]:
                            not_found = True
                        # 2) the node is not an input loop AND the next (backward) is also in the loop
                        elif loop.is_node_in_loop_complete_net(inner_in_arc.source.name):
                            go_on = True
                        # in all other cases the recursion must stop
                        else:
                            stop_recursion = True
                    # for all the other loops
                    if not loop in common_loops and len(common_loops) == 0:
                        # 3) the node is a loop input AND the next (backward) activity is NOT in the loop
                        if loop.is_input_node_complete_net(in_arc.source.name) and not loop.is_node_in_loop_complete_net(inner_in_arc.source.name):
                            go_on = True
                        # 4) both the node and the next (backward) transition are in the loop
                        elif not loop.is_input_node_complete_net(in_arc.source.name) and loop.is_node_in_loop_complete_net(in_arc.source.name) and loop.is_node_in_loop_complete_net(inner_in_arc.source.name):
                            go_on = True
                        else:
                        # in all other cases the recursion must stop
                        # TODO check this condition
                            stop_recursion = True
                    if go_on:
                        # if the transition is silent and we haven't already seen it
                        if not inner_in_arc.source.name == previous and inner_in_arc.source.label is None and not inner_in_arc.source.name in passed_inv_act.keys():
                            # current is a transition, in_arc.source is a place
                            if len(in_arc.source.out_arcs) > 1:
                                if in_arc.source.name in decision_points.keys():
                                    # using set ecause it is possible to pass multiple time through the same decision point
                                    decision_points[in_arc.source.name].add(current.name)
                                else:
                                    decision_points[in_arc.source.name] = {current.name}
                            decision_points, not_found, previous_reached = get_dp_to_previous_event(previous, inner_in_arc.source, loops, common_loops, decision_points, reachability, passed_inv_act)
                            # add the silent transition to the alread seen list
                            # I need to know if passing through the invisible activity already seen would have led me to the target activity
                            passed_inv_act[inner_in_arc.source.name] = {'previous_reached': previous_reached, 'not_found': not_found}
                            if not_found:
                                if in_arc.source.name in decision_points.keys():
                                    if current.name in decision_points[in_arc.source.name] and current.label is None: 
                                        decision_points[in_arc.source.name].remove(current.name)
                        elif inner_in_arc.source.name == previous:
                            previous_reached = True
                            if len(in_arc.source.out_arcs) > 1:
                                if in_arc.source.name in decision_points.keys():
                                    # using set ecause it is possible to pass multiple time through the same decision point
                                    decision_points[in_arc.source.name].add(current.name)
                                else:
                                    decision_points[in_arc.source.name] = {current.name}
                        elif inner_in_arc.source.name in passed_inv_act.keys():
                            if passed_inv_act[inner_in_arc.source.name]['previous_reached']:
                                if len(in_arc.source.out_arcs) > 1:
                                    if in_arc.source.name in decision_points.keys():
                                        # using set ecause it is possible to pass multiple time through the same decision point
                                        decision_points[in_arc.source.name].add(current.name)
                                    else:
                                        decision_points[in_arc.source.name] = {current.name}
                            elif passed_inv_act[inner_in_arc.source.name]['not_found']:
                                not_found = True
                        elif not inner_in_arc.source.label is None:
                            not_found = True
                    if loop.is_node_in_loop_complete_net(inner_in_arc.source.name):
                        is_activity_in_some_loop = True
                    # if we reached the target, stops the search of other loops
                    if previous_reached == True:
                        break
                # if there aren't loops or the activity doesn't belong to any of them and we didn't pass inside the previous selections, go on if invisible not alread seen
                if not is_activity_in_some_loop and not stop_recursion:
                    if not inner_in_arc.source.name == previous and inner_in_arc.source.label is None and not inner_in_arc.source.name in passed_inv_act.keys():
                        if len(in_arc.source.out_arcs) > 1:
                            if in_arc.source.name in decision_points.keys():
                                # using set ecause it is possible to pass multiple time through the same decision point
                                decision_points[in_arc.source.name].add(current.name)
                            else:
                                decision_points[in_arc.source.name] = {current.name}
                        decision_points, not_found, previous_reached = get_dp_to_previous_event(previous, inner_in_arc.source, loops, common_loops, decision_points, reachability, passed_inv_act)
                        # add the silent transition to the alread seen list
                        passed_inv_act.add(inner_in_arc.source.name)
                        #breakpoint()
                    elif inner_in_arc.source.name == previous:
                        previous_reached = True
                        if len(in_arc.source.out_arcs) > 1:
                            if in_arc.source.name in decision_points.keys():
                                # using set ecause it is possible to pass multiple time through the same decision point
                                decision_points[in_arc.source.name].add(current.name)
                            else:
                                decision_points[in_arc.source.name] = {current.name}
                # if not found, delete the target activity added at the begininng for the considered decision point
                # TODO check not found for each loop separately
                # TODO check this condition
                if not_found:
                    #if current.name in decision_points[in_arc.source.name]: 
                    #    decision_points[in_arc.source.name].remove(current.name)
                    continue
            # if i finished to check inner arcs and there is at least one previous reached: previous_reached is TRue
            for inner_in_arc in in_arc.source.in_arcs:
                if inner_in_arc.source.name in passed_inv_act.keys():
                    if passed_inv_act[inner_in_arc.source.name]['previous_reached']:
                        previouse_reched = True
                        not_found = False
        # if previous in the inputs, stop
        elif previous in inner_in_arcs_names:
            previous_reached = True
            if len(in_arc.source.out_arcs) > 1:
                if in_arc.source.name in decision_points.keys():
                    # using set ecause it is possible to pass multiple time through the same decision point
                    decision_points[in_arc.source.name].add(current.name)
                else:
                    decision_points[in_arc.source.name] = {current.name}
        else:
            not_found = True
            if in_arc.source.name in decision_points.keys():
                if current.name in decision_points[in_arc.source.name] and current.label is None: 
                    decision_points[in_arc.source.name].remove(current.name)
                
    for in_arc in current.in_arcs:
        if in_arc.source.name in passed_inv_act.keys():
            if passed_inv_act[in_arc.source.name]['previous_reached']:
                previouse_reched = True
                not_found = False
    return decision_points, not_found, previous_reached

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
    the next not silent transitions are taken as added targets, following rules regarding loops if present.
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
                                next_not_silent = get_next_not_silent(next_place_silent, next_not_silent, loops, [place.name], loop.name)
                            else:
                                next_place_silent = silent_out_arc.target
                                next_not_silent = get_next_not_silent(next_place_silent, next_not_silent, loops, [place.name], 'None')
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
    if place.name == 'sink':
        return not_silent
    is_input = None
    for loop in loops:
        if is_input == loop.is_vertex_input_loop(place.name):
            is_input = True
    is_input_a_skip = check_if_skip(place)
    if (len(place.in_arcs) > 1 and not (is_input or is_input_a_skip)) or place.name in start_places:
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
                    # add start place if it is an input node not already seen
                    if loop.is_vertex_input_loop(out_arc_inn.target.name) and not out_arc_inn.target.name in start_places:
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
                                not_silent = get_next_not_silent(next_place_silent, not_silent, loops, start_places, loop_name_start)
                else:
                    not_silent = get_next_not_silent(out_arc_inn.target, not_silent, loops, start_places, loop_name_start)
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
    map_events_trans= dict()
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

def update_places_map_dp_list_if_looping(net, dp_list, places_map, loops, event_sequence, number_of_loops, trans_events_map) -> [dict, list]:
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

def get_all_dp_from_event_to_sink(event, loops, places, act_inv_already_seen) -> list:
    """ Returns all the decision points from the event to the sink, passing through invisible transitions

    Given an event, recursively explore the net collecting decision points and related invisible transitions
    that need to be passed in order to reach the sink place. If a loop is encountered, the algorithm choose to 
    exit at the first output seen.
    """
    #breakpoint()
    for out_arc in event.out_arcs:
        if out_arc.target.name == 'sink':
           return places
        else:
            #is_output = False
#            for loop in loops:
#                if loop.is_vertex_output_loop(out_arc.target.name):
#                    is_output = True
#                    loop_selected = loop
            for out_arc_inn in out_arc.target.out_arcs:
                places_length_before = sum([len(places[place]) for place in places.keys()])
                # if invisible transition
                for loop in loops:
                    if (not loop.is_output_node_complete_net(out_arc.target.name) and loop.is_node_in_loop_complete_net(out_arc_inn.target.name)) or (loop.is_output_node_complete_net(out_arc.target.name) and not loop.is_node_in_loop_complete_net(out_arc_inn.target.name)):
                        if out_arc_inn.target.label is None and not out_arc_inn.target.name in act_inv_already_seen:
                    # if it is a decision point append to the list and recurse
#                    if len(out_arc.target.out_arcs) > 1:
#                        places.append((out_arc.target.name, out_arc_inn.target.name))
                    # if is not an output of the considered loop
                        #if loop.is_vertex_output_loop(out_arc.target.name):
                            act_inv_already_seen.add(out_arc_inn.target.name)
                            if len(out_arc.target.out_arcs) > 1:
                                if not out_arc.target.name in places.keys():
                                    places[out_arc.target.name] = {out_arc_inn.target.name}
                                else:
                                    places[out_arc.target.name].add(out_arc_inn.target.name)
                                #places.append((out_arc.target.name, out_arc_inn.target.name))
                            #is_output = True
                            #loop_selected = loop
                        #if not is_output or (is_output and not loop_selected.is_vertex_in_loop(out_arc_inn.target.name)):
                            places = get_all_dp_from_event_to_sink(out_arc_inn.target, loops, places, act_inv_already_seen)
                # if anything changes, means that the cycle stopped: remove the last tuple appended
                            total_number_items = sum([len(places[place]) for place in places.keys()])
                            if total_number_items == places_length_before and total_number_items > 0:
                                if out_arc.target.name in places.keys():
                                    if out_arc_inn.target.name in places[out_arc.target.name]:
                                        places[out_arc.target.name].remove(out_arc_inn.target.name)
    return places
    
def check_if_skip(place) -> bool:
    """ Checks if a place is a 'skip'
    
    A place is a 'skip' if has N input arcs of which only one is an invisible activity and all
    are coming from the same place
    """
    #breakpoint()
    is_skip = False
    if len(place.in_arcs) > 1 and len([arc.source.name for arc in place.in_arcs if arc.source.label is None]) == 1:
        source_name = None
        source_name_count = 0
        for arc in place.in_arcs:
            if len(arc.source.in_arcs) == 1:
                for source_arc in arc.source.in_arcs:
                    if source_name is None:
                        source_name = source_arc.source.name
                    elif source_arc.source.name == source_name:
                        source_name_count += 1
        if source_name_count == len(place.in_arcs) - 1:
            is_skip = True
    return is_skip

def check_if_reducible(node):
    #breakpoint()
    is_reducible = False
    if len(node.in_arcs) > 1:
        source_name = None
        source_name_count = 0
        # check if the node incoming arcs are the same number as the outcoming from the previous one of the same type
        for arc in node.in_arcs:
            if len(arc.source.in_arcs) == 1:
                for source_arc in arc.source.in_arcs:
                    # check if the node has only the output going to the initial one
                    if len(source_arc.target.out_arcs) == 1:
                        if source_name is None:
                            source_name = source_arc.source.name
                            source_name_count = 1
                        elif source_arc.source.name == source_name:
                            source_name_count += 1
        if source_name_count == len(node.in_arcs) and source_name_count != 0:
            is_reducible = True
    # if there is only one incoming arcs, only the configuration place -> trans -> place is reducible
    elif len(node.in_arcs) == 1 and isinstance(node, PetriNet.Place):
        for arc in node.in_arcs:
            if len(arc.source.in_arcs) == 1:
                for arc_inner in arc.source.in_arcs:
                    if len(arc_inner.source.out_arcs) == 1:
                        is_reducible = True
    return is_reducible

def get_previous_same_type(node):
    #breakpoint()
    if len(node.in_arcs) > 0:
        previous = list(node.in_arcs)[0]
        if len(previous.source.in_arcs) > 1:
            print("Can't find a unique previous node of the same type of {}".format(node))
            previous_same_type = None
        elif len(previous.source.in_arcs) == 0:
            print("Previous node of the same type of {} not present".format(node))
            previous_same_type = None
        else:
            previous_same_type = list(previous.source.in_arcs)[0].source
    else:
        previous_same_type = None
    return previous_same_type

def get_previous(node):
    #breakpoint()
    previous = list(node.in_arcs)[0].source
    return previous


def discover_overlapping_rules(base_tree, dataset, attributes_map, original_rules):
    """ Discovers overlapping rules, if any.

    Given the fitted decision tree, extracts the training set instances that have been wrongly classified, i.e., for
    each leaf node, all those instances whose target is different from the leaf label. Then, it fits a new decision tree
    on those instances, builds a rules dictionary as before (disjunctions of conjunctions) and puts the resulting rules
    in disjunction with the original rules, according to the target value.
    Method taken by "Decision Mining Revisited - Discovering Overlapping Rules" by Felix Mannhardt, Massimiliano de
    Leoni, Hajo A. Reijers, Wil M.P. van der Aalst (2016).
    """

    leaf_nodes = base_tree.get_leaves_nodes()
    leaf_nodes_with_wrong_instances = [ln for ln in leaf_nodes if len(ln.get_class_names()) > 1]

    for leaf_node in leaf_nodes_with_wrong_instances:
        vertical_rules = extract_rules_from_leaf(leaf_node)

        vertical_rules_query = ""
        for r in vertical_rules:
            r_attr, r_comp, r_value = r.split(' ')
            vertical_rules_query += r_attr
            if r_comp == '=':
                vertical_rules_query += ' == '
            else:
                vertical_rules_query += ' ' + r_comp + ' '
            if dataset.dtypes[r_attr] == 'float64' or dataset.dtypes[r_attr] == 'bool':
                vertical_rules_query += r_value
            else:
                vertical_rules_query += '"' + r_value + '"'
            if r != vertical_rules[-1]:
                vertical_rules_query += ' & '

        leaf_instances = dataset.query(vertical_rules_query)
        # TODO not considering missing values for now, so wrong_instances could be empty
        # This happens because all the wrongly classified instances have missing values for the query attribute(s)
        wrong_instances = leaf_instances[leaf_instances['target'] != leaf_node._label_class]

        sub_tree = DecisionTree(attributes_map)
        sub_tree.fit(wrong_instances)

        sub_leaf_nodes = sub_tree.get_leaves_nodes()
        if len(sub_leaf_nodes) > 1:
            sub_rules = {}
            for sub_leaf_node in sub_leaf_nodes:
                new_rule = ' && '.join(vertical_rules + [extract_rules_from_leaf(sub_leaf_node)])
                if sub_leaf_node._label_class not in sub_rules.keys():
                    sub_rules[sub_leaf_node._label_class] = set()
                sub_rules[sub_leaf_node._label_class].add(new_rule)
            for sub_target_class in sub_rules.keys():
                sub_rules[sub_target_class] = ' || '.join(sub_rules[sub_target_class])
                original_rules[sub_target_class] += ' || ' + sub_rules[sub_target_class]
        # Only root in sub_tree = could not find a suitable split of the root node -> most frequent target is chosen
        elif len(wrong_instances) > 0:  # length 0 could happen since we do not consider missing values for now
            sub_target_class = wrong_instances['target'].mode()[0]
            if sub_target_class not in original_rules.keys():
                original_rules[sub_target_class] = ' && '.join(vertical_rules)
            else:
                original_rules[sub_target_class] += ' || ' + ' && '.join(vertical_rules)

    return original_rules


def compress_many_valued_attributes(rules, attributes_map):
    """ Rewrites the final rules dictionary to compress disjunctions of many-valued categorical attributes equalities

    For example, a series of atoms "org:resource = 10 || org:resource = 144 || org:resource = 68 || org:resource = 43 ||
    org:resource == 632" is rewritten as "org:resource one of [10, 144, 68, 43, 632]"
    """

    for target_class in rules.keys():
        atoms = rules[target_class].split(' || ')
        atoms_to_remove = dict()
        cat_atoms_same_attr = dict()
        for atom in atoms:
            if ' && ' not in atom:
                a_attr, a_comp, a_value = atom.split(' ')
                if a_comp == '=' and attributes_map[a_attr] == 'categorical':
                    if a_attr not in cat_atoms_same_attr.keys():
                        cat_atoms_same_attr[a_attr] = list()
                        atoms_to_remove[a_attr] = list()
                    cat_atoms_same_attr[a_attr].append(a_value)
                    atoms_to_remove[a_attr].append(atom)

        compressed_rules = list()
        for attr in cat_atoms_same_attr:
            if len(cat_atoms_same_attr[attr]) > 1:
                compressed_rules.append(attr + ' one of [' + ', '.join(sorted(cat_atoms_same_attr[attr])) + ']')
                atoms = [a for a in atoms if a not in atoms_to_remove[attr]]

        rules[target_class] = ' || '.join(atoms + compressed_rules)

    return rules
