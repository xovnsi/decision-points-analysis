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
                    #breakpoint()
                    next_not_silent_to_be_removed = set()
                    for not_silent in next_not_silent:
                        if not check_if_reachable_without_loops(loops, arc.target, not_silent, False):
                            next_not_silent_to_be_removed.add(not_silent)
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
    # first stop condition
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
    is_input_a_skip = len(place.in_arcs) == 2 and len([arc.source.name for arc in place.in_arcs if arc.source.label is None]) == 1
    if (len(place.in_arcs) > 1 and not (is_input or is_input_a_skip)) or place.name in start_places: # or (is_output and loop_name == loop_name_start):
        return not_silent
    out_arcs_label = [arc.target.label for arc in place.out_arcs]
    # second stop condition
    if not None in out_arcs_label:
        not_silent = not_silent.union(set(out_arcs_label))
        return not_silent
    for out_arc in place.out_arcs:
        # add not silent
        if not out_arc.target.label is None:
            not_silent.add(out_arc.target.label)
        else:
            # recursive part if is silent
            for out_arc_inn in out_arc.target.out_arcs:
                added_start_place = False
                for loop in loops:
                    if loop.is_vertex_input_loop(out_arc_inn.target.name) and int(loop.name.split("_")[1]) != int(loop_length) and not out_arc_inn.target.name in start_places:
                        start_places.append(out_arc_inn.target.name)
                        added_start_place = True
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

def get_out_place_loop(net, vertex_in_loops) -> list:
    """ Gets the name of places that are output of a loop

    Given a net and places and transitions belonging to whatever loop in it,
    computes places that are output in one of them. A place is an output if at least one output
    arc is coming from a transition outside the loop or the transition leads to a place inside the loop (nested loops).
    """
    # initialize
    out_places = list()
    for place in net.places:
        if place.name in vertex_in_loops:
            out_trans = set([arc.target.name for arc in place.out_arcs])
            # check if all the output transitions are contained in a loop
            if not out_trans.issubset(vertex_in_loops):
                out_places_trans = set()
                for arc in place.out_arcs:
                    for arc_inner in arc.target.out_arcs:
                        out_places_trans.add(arc_inner.target.name)
                if not out_places_trans.issubset(vertex_in_loops):
                    out_places.append(place.name)
    return out_places

def get_in_place_loop(net, vertex_in_loops) -> list:
    """ Gets the name of places that are input of a loop

    Given a net and places and transitions belonging to whatever loop in it,
    computes places that are input in one of them. A place is an input if at least one input
    arc is coming from a transition outside the loop and the place of that transition is also outside (nested loops).
    """
    # initialize
    #breakpoint()
    in_places = list()
    for place in net.places:
        if place.name in vertex_in_loops:
            in_trans = set([arc.source.name for arc in place.in_arcs])
            # check if all the input transitions are contained in a loop
            if not in_trans.issubset(vertex_in_loops):
                in_places_trans = set()
                # TODO check wtf i have done
                for arc in place.in_arcs:
                    for arc_inner in arc.source.in_arcs:
                        in_places_trans.add(arc_inner.source.name)
                if not in_places_trans.issubset(vertex_in_loops):
                    #breakpoint()
                    in_places.append(place.name)
    #breakpoint()
    return in_places

def get_place_from_event(places_map, event, dp_list) -> list:
    """ Computes the places that are decision points of a certain transition

    Given a lot of input, for every decision point checks different conditions in order to decide if it has to be added to the list of 
    decision points for that transition. The place is considered a valid candidate if 
    I) the transition considered and the previous are both in a loop and the place belong to a loop
    II) the transition considered and the previous are both not in a loop
    III) the transition considered is in loop, the last is not and the place is an input place of the loop
    IV) the last transition is in the loop and the place is an output place of the loop
    V) one transition and the place are outside the loop (deals with invisible activities before the loop)
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

def get_map_events_transitions(net) -> dict:
    """ Compute a map of event name and transitions name

    Given a Petri Net in the implementation of Pm4Py library, the function creates
    a dictionary containing for every event the corresponding transition name
    """
    # initialize
    map_events_trans= dict()
    for trans in net.transitions:
        if not trans.label is None:
            map_events_trans[trans.name] = trans.label
    return map_events_trans

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
    return sorted(vertex_in_loop)

def detect_loops(net) -> set:
    """ Detects loops in a Petri Net

    Given a Petri Net in the implementation of Pm4Py library, the code look for every 
    place/transition belonging to a loop of whatever length from the minimum to the maximum. 
    The algorithm used for loop detection is taken from - Davidrajuh: `Detecting Existence of
    cycle in Petri Nets'. Advances in Intelligent Systems and Computing (2017)
    """
    #breakpoint()
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
    vertex_in_loop = dict()
    # Every iteration detects loop of length 2*cont_r, due to the bipartite nature of the net
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
            if len(vert_loop) > 0:
                vert_loop.extend(get_vertex_names_from_c_matrix(c_quadrant_4, places_names, places_names))
            else:
                vert_loop = get_vertex_names_from_c_matrix(c_quadrant_4, places_names, places_names)
        else:
            new_quadrant_1 = np.dot(out_adj, old_quadrant_3)
            new_quadrant_4 = np.dot(in_adj.T, old_quadrant_2)
            new_quadrant_1T = new_quadrant_1.T
            new_quadrant_4T = new_quadrant_4.T
            c_quadrant_1 = np.logical_and(new_quadrant_1, new_quadrant_1T)
            c_quadrant_4 = np.logical_and(new_quadrant_4, new_quadrant_4T)
            vert_loop = get_vertex_names_from_c_matrix(c_quadrant_1, trans_names, trans_names)
            places_trans_in_loop = places_trans_in_loop.union(vert_loop)
            if len(vert_loop) > 0:
                vert_loop.extend(get_vertex_names_from_c_matrix(c_quadrant_4, places_names, places_names))
            else:
                vert_loop = get_vertex_names_from_c_matrix(c_quadrant_4, places_names, places_names)
            places_trans_in_loop = places_trans_in_loop.union(vert_loop)
        # update old quadrants
        old_quadrant_1 = new_quadrant_1
        old_quadrant_2 = new_quadrant_2
        old_quadrant_3 = new_quadrant_3
        old_quadrant_4 = new_quadrant_4

        if len(vert_loop) > 0:
            found_multiple_loops = False
            vert_loops_discriminated = discriminate_loops(net, vert_loop)
            for el in vert_loops_discriminated:
                el.sort()
            #breakpoint()
            for vert_loop_discriminated in vert_loops_discriminated:
                for loop_length in vertex_in_loop.keys():
                    if 2*cont_r % int(loop_length.split('_')[1]) == 0 and vert_loop_discriminated in vertex_in_loop[loop_length]:
                        found_multiple_loops = True
                if not found_multiple_loops:
                    # TODO directly insert loop sequence at the right length in the dictionary
                    if not "length_{}".format(2*cont_r) in vertex_in_loop.keys():
                        vertex_in_loop["length_{}".format(2*cont_r)] = [vert_loop_discriminated]
                    else:
                        vertex_in_loop["length_{}".format(2*cont_r)].append(vert_loop_discriminated)
                found_multiple_loops = False
    return vertex_in_loop

def discriminate_loops(net, vertex_in_loop):
    vertex_set = set(vertex_in_loop) 
    vertex_list = list()
    while len(vertex_set) > 0:
        first_element_name = list(vertex_set)[0]
        if ("p_" in first_element_name and not ("skip" in first_element_name or "init" in first_element_name)) or "source" in first_element_name or "sink" in first_element_name:
            if len([place for place in net.places if place.name == first_element_name]) == 0:
                breakpoint()
            first_element = [place for place in net.places if place.name == first_element_name][0]
        else:
            first_element = [trans for trans in net.transitions if trans.name == first_element_name][0]
        #breakpoint()
        vertex = search_adjacent_loop_vertex(first_element, [], vertex_in_loop)
        vertex = [vert.name for vert in vertex]
        vertex.sort()
        vertex_list.append(vertex)
        vertex_loop_set = set(vertex)
        vertex_set = vertex_set.difference(vertex_loop_set)
        #breakpoint()
    return vertex_list

def search_adjacent_loop_vertex(vertex, vertex_in_loop, total_vertex_in_loop):
    #breakpoint()
    vertex_in_loop.append(vertex)
    if vertex_in_loop[0] in vertex.out_arcs:
        return vertex_in_loop
    else:
        for out_arc in vertex.out_arcs:
            if out_arc.target.name in total_vertex_in_loop and not out_arc.target in vertex_in_loop:
                search_adjacent_loop_vertex(out_arc.target, vertex_in_loop, total_vertex_in_loop)
    return vertex_in_loop
    
def delete_composite_loops(vertex_in_loop):
    for loop_length in vertex_in_loop.keys():
        for sequence in vertex_in_loop[loop_length]:
            for loop_length_inner in vertex_in_loop.keys():
                if not loop_length_inner == loop_length:
                    for inner_sequence in vertex_in_loop[loop_length_inner]:
                        if len(set(sequence).difference(inner_sequence)) == 0:
                            vertex_in_loop[loop_length_inner].remove(inner_sequence)
#                            if len(vertex_in_loop[loop_length_inner]) == 0:
#                                vertex_in_loop.pop(loop_length_inner)
    new_vertex_in_loop = dict((key, value) for key, value in vertex_in_loop.items() if value)
    return new_vertex_in_loop

def get_loop_not_silent(net, vertex_in_loop):  
    not_silent = list() 
    for trans in net.transitions:
        if trans.name in vertex_in_loop and not trans.label is None:
            not_silent.append(trans.name)
    return not_silent

def get_input_near_source(net, input_places, loops):
    #breakpoint()
    if len(input_places) > 1:
        source = [place for place in net.places if place.name == 'source'][0]
        count = 0
        lengths = dict()
        #breakpoint()
        lengths = count_length_from_source(source, input_places, count, loops, lengths, input_places.copy())
        count = -1
        for input_place in lengths:
            if count == -1:
                nearest = input_place
                count = lengths[input_place]
            elif lengths[input_place] < count:
                nearest = input_place
                count = lengths[input_place]
    else:
        nearest = input_places[0]
    return nearest

def count_length_from_source(place, input_places, count, loops, lengths, initial_input_places):
    #breakpoint()
    if not place.name in input_places and place.name in initial_input_places:
        return lengths
    for out_arc in place.out_arcs:
        for out_arc_inn in out_arc.target.out_arcs:
            if not len(input_places) == 0:
                if out_arc_inn.target.name == 'sink':
                    return lengths
                if out_arc_inn.target.name in input_places:
                    count += 1
                    lengths[out_arc_inn.target.name] = count
                    input_places.remove(out_arc_inn.target.name)
                    if len(input_places) == 0:
                        return lengths
                    else:
                        lengths = count_length_from_source(out_arc_inn.target, input_places, count, loops, lengths, initial_input_places)
                else:
                    loop_name = 'None'
                    not_in_loop = False
                    for loop in loops:
                        if loop.is_vertex_output_loop(out_arc_inn.target) and not loop.is_vertex_output_loop(out_arc_inn.source):
                            count += 1
                            lengths = cont_length_from_source(out_arc_inn.target, input_places, count, loops, lengths, initial_input_places)
                        else:
                            not_in_loop = True
                    if not_in_loop:
                        count += 1
                        lengths = count_length_from_source(out_arc_inn.target, input_places, count, loops, lengths, initial_input_places)
    return lengths

def check_if_reachable_without_loops(loops, start_trans, end_trans, reachable):
    for out_arc in start_trans.out_arcs:
        if out_arc.target.name == "sink":
            return reachable
        else:
            is_a_loop_output = False
            loop_name = 'None'
            for loop in loops:
                if loop.is_vertex_output_loop(out_arc.target.name):
                    is_a_loop_output = True
                    loop_name = loop.name
                    loop_selected = loop
            for out_arc_inn in out_arc.target.out_arcs:
                if out_arc_inn.target.label is None and not is_a_loop_output:
                    reachable = check_if_reachable_without_loops(loops, out_arc_inn.target, end_trans, reachable)
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

def update_places_map_dp_list_if_looping(net, dp_list, places_map, loops, event_sequence, number_of_loops, trans_events_map):
    transition_sequence = [trans_events_map[event] for event in event_sequence]
    loop_name = 'None'
    for loop in loops:
        loop.check_if_loop_is_active(net, transition_sequence)
        if loop.is_active():
            loop.count_number_of_loops(net, transition_sequence.copy())
            if number_of_loops[loop.name] < loop.number_of_loops:
                number_of_loops[loop.name] = loop.number_of_loops
                for dp in loop.dp_forward.keys():
                    if not dp in dp_list:
                        dp_list.append(dp)
                    loop_reachable = loop.dp_forward.copy()
                    for trans in places_map[dp].keys():
                        if loop.is_vertex_in_loop(trans):
                            if trans in loop.dp_forward[dp].keys():
                                if event_sequence[-1] in loop.dp_forward[dp][trans]:
                                    if isinstance(places_map[dp][trans], set):
                                        #breakpoint()
                                        places_map[dp][trans] = places_map[dp][trans].difference(loop.events)
                                        places_map[dp][trans] = places_map[dp][trans].union(loop_reachable[dp][trans])
                for dp in loop.dp_backward.keys():
                    if not dp in dp_list:
                        dp_list.append(dp)
                    loop_reachable = loop.dp_backward.copy()
                    for trans in places_map[dp].keys():
                        if loop.is_vertex_in_loop(trans):
                            if trans in loop.dp_backward[dp].keys():
                                if event_sequence[-1] in loop.dp_backward[dp][trans]:
                                    if isinstance(places_map[dp][trans], set):
                                        places_map[dp][trans] = places_map[dp][trans].difference(loop.events)
                                        places_map[dp][trans] = places_map[dp][trans].union(loop_reachable[dp][trans])
    return places_map, dp_list
        
def update_dp_list(places_from_event, dp_list):
    for place, trans_name in places_from_event:
        if place in dp_list:
            dp_list.remove(place)
    return dp_list

def get_all_dp_from_event_to_sink(event, loops, places):
    #breakpoint()
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
                if out_arc_inn.target.label is None:
                    if not is_output or (is_output and not loop_selected.is_vertex_in_loop(out_arc_inn.target.name)):
                        if len(out_arc.target.out_arcs) > 1:
                            places.append((out_arc.target.name, out_arc_inn.target.name))
                        places = get_all_dp_from_event_to_sink(out_arc_inn.target, loops, places)
            if len(places) == places_length_before:
                places.remove((out_arc.target.name, out_arc_inn.target.name))
    return places
    
