import numpy as np
from itertools import combinations
from utils import check_if_reducible, get_previous_same_type, get_previous
from pm4py.objects.petri_net.utils.petri_utils import remove_arc, remove_transition, remove_place, add_arc_from_to
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net.obj import Marking
import copy

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
                # check if all the places from where the input arcs are coming are part of the loop
                for arc in place.in_arcs:
                    for arc_inner in arc.source.in_arcs:
                        in_places_trans.add(arc_inner.source.name)
                if not in_places_trans.issubset(vertex_in_loops):
                    in_places.append(place.name)
    return in_places

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

def get_loop_not_silent(net, vertex_in_loop) -> list:  
    """ Returns not silent activities contained in a loop

    Given a set of vertices and the Petri net, returns only the vertices 
    corresponding to non silent transitions
    """
    not_silent = list() 
    for trans in net.transitions:
        if trans.name in vertex_in_loop and not trans.label is None:
            not_silent.append(trans.name)
    return not_silent

def get_input_near_source(net, input_places, loops) -> str:
    """ Returns the nearest input node of a loop to the source of the net

    Given the set of places inputs of a loop , the net and a list of loops objects,
    keeps only the with the shortest path starting from the source place
    """
    # if there are more than one input place
    if len(input_places) > 1:
        source = [place for place in net.places if place.name == 'source'][0]
        count = 0
        lengths = dict()
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

def get_output_adjacency_matrix(places_list, transitions_list) -> np.ndarray:
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

def get_vertex_names_from_c_matrix(c_matrix, row_names, columns_names) -> set:
    """ Returns the name of verteces contained in a loop

    Given the C matrix with the names of rows and columns, add them if the 
    corresponding matrix value is 1 (there is a loop)
    """
    vertex_in_loop = set()
    for i, row in enumerate(row_names):
        for j, col in enumerate(columns_names):
            if c_matrix[i, j] == True:
                vertex_in_loop.add(row)
                vertex_in_loop.add(col)
    return sorted(vertex_in_loop)

def simplify_net(net):
    #breakpoint()
    im = Marking()
    fm = Marking()
    for place in net.places:
        if place.name == 'source':
            source = place
        elif place.name == 'sink':
            sink = place
    im[source] = 1
    fm[sink] = 1
#    gviz = pn_visualizer.apply(net, im, fm)
#    pn_visualizer.view(gviz)
    #nodes = net.places.union(net.transitions)
    simplified = True
    simplify_map = dict()
    last_collapsed_left = dict()
    last_collapsed_right = dict()
    parallel_branches = dict()
    cont_new_place = 1
    cont_new_trans = 1
    #breakpoint()
    while simplified:
        #breakpoint()
        nodes = net.places.union(net.transitions).copy()
        im = Marking()
        fm = Marking()
        for place in net.places:
            if place.name == 'source':
                source = place
            elif place.name == 'sink':
                sink = place
        im[source] = 1
        fm[sink] = 1
       # gviz = pn_visualizer.apply(net, im, fm)
       # pn_visualizer.view(gviz)
        already_reduced = list()
        for i, node in enumerate(nodes):
            #breakpoint()
            if i == 0:
                simplified = False
                #print("set simplified to false")
            reducible = check_if_reducible(node)
            if reducible: 
                # check if already reduced in the same for loop
#                for simp_node in simplify_map.keys():
#                    if node.name in simplify_map[simp_node]:
#                        already_reduced = True
                if node.name in already_reduced:
                    continue
                if isinstance(node, PetriNet.Place):
                    #breakpoint()
#                    if node.name == 'p_14' or node.name == 'p_15':
#                        breakpoint()
                    previous_place = get_previous_same_type(node)
                    if previous_place.name in already_reduced:
                        continue
                    new_place = PetriNet.Place('p_simp_{}'.format(cont_new_place))
                    net.places.add(new_place)
                    cont_new_place += 1
                    simplify_map[new_place.name] = set()
                    if node.name in simplify_map.keys():
                        simplify_map[new_place.name] = simplify_map[new_place.name].union(simplify_map[node.name])
                        last_collapsed_right[new_place.name] = last_collapsed_right[node.name]
                        del simplify_map[node.name]
                        del last_collapsed_left[node.name]
                        del last_collapsed_right[node.name]
                    else:
                        simplify_map[new_place.name].add(node.name)
                        last_collapsed_right[new_place.name] = node.name
#\                   simplify_map[new_place.name].add(node.name)
#                    if previous_place.name == 'p_15' or previous_place.name == 'p_14':
#                        breakpoint()
#                    already_reduced = False
#                    for simp_node in simplify_map.keys():
#                        if previous_place.name in simplify_map[simp_node]:
#                            already_reduced = True
                    if previous_place is None:
                        raise Exception("Can't reduce a {}".format(node))
                    if previous_place.name in simplify_map.keys():
                        simplify_map[new_place.name] = simplify_map[new_place.name].union(simplify_map[previous_place.name])
                        last_collapsed_left[new_place.name] = last_collapsed_left[previous_place.name]
                        del simplify_map[previous_place.name]
                        del last_collapsed_left[previous_place.name]
                        del last_collapsed_right[previous_place.name]
                    else:
                        simplify_map[new_place.name].add(previous_place.name)
                        last_collapsed_left[new_place.name] = previous_place.name
#                    simplify_map[new_place.name].add(previous_place.name)
                    #breakpoint()
                    nodes_to_remove = node.in_arcs.copy()
                    for arc in nodes_to_remove:
                        if arc.source.name in simplify_map.keys():
                            simplify_map[new_place.name] = simplify_map[new_place.name].union(simplify_map[arc.source.name])
                            del simplify_map[arc.source.name]
                            del last_collapsed_left[arc.source.name]
                            del last_collapsed_right[arc.source.name]
                        else:
                            simplify_map[new_place.name].add(arc.source.name)
#                        simplify_map[new_place.name].add(arc.source.name)
                        remove_transition(net, arc.source) 
                    #previous_trans = get_previous(previous_place)
                    #if previous_trans is None:
                    #    raise Exception("Can't reduce a {}".format(node))
                    nodes_to_add = previous_place.in_arcs.copy()
                    for in_arc in nodes_to_add:
                        add_arc_from_to(in_arc.source, new_place, net)
                    nodes_to_add = previous_place.out_arcs.copy()
                    for out_arc in nodes_to_add:
                        if not out_arc.target in nodes_to_remove:
                            add_arc_from_to(new_place, out_arc.target, net)
                    nodes_to_add = node.out_arcs.copy()
                    for out_arc in nodes_to_add:
                        add_arc_from_to(new_place, out_arc.target, net)
                    #breakpoint()
                    remove_place(net, node)
                    remove_place(net, previous_place)
                    already_reduced.extend([node.name, previous_place.name])
                elif isinstance(node, PetriNet.Transition):
                    #breakpoint()
                    previous_trans = get_previous_same_type(node)
                    if previous_trans.name in already_reduced:
                        continue
                    new_trans = PetriNet.Transition('trans_simp_{}'.format(cont_new_trans),
                            'label_simp_{}'.format(cont_new_trans))
                    net.transitions.add(new_trans)
                    cont_new_trans += 1
                    simplify_map[new_trans.name] = set()
                    if node.name in simplify_map.keys():
                        breakpoint()
                        simplify_map[new_trans.name] = simplify_map[new_trans.name].union(simplify_map[node.name])
                        last_collapsed_right[new_trans.name] = last_collapsed_right[node.name]
                        del simplify_map[node.name]
                        del last_collapsed_left[node.name]
                        del last_collapsed_right[node.name]
                    else:
                        simplify_map[new_trans.name].add(node.name)
                        last_collapsed_right[new_trans.name] = node.name
#                    simplify_map[new_trans.name].add(node.name)
                    if previous_trans is None:
                        raise Exception("Can't reduce a {}".format(node))
                    if previous_trans.name in simplify_map.keys():
                        simplify_map[new_trans.name] = simplify_map[new_trans.name].union(simplify_map[previous_trans.name])
                        last_collapsed_left[new_trans.name] = last_collapsed_left[previous_trans.name]
                        del simplify_map[previous_trans.name]
                        del last_collapsed_left[previous_trans.name]
                        del last_collapsed_right[node.name]
                        # track parallelism in the net
                        trans_name = last_collapsed_right[previous_trans.name]
                    else:
                        simplify_map[new_trans.name].add(previous_trans.name)
                        last_collapsed_left[new_trans.name] = previous_trans.name
                        trans_name = previous_trans.name
                    parallel_branches[trans_name] = dict()
#                    simplify_map[new_trans.name].add(previous_trans.name)
                    nodes_to_remove = node.in_arcs.copy()
                    for i, arc in enumerate(nodes_to_remove):
                        if arc.source.name in simplify_map.keys():
                            simplify_map[new_trans.name] = simplify_map[new_trans.name].union(simplify_map[arc.source.name])
                            parallel_branches[trans_name]['branch{}'.format(i+1)] = simplify_map[arc.source.name]
                            del simplify_map[arc.source.name]
                            del last_collapsed_left[arc.source.name]
                            del last_collapsed_right[arc.source.name]
                        else:
                            simplify_map[new_trans.name].add(arc.source.name)
                            parallel_branches[trans_name]['branch{}'.format(i+1)] = arc.source.name
#                        simplify_map[new_trans.name].add(arc.source.name)
                        remove_place(net, arc.source) 
                    #previous_place = get_previous(previous_trans)
                    #if previous_place is None:
                    #    raise Exception("Can't reduce a {}".format(node))
                    nodes_to_add = previous_trans.in_arcs.copy()
                    for in_arc in nodes_to_add:
                        add_arc_from_to(in_arc.source, new_trans, net)
                    nodes_to_add = node.out_arcs.copy()
                    for out_arc in nodes_to_add:
                        add_arc_from_to(new_trans, out_arc.target, net)
                    remove_transition(net, node)
                    remove_transition(net, previous_trans)
                    already_reduced.extend([node.name, previous_trans.name])
                simplified = True
                #gviz = pn_visualizer.apply(net, im, fm)
                #pn_visualizer.view(gviz)
    return net, simplify_map, last_collapsed_left, last_collapsed_right, parallel_branches

def detect_loops(net) -> set:
    """ Detects loops in a Petri Net

    Given a Petri Net in the implementation of Pm4Py library, the code looks for every 
    place/transition belonging to a loop of whatever length from the minimum to the maximum. 
    The algorithm used for loop detection is taken from - Davidrajuh: `Detecting Existence of
    cycle in Petri Nets'. Advances in Intelligent Systems and Computing (2017)
    """
    places_list = list(net.places)
    trans_list = list(net.transitions)
    # get list of transitions and places names in the net
    places_names = [place.name for place in places_list]
    trans_names = [tran.name for tran in trans_list]
    # compute input and output adjacency matrix (input and output considering transitions)
    in_adj = get_input_adjacency_matrix(places_list, trans_list)
    out_adj = get_output_adjacency_matrix(places_list, trans_list)
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

def discriminate_loops(net, vertex_in_loop) -> list:
    """ Discriminate vertices belonging to different loops 

    Given a list of vertices discovered at a certain loop cycle by the loop detection algorithm, 
    discriminate vertices of belonging to different loops (i.e. not directly connected)
    """
    net = copy.deepcopy(net)
    vertex_in_loop = copy.deepcopy(vertex_in_loop)
    vertex_set = set(vertex_in_loop) 
    vertex_list = list()
    already_allocated = list()
    #breakpoint()
    while len(vertex_set) > 0:
#        first_element_name = list(vertex_set)[0]
#        # check if the vertex is a place or a transition (pm4py calls place as 'p_1')
#        if ("p_" in first_element_name and not ("skip" in first_element_name or "init" in first_element_name)) or "source" in first_element_name or "sink" in first_element_name:
#            first_element = [place for place in net.places if place.name == first_element_name][0]
#        else:
#            first_element = [trans for trans in net.transitions if trans.name == first_element_name][0]
        # select a place as first element if some remains
        #breakpoint()
        places_in_loop = [copy.deepcopy(place) for place in net.places if place.name in vertex_set]
        if len(places_in_loop) > 0:
            first_element = places_in_loop[0]
        else:
            first_element_name = list(vertex_set)[0]
            first_element = [copy.deepcopy(trans) for trans in net.transitions if trans.name == first_element_name][0]
        # get all the adjacent vertices that are inside the loop
        vertex, end_loop = search_adjacent_loop_vertex(first_element, [], vertex_in_loop, first_element, already_allocated, False)
        vertex = [vert.name for vert in vertex]
        vertex.sort()
        #breakpoint()
        vertex_list.append(vertex)
        already_allocated.extend(vertex)
        vertex_loop_set = set(vertex)
        # exclude the vrtices found, the remaining are part of another loop
        vertex_set = vertex_set.difference(vertex_loop_set)
    return vertex_list

def search_adjacent_loop_vertex(vertex, vertex_in_loop, total_vertex_in_loop, start_element, already_allocated, end_one_loop) -> list:
    """ Returns the list of adjacent vertices belonging to the same loop

    Given a vertex and all the vertices belonging to a loop, returns all the vertices that are
    directly connected to the first one
    """
    vertex_in_loop.append(vertex)
    # stops if the vertex is an output vertex of the loop
    target_nodes = [out_arc.target for out_arc in vertex.out_arcs]
    if start_element in target_nodes:
        end_one_loop = True
        return vertex_in_loop, end_one_loop
    else:
        for out_arc in vertex.out_arcs:
            if not end_one_loop:
                if out_arc.target in vertex_in_loop and out_arc.target != start_element:
                    #breakpoint()
                    vertex_in_loop.remove(vertex)
                else:
                    if out_arc.target.name in total_vertex_in_loop and not out_arc.target in vertex_in_loop:# and not out_arc.target.name in already_allocated:
                        vertex_in_loop, end_one_loop = search_adjacent_loop_vertex(out_arc.target, vertex_in_loop, total_vertex_in_loop, start_element, already_allocated, end_one_loop)
                    elif out_arc.target.name in total_vertex_in_loop and not out_arc.target in vertex_in_loop:
                        vertex_in_loop.append(out_arc.target)
                        end_one_loop = True
    return vertex_in_loop, end_one_loop
    
def delete_composite_loops(vertex_in_loop) -> dict:
    """ Delete loops that are composed by others already detected

    The loop detection algorithm can discover loops that are actually composed by 
    different loops of minor length already discovered. Given the dictionary of loops,
    checks if a loop of minor length is completely contained in a longer one. Longer loops
    containing shorter ones are discarded
    """
    #breakpoint()
#    for loop_length in vertex_in_loop.keys():
#        for sequence in vertex_in_loop[loop_length]:
#            for loop_length_inner in vertex_in_loop.keys():
#                if not loop_length_inner == loop_length:
#                    for inner_sequence in vertex_in_loop[loop_length_inner]:
#                        if len(set(sequence).difference(inner_sequence)) == 0:
#                            vertex_in_loop[loop_length_inner].remove(inner_sequence)
#    new_vertex_in_loop = dict((key, value) for key, value in vertex_in_loop.items() if value)
    new_vertex_in_loop = dict()
    unique_loops = list()
    unique_lengths = list()
    for loop_length in vertex_in_loop.keys():
        for sequence in vertex_in_loop[loop_length]:
            duplicate_sequence = False
            combined_loop = False
            sequence_set = set(sequence)
            if len(unique_loops) == 0:
                unique_loops = [sequence]
                unique_lengths = [loop_length]
            else:
                for cont_seq in range(len(unique_loops)):
                    sequence_combinations = combinations(unique_loops, cont_seq+1)
                    seq_set = set()
                    #breakpoint()
                    for comb in sequence_combinations:
                        #breakpoint()
                        seq_set_unique_comb = [seq_set.union(prova) for prova in comb][0]
                        if seq_set_unique_comb == sequence_set:
                            # if the combinations are only of the 'first order' must be a duplicate
                            if cont_seq + 1 == 1:
                                duplicate_sequence = True
                                combined_loop = False
                                break
                            else:
                                duplicate_sequence = False
                                combined_loop = True
                                comb_selected = comb
                                break
                    if duplicate_sequence or combined_loop: 
                        break
#                for unique_sequence in unique_loops:
#                    if len(sequence) == len(unique_sequence) and len(sequence_set.difference(set(unique_sequence))) == 0:
#                        duplicate_sequence = True
#                        break
                #breakpoint()
                if duplicate_sequence: 
                    break
                elif combined_loop:
                    #breakpoint()
                    for seq_comb in comb_selected:
                        idx_list = [idx for idx, seq in enumerate(unique_loops) if seq == sequence][0]
                        del unique_loops[idx_list]
                        del unique_lengths[idx_list]
                else:
                    unique_loops.append(sequence)
                    unique_lengths.append(loop_length)
    #breakpoint()
    new_vertex_in_loop = {key: [] for key in unique_lengths}
    for i, loop_length in enumerate(unique_lengths):
        new_vertex_in_loop[loop_length].append(unique_loops[i])
    return new_vertex_in_loop

def count_length_from_source(place, input_places, count, loops, lengths, initial_input_places) -> int:
    """ Counts the number of vertices between an input place of a loop and the source of the net

    Given a place and a list of input places names recursively add one to the count until the place 
    has been reached starting from the net source or the sink has been found.
    """
    # if the place is inside the initial set but not in the actual one, it has been removed thus the cycle ends
    if not place.name in input_places and place.name in initial_input_places:
        return lengths
    for out_arc in place.out_arcs:
        for out_arc_inn in out_arc.target.out_arcs:
            if not len(input_places) == 0:
                # if the sink has been reached ends immediately
                if out_arc_inn.target.name == 'sink':
                    return lengths
                if out_arc_inn.target.name in input_places:
                    count += 1
                    lengths[out_arc_inn.target.name] = count
                    input_places.remove(out_arc_inn.target.name)
                    # if there is only one input place, ends immediately
                    if len(input_places) == 0:
                        return lengths
                    # recurse
                    else:
                        lengths = count_length_from_source(out_arc_inn.target, input_places, count, loops, lengths, initial_input_places)
                else:
                    loop_name = 'None'
                    not_in_loop = False
                    # check if in loop
                    for loop in loops:
                        # TODO check if this condition is right
                        if loop.is_vertex_output_loop(out_arc_inn.target) and not loop.is_vertex_output_loop(out_arc_inn.source):
                            count += 1
                            lengths = count_length_from_source(out_arc_inn.target, input_places, count, loops, lengths, initial_input_places)
                        else:
                            not_in_loop = True
                    if not_in_loop:
                        count += 1
                        lengths = count_length_from_source(out_arc_inn.target, input_places, count, loops, lengths, initial_input_places)
    return lengths
