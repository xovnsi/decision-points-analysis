import numpy as np

def get_in_place_loop(net, vertex_in_loops) -> list:
    """ Gets the name of places that are input of a loop

    Given a net and places and transitions belonging to whatever loop in it,
    computes places that are input in one of them. A place is an input if at least one input
    arc is coming from a transition outside the loop and the place of that transition is also outside (nested loops).
    """
    # initialize
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
    vertex_set = set(vertex_in_loop) 
    vertex_list = list()
    while len(vertex_set) > 0:
        first_element_name = list(vertex_set)[0]
        # check if the vertex is a place or a transition (pm4py calls place as 'p_1')
        if ("p_" in first_element_name and not ("skip" in first_element_name or "init" in first_element_name)) or "source" in first_element_name or "sink" in first_element_name:
            first_element = [place for place in net.places if place.name == first_element_name][0]
        else:
            first_element = [trans for trans in net.transitions if trans.name == first_element_name][0]
        # get all the adjacent vertices that are inside the loop
        vertex = search_adjacent_loop_vertex(first_element, [], vertex_in_loop)
        vertex = [vert.name for vert in vertex]
        vertex.sort()
        vertex_list.append(vertex)
        vertex_loop_set = set(vertex)
        # exclude the vertices found, the remaining are part of another loop
        vertex_set = vertex_set.difference(vertex_loop_set)
    return vertex_list

def search_adjacent_loop_vertex(vertex, vertex_in_loop, total_vertex_in_loop) -> list:
    """ Returns the list of adjacent vertices belonging to the same loop

    Given a vertex and all the vertices belonging to a loop, returns all the vertices that are
    directly connected to the first one
    """
    vertex_in_loop.append(vertex)
    # stops if the vertex is an output vertex of the loop
    if vertex_in_loop[0] in vertex.out_arcs:
        return vertex_in_loop
    else:
        for out_arc in vertex.out_arcs:
            if out_arc.target.name in total_vertex_in_loop and not out_arc.target in vertex_in_loop:
                search_adjacent_loop_vertex(out_arc.target, vertex_in_loop, total_vertex_in_loop)
    return vertex_in_loop
    
def delete_composite_loops(vertex_in_loop) -> dict:
    """ Delete loops that are composed by others already detected

    The loop detection algorithm can discover loops that are actually composed by 
    different loops of minor length already discovered. Given the dictionary of loops,
    checks if a loop of minor length is completely contained in a longer one. Longer loops
    containing shorter ones are discarded
    """
    for loop_length in vertex_in_loop.keys():
        for sequence in vertex_in_loop[loop_length]:
            for loop_length_inner in vertex_in_loop.keys():
                if not loop_length_inner == loop_length:
                    for inner_sequence in vertex_in_loop[loop_length_inner]:
                        if len(set(sequence).difference(inner_sequence)) == 0:
                            vertex_in_loop[loop_length_inner].remove(inner_sequence)
    new_vertex_in_loop = dict((key, value) for key, value in vertex_in_loop.items() if value)
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

