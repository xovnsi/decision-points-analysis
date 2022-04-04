from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.visualization.petri_net import visualizer as pn_visualizer
import numpy as np

#net, initial_marking, final_marking = pnml_importer.apply("models/running-example-Will-BPM-silent.pnml")
#net, initial_marking, final_marking = pnml_importer.apply("models/one-split-PetriNet.pnml")
#places = list(net.places)
net = PetriNet("pn-loop")
#
## create and add places
source = PetriNet.Place("source")
sink = PetriNet.Place("sink")
p_1 = PetriNet.Place("p_1")
p_2 = PetriNet.Place("p_2")
p_3 = PetriNet.Place("p_3")
p_4 = PetriNet.Place("p_4")
net.places.add(source)
net.places.add(sink)
net.places.add(p_1)
net.places.add(p_2)
net.places.add(p_3)
net.places.add(p_4)
##
### create and add trasitions
t_1 = PetriNet.Transition("trans_1", "t1")
t_2 = PetriNet.Transition("trans_2", "t2")
t_3 = PetriNet.Transition("trans_3", "t3")
t_4 = PetriNet.Transition("trans_4", "t4")
t_5 = PetriNet.Transition("trans_5", "t5")
t_6 = PetriNet.Transition("trans_6", "t6")
t_7 = PetriNet.Transition("trans_7", "t7")
t_8 = PetriNet.Transition("trans_8", "t8")
net.transitions.add(t_1)
net.transitions.add(t_2)
net.transitions.add(t_3)
net.transitions.add(t_4)
net.transitions.add(t_5)
net.transitions.add(t_6)
net.transitions.add(t_7)
net.transitions.add(t_8)
### add arcs
petri_utils.add_arc_from_to(source, t_1, net)
petri_utils.add_arc_from_to(t_1, p_1, net)
petri_utils.add_arc_from_to(p_1, t_2, net)
petri_utils.add_arc_from_to(p_1, t_3, net)
petri_utils.add_arc_from_to(t_2, p_2, net)
petri_utils.add_arc_from_to(t_3, p_2, net)
petri_utils.add_arc_from_to(p_2, t_4, net)
petri_utils.add_arc_from_to(t_4, p_3, net)
petri_utils.add_arc_from_to(p_3, t_5, net)
petri_utils.add_arc_from_to(p_3, t_6, net)
petri_utils.add_arc_from_to(t_5, p_4, net)
petri_utils.add_arc_from_to(t_6, p_4, net)
petri_utils.add_arc_from_to(p_4, t_8, net)
petri_utils.add_arc_from_to(p_4, t_7, net)
petri_utils.add_arc_from_to(t_7, p_2, net)
petri_utils.add_arc_from_to(t_8, sink, net)
#
#gviz = pn_visualizer.apply(net, initial_marking, final_marking)
gviz = pn_visualizer.apply(net)
pn_visualizer.view(gviz)

def get_next_not_silent(place, not_silent):
    #breakpoint()
    if len(place.in_arcs) > 1:
        return not_silent
    out_arcs_label = [arc.target.label for arc in place.out_arcs]
    if not None in out_arcs_label:
        not_silent.extend(out_arcs_label)
        return not_silent
    for out_arc in place.out_arcs:
        if not out_arc.target.label is None:
            not_silent.extend(out_arc.target.label)
        else:
            for out_arc_inn in out_arc.target.out_arcs:
                not_silent = get_next_not_silent(out_arc_inn.target, not_silent)
    return not_silent

def get_previous_not_silent(place, not_silent):
    breakpoint()
    in_arcs_label = [arc.source.label for arc in place.in_arcs]
    if not None in in_arcs_label:
        not_silent.extend(in_arcs_label)
        return not_silent
    for in_arc in place.in_arcs:
        if not in_arc.source.label is None:
            not_silent.extend(in_arc.source.label)
        else:
            for in_arc_inn in in_arc.target.out_arcs:
                not_silent = get_previous_not_silent(in_arc_inn.source, not_silent)
    return not_silent

def get_silent_activities_map(net):
    breakpoint()
    not_silent_dict = dict()
    for transition in net.transitions:
        if transition.label is None:
            for out_place in transition.out_arcs:
                not_silent_trans = get_next_not_silent(out_place.target, [])
            not_silent_dict[transition.name] = {"labels": not_silent_trans}
            for in_place in transition.in_arcs:
                not_silent_trans = get_previous_not_silent(in_place.source, [])
            not_silent_trans = list(dict.fromkeys(not_silent_trans))
            not_silent_dict[transition.name]["attributes"] = not_silent_trans
    return not_silent_dict

def get_input_adjacency_matrix(places_list, transitions_list):
    # adjacency matrix of TRANSITIONS INPUT (place -> transition)
    #breakpoint()
    adj = np.zeros((len(transitions_list), len(places_list)))
    for i, trans in enumerate(transitions_list):
        for j, place in enumerate(places_list):
            in_arcs_name = [arc.target.name for arc in place.out_arcs]
            if trans.name in in_arcs_name:
                adj[i, j] = 1
    return adj

def get_output_adjacency_matrix(places_list, transitions_list):
    # adjacency matrix of TRANSITIONS OUTPUT (transition -> place)
    #breakpoint()
    adj = np.zeros((len(transitions_list), len(places_list)))
    for i, trans in enumerate(transitions_list):
        for j, place in enumerate(places_list):
            out_arcs_name = [arc.source.name for arc in place.in_arcs]
            if trans.name in out_arcs_name:
                adj[i, j] = 1
    return adj

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
        
#def detect_cycles(adj_mat):
#breakpoint()
places = list(net.places)
places_names = [place.name for place in places]
trans = list(net.transitions)
trans_names = [tran.name for tran in trans]
in_adj = get_input_adjacency_matrix(places, trans)
out_adj = get_output_adjacency_matrix(places, trans)
print_matrix(in_adj, trans_names, places_names)
print_matrix(out_adj, trans_names, places_names)
#breakpoint()
old_quadrant_1 = np.zeros((len(trans), len(trans)))
old_quadrant_2 = out_adj
old_quadrant_3 = in_adj.T
old_quadrant_4 = np.zeros((len(places), len(places)))
print(np.logical_and(out_adj, in_adj))
print(np.logical_and(in_adj.T, out_adj.T))
cont_r = 1
while (cont_r <= len(places) + len(trans)):
    #breakpoint()
    cont_r += 1
    new_quadrant_1 = np.zeros((len(trans), len(trans)))
    new_quadrant_2 = np.zeros((len(trans), len(places)))
    new_quadrant_3 = np.zeros((len(places), len(trans)))
    new_quadrant_4 = np.zeros((len(places), len(places)))
    if cont_r % 2 == 1:
        new_quadrant_2 = np.dot(out_adj, old_quadrant_4)
        new_quadrant_3 = np.dot(in_adj.T, old_quadrant_1)
        new_quadrant_2T = new_quadrant_2.T
        new_quadrant_3T = new_quadrant_3.T
        c_quadrant_2 = np.logical_and(new_quadrant_2, new_quadrant_3T)
        c_quadrant_3 = np.logical_and(new_quadrant_3, new_quadrant_2T)
        #breakpoint()
        print("C quadrant 2")
        print_matrix(c_quadrant_2, trans_names, places_names)
        print("C quadrant 3")
        print_matrix(c_quadrant_3.T, trans_names, places_names)
    else:
        new_quadrant_1 = np.dot(out_adj, old_quadrant_3)
        new_quadrant_4 = np.dot(in_adj.T, old_quadrant_2)
        new_quadrant_1T = new_quadrant_1.T
        new_quadrant_4T = new_quadrant_4.T
        c_quadrant_1 = np.logical_and(new_quadrant_1, new_quadrant_1T)
        c_quadrant_4 = np.logical_and(new_quadrant_4, new_quadrant_4T)
        print("C quadrant 1")
        print_matrix(c_quadrant_1, trans_names, trans_names)
        print("C quadrant 4")
        print_matrix(c_quadrant_4.T, places_names, places_names)

    old_quadrant_1 = new_quadrant_1
    old_quadrant_2 = new_quadrant_2
    old_quadrant_3 = new_quadrant_3
    old_quadrant_4 = new_quadrant_4

#not_silent_dict = get_silent_activities_map(net)
