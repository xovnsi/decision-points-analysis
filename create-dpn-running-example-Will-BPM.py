import pm4py
import copy
import numpy as np
from tqdm import tqdm
from random import choice
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.petri_net.data_petri_nets import semantics as dpn_semantics
from pm4py.objects.petri_net import properties as petri_properties
from pm4py.objects.petri_net.data_petri_nets.data_marking import DataMarking

# create empty petri net
net = PetriNet("running-example-Will-BPM")

# create and add places
source = PetriNet.Place("source")
sink = PetriNet.Place("sink")
p_0 = PetriNet.Place("p_0")
p_1 = PetriNet.Place("p_1")
p_2 = PetriNet.Place("p_2")
p_3 = PetriNet.Place("p_3")
p_4 = PetriNet.Place("p_4")
p_5 = PetriNet.Place("p_5")
net.places.add(source)
net.places.add(sink)
net.places.add(p_0)
net.places.add(p_1)
net.places.add(p_2)
net.places.add(p_3)
net.places.add(p_4)
net.places.add(p_5)

# create and add trasitions
t_A = PetriNet.Transition("trans_A", "A")
t_B = PetriNet.Transition("trans_B", "B")
t_C = PetriNet.Transition("trans_C", "C")
t_D = PetriNet.Transition("trans_D", "D")
t_E = PetriNet.Transition("trans_E", "E")
t_F = PetriNet.Transition("trans_F", "F")
t_G = PetriNet.Transition("trans_G", "G")
t_H = PetriNet.Transition("trans_H", "H")
net.transitions.add(t_A)
net.transitions.add(t_B)
net.transitions.add(t_C)
net.transitions.add(t_D)
net.transitions.add(t_E)
net.transitions.add(t_F)
net.transitions.add(t_G)
net.transitions.add(t_H)

# properties
t_B.properties[petri_properties.TRANS_GUARD] = 'amount > 500 && policyType == "normal"'
t_B.properties[petri_properties.READ_VARIABLE] = ['amount', 'policyType']
t_B.properties[petri_properties.WRITE_VARIABLE] = []

t_C.properties[petri_properties.TRANS_GUARD] = 'amount <= 500 || policyType == "premium"'
t_C.properties[petri_properties.READ_VARIABLE] = ['amount', 'policyType']
t_C.properties[petri_properties.WRITE_VARIABLE] = []

t_E.properties[petri_properties.TRANS_GUARD] = 'status == "approved"'
t_E.properties[petri_properties.READ_VARIABLE] = ['status']
t_E.properties[petri_properties.WRITE_VARIABLE] = []

t_F.properties[petri_properties.TRANS_GUARD] = 'status == "rejected"'
t_F.properties[petri_properties.READ_VARIABLE] = ['status']
t_F.properties[petri_properties.WRITE_VARIABLE] = []

t_G.properties[petri_properties.TRANS_GUARD] = 'status == "approved"'
t_G.properties[petri_properties.READ_VARIABLE] = ['status']
t_G.properties[petri_properties.WRITE_VARIABLE] = []

# add arcs
petri_utils.add_arc_from_to(source, t_A, net)
petri_utils.add_arc_from_to(t_A, p_0, net)
petri_utils.add_arc_from_to(p_0, t_B, net)
petri_utils.add_arc_from_to(p_0, t_C, net)
petri_utils.add_arc_from_to(t_B, p_1, net)
petri_utils.add_arc_from_to(t_C, p_1, net)
petri_utils.add_arc_from_to(p_1, t_D, net)
petri_utils.add_arc_from_to(t_D, p_2, net)
petri_utils.add_arc_from_to(t_D, p_3, net)
petri_utils.add_arc_from_to(p_2, t_E, net)
petri_utils.add_arc_from_to(p_2, t_F, net)
petri_utils.add_arc_from_to(p_3, t_F, net)
petri_utils.add_arc_from_to(p_3, t_G, net)
petri_utils.add_arc_from_to(t_E, p_4, net)
petri_utils.add_arc_from_to(t_F, p_4, net)
petri_utils.add_arc_from_to(t_F, p_5, net)
petri_utils.add_arc_from_to(t_G, p_5, net)
petri_utils.add_arc_from_to(p_4, t_H, net)
petri_utils.add_arc_from_to(p_5, t_H, net)
petri_utils.add_arc_from_to(t_H, sink, net)

# initial and final marking
initial_marking = DataMarking()
initial_marking[source] = 1
final_marking = DataMarking()
final_marking[sink] = 1

pnml_exporter.apply(net, initial_marking, "running-example-Will-BPM.pnml", final_marking=final_marking)
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)
pn_visualizer.save(gviz, "running-example-Will-BPM.svg")
