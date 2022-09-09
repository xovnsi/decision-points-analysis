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
net_name = "running-example-Will-BPM-silent"
net = PetriNet(net_name)

# create and add places
source = PetriNet.Place("source")
sink = PetriNet.Place("sink")
p_0 = PetriNet.Place("p_0")
p_1 = PetriNet.Place("p_1")
p_2 = PetriNet.Place("p_2")
p_3 = PetriNet.Place("p_3")
p_4 = PetriNet.Place("p_4")
p_5 = PetriNet.Place("p_5")
p_6 = PetriNet.Place("p_6")
p_7 = PetriNet.Place("p_7")
p_8 = PetriNet.Place("p_8")
p_9 = PetriNet.Place("p_9")
net.places.add(source)
net.places.add(sink)
net.places.add(p_0)
net.places.add(p_1)
net.places.add(p_2)
net.places.add(p_3)
net.places.add(p_4)
net.places.add(p_5)
net.places.add(p_6)
net.places.add(p_7)
net.places.add(p_8)
net.places.add(p_9)

# create and add trasitions
t_A = PetriNet.Transition("trans_A", "Register claim")
t_B = PetriNet.Transition("trans_B", "Check all")
t_C = PetriNet.Transition("trans_C", "Check payment only")
t_D = PetriNet.Transition("trans_D", "Evaluate Claim")
t_E = PetriNet.Transition("trans_E", "Issue Payment")
t_F = PetriNet.Transition("trans_F", None)
t_G = PetriNet.Transition("trans_G", None)
t_H = PetriNet.Transition("trans_H", "Send rejection letter")
t_I = PetriNet.Transition("trans_I", "Send rejection email")
t_L = PetriNet.Transition("trans_L", "Send approval letter")
t_M = PetriNet.Transition("trans_M", "Send approval email")
t_N = PetriNet.Transition("trans_N", None)
t_O = PetriNet.Transition("trans_O", None)
t_P = PetriNet.Transition("trans_P", "Archive")
net.transitions.add(t_A)
net.transitions.add(t_B)
net.transitions.add(t_C)
net.transitions.add(t_D)
net.transitions.add(t_E)
net.transitions.add(t_F)
net.transitions.add(t_G)
net.transitions.add(t_H)
net.transitions.add(t_I)
net.transitions.add(t_L)
net.transitions.add(t_M)
net.transitions.add(t_N)
net.transitions.add(t_O)
net.transitions.add(t_P)

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
petri_utils.add_arc_from_to(t_E, p_8, net)
petri_utils.add_arc_from_to(t_F, p_4, net)
petri_utils.add_arc_from_to(p_4, t_H, net)
petri_utils.add_arc_from_to(p_4, t_I, net)
petri_utils.add_arc_from_to(t_G, p_5, net)
petri_utils.add_arc_from_to(p_5, t_L, net)
petri_utils.add_arc_from_to(p_5, t_M, net)
petri_utils.add_arc_from_to(t_H, p_6, net)
petri_utils.add_arc_from_to(t_I, p_6, net)
petri_utils.add_arc_from_to(t_L, p_7, net)
petri_utils.add_arc_from_to(t_M, p_7, net)
petri_utils.add_arc_from_to(p_6, t_N, net)
petri_utils.add_arc_from_to(p_7, t_O, net)
petri_utils.add_arc_from_to(t_N, p_8, net)
petri_utils.add_arc_from_to(t_N, p_9, net)
petri_utils.add_arc_from_to(t_O, p_9, net)
petri_utils.add_arc_from_to(p_8, t_P, net)
petri_utils.add_arc_from_to(p_9, t_P, net)
petri_utils.add_arc_from_to(t_P, sink, net)

# transitions properties
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

t_H.properties[petri_properties.TRANS_GUARD] = 'communication == "letter"'
t_H.properties[petri_properties.READ_VARIABLE] = ['communication']
t_H.properties[petri_properties.WRITE_VARIABLE] = []

t_I.properties[petri_properties.TRANS_GUARD] = 'communication == "email"'
t_I.properties[petri_properties.READ_VARIABLE] = ['communication']
t_I.properties[petri_properties.WRITE_VARIABLE] = []

t_L.properties[petri_properties.TRANS_GUARD] = 'communication == "letter"'
t_L.properties[petri_properties.READ_VARIABLE] = ['communication']
t_L.properties[petri_properties.WRITE_VARIABLE] = []

t_M.properties[petri_properties.TRANS_GUARD] = 'communication == "email"'
t_M.properties[petri_properties.READ_VARIABLE] = ['communication']
t_M.properties[petri_properties.WRITE_VARIABLE] = []

# initial and final marking
initial_marking = DataMarking()
initial_marking[source] = 1
final_marking = DataMarking()
final_marking[sink] = 1

pnml_exporter.apply(net, initial_marking, "{}.pnml".format(net_name), final_marking=final_marking)
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)
pn_visualizer.save(gviz, "{}.svg".format(net_name))
