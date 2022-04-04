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
net = PetriNet("petri_net")

# create and add fundamental places
source = PetriNet.Place("source")
sink = PetriNet.Place("sink")
p_1 = PetriNet.Place("p_1")
p_2 = PetriNet.Place("p_2")
net.places.add(source)
net.places.add(sink)
net.places.add(p_1)
net.places.add(p_2)

# create and add trasitions
t_1 = PetriNet.Transition("name_1", "label_1")
t_2 = PetriNet.Transition("name_2", "label_2")
#t_2.properties[petri_properties.TRANS_GUARD] = '(A >= 5 && cat_1 == 0) || (A == -1 && cat_1 > 0.5) || (A >= 5 && cat_1 > 0.5)'
t_2.properties[petri_properties.TRANS_GUARD] = '(A > 0 && A >= 5) || cat == "cat_1"'
t_2.properties[petri_properties.READ_VARIABLE] = ['A', 'cat']
t_2.properties[petri_properties.WRITE_VARIABLE] = []
t_3 = PetriNet.Transition("name_3", "label_3")
#t_3.properties[petri_properties.TRANS_GUARD] = '(A < 5 && cat_2 == 0) || (A == -1 && cat_2 > 0.5) || (A < 5 && cat_2 > 0.5)'
t_3.properties[petri_properties.TRANS_GUARD] = '(A > 0 && A < 5) || cat == "cat_2"'
t_3.properties[petri_properties.READ_VARIABLE] = ['A', 'cat']
t_3.properties[petri_properties.WRITE_VARIABLE] = []
t_4 = PetriNet.Transition("name_4", "label_4")
net.transitions.add(t_1)
net.transitions.add(t_2)
net.transitions.add(t_3)
net.transitions.add(t_4)

# add arcs
petri_utils.add_arc_from_to(source, t_1, net)
petri_utils.add_arc_from_to(t_1, p_1, net)
petri_utils.add_arc_from_to(p_1, t_2, net)
petri_utils.add_arc_from_to(p_1, t_3, net)
petri_utils.add_arc_from_to(t_2, p_2, net)
petri_utils.add_arc_from_to(t_3, p_2, net)
petri_utils.add_arc_from_to(p_2, t_4, net)
petri_utils.add_arc_from_to(t_4, sink, net)

# initial and final marking
initial_marking = DataMarking()
initial_marking[source] = 1
final_marking = DataMarking()
final_marking[sink] = 1

pnml_exporter.apply(net, initial_marking, "one-split-PetriNet-categorical.pnml", final_marking=final_marking)
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)
pn_visualizer.save(gviz, "one-split-PetriNet-categorical.svg")
