import pm4py
import copy
from random import choice
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.decision_mining import algorithm as decision_mining
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.data_petri_nets import semantics as dpn_semantics
from pm4py.objects.petri_net import properties as petri_properties
from pm4py.objects.petri_net.data_petri_nets.data_marking import DataMarking

# create empty petri net
net = PetriNet("petri_net")

# create and add fundamental places
source = PetriNet.Place("source")
sink = PetriNet.Place("sink")
p_1 = PetriNet.Place("p_1")
net.places.add(source)
net.places.add(sink)
net.places.add(p_1)

# create and add trasitions
t_1 = PetriNet.Transition("name_1", "label_1")
t_2 = PetriNet.Transition("name_2", "label_2")
net.transitions.add(t_1)
net.transitions.add(t_2)

# add arcs
petri_utils.add_arc_from_to(source, t_1, net)
petri_utils.add_arc_from_to(t_1, p_1, net)
petri_utils.add_arc_from_to(p_1, t_2, net)
petri_utils.add_arc_from_to(t_2, sink, net)

# add tokens
initial_marking = Marking()
initial_marking[source] = 1
final_marking = Marking()
final_marking[sink] = 1

# export and visualize
pnml_exporter.apply(net, initial_marking, "toyPetriNet.pnml", final_marking=final_marking)
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
#pn_visualizer.view(gviz)
#pn_visualizer.save(gviz, "toy-net.svg")

# simulation of running example
try:
    net, initial_marking, final_marking = pnml_importer.apply("runningExampleNet.pnml")
except:
    raise Exception("File not found")

simulated_log_basic = simulator.apply(net, initial_marking, variant=simulator.Variants.BASIC_PLAYOUT, parameters={
    simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: 50})
xes_exporter.apply(simulated_log_basic, 'sim_log_basic.xes')

# discover a data petri net
#log = xes_importer.apply("roadtraffic100traces.xes")
log = pm4py.read_xes('roadtraffic100traces.xes')
net, initial_marking, final_marking = inductive_miner.apply(log)
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)
pn_visualizer.save(gviz, "road-fine.svg")
pnml_exporter.apply(net, initial_marking, "doaf-fine.pnml", final_marking=final_marking)
dpnet, im, fm = decision_mining.create_data_petri_nets_with_decisions(log, net, initial_marking, final_marking)
#gviz = pn_visualizer.apply(dpnet, im, fm)
#pn_visualizer.view(gviz)
#simulated_log_dpnet = simulator.apply(dpnet, im, variant=simulator.Variants.BASIC_PLAYOUT, parameters={
#    simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: 50, 
#    simulator.Variants.BASIC_PLAYOUT.value.Parameters.PETRI_SEMANTICS: dpn_semantics.DataPetriNetSemantics})
e = {"dismissal_A": 0.5, "amount": 25.0, "totalPaymentAmount": 33.0, "dismissal_NIL": 0.5, "article": 175.0, 
        "vehicleClass_A": 0.5, "dismissal_I": 0.5, "points": 3.0}
# transform initial marking in nm DataMarking for data petri net
dm = DataMarking()
dm[list(im.keys())[0]] = im.get(list(im.keys())[0])
max_trace_length = 100
#breakpoint()
visited_elements = []
while dm != fm and len(visited_elements) < max_trace_length:
    #breakpoint()
    all_enabled_trans = dpn_semantics.enabled_transitions(dpnet, dm, e)
    print(all_enabled_trans)
    for enabled in list(all_enabled_trans):
        if "guard" in enabled.properties:
            if not dpn_semantics.evaluate_guard(enabled.properties["guard"], enabled.properties["readVariable"], dm.data_dict):
                all_enabled_trans.discard(enabled)
    #breakpoint()
    if len(all_enabled_trans) > 0:
        trans = choice(list(all_enabled_trans))
        dm = dpn_semantics.execute(trans, dpnet, dm, e)
        print(dm)
        visited_elements.append(trans)
    else:
        break
            
   

