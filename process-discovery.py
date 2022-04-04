from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer

# import log, discover process, visualize and save petri net
log = xes_importer.apply("running-example.xes")
net, initial_marking, final_marking = inductive_miner.apply(log)
pnml_exporter.apply(net, initial_marking, "runningExampleNet.pnml", final_marking=final_marking)
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)
pn_visualizer.save(gviz, "running-example-net.svg")

print("\nPLACES")
for place in net.places:
    print(place)

print("\nTRANSITIONS")
for transition in net.transitions:
    print(transition)

print("\nARCS")
for arc in net.arcs:
    print(arc)
