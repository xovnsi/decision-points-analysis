import pm4py
from pm4py.objects.petri_net.importer import importer as pnml_importer
from time import time
from pm4py.algo.decision_mining import algorithm as decision_mining
#log = pm4py.read_xes('data/log-roadtraffic100traces.xes')
log = pm4py.read_xes('data/log-running-example-Will-BPM-silent-loops-silent-loopB.xes')
tic = time()
for trace in log:
    for event in trace:
        for attr in event.keys():
           # if attr == 'appeal':
           #     breakpoint()
            if not isinstance(event[attr], bool):
                try:
                    event[attr] = float(event[attr])
                except:
                    pass
breakpoint()
net, im, fm = pm4py.discover_petri_net_inductive(log)
net, im, fm = decision_mining.create_data_petri_nets_with_decisions(log, net, im, fm)
#net, im, fm = pnml_importer.apply("models/running-example-Will-BPM-silent-loops-silent-loopB.pnml")
for t in net.transitions:
    if "guard" in t.properties:
        print("")
        print(t)
        print(t.properties["guard"])
toc = time()
print("Total time: {}".format(toc-tic))

