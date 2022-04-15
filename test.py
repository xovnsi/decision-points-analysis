import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.alpha import algorithm as aplha_miner
from pm4py.algo.decision_mining import algorithm as decision_mining

event_log = pm4py.read_xes('./data/roadtraffic100traces.xes')

net, im, fm = aplha_miner.apply(event_log)

decision_points = []
for place in net.places:
    if len(place.out_arcs) >= 2:
        decision_points.append(place.name)

print("\nDecision points: ", decision_points)

for dp in decision_points:
    print('\nDecision point ', dp)
    X, y, class_names = decision_mining.apply(event_log, net, im, fm, decision_point=dp)
    print(X)
