import pm4py
import pandas as pd
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.decision_mining import algorithm as decision_mining
from DecisionTree import DecisionTree
from sklearn import metrics

event_log = pm4py.read_xes('./data/log-running-example-Will-BPM-silent-loops-silent-loopB.xes')

attributes_map = {'policyType_normal': 'categorical', 'policyType_premium': 'categorical'}

#net, im, fm = alpha_miner.apply(event_log)
net, im, fm = pm4py.discover_petri_net_inductive(event_log)

decision_points = []
for place in net.places:
    if len(place.out_arcs) >= 2:
        decision_points.append(place.name)

print("\nDecision points: ", decision_points)

for dp in decision_points:
    print('\nDecision point ', dp)
    X, y, class_names = decision_mining.apply(event_log, net, im, fm, decision_point=dp)

    y_df = y.to_frame(name='target')
    class_names_dict = {k: class_names[k] for k in range(0, len(class_names))}
    y_df['target'] = y_df['target'].map(arg=class_names_dict, na_action='ignore')
    dataset = pd.concat([X, y_df], axis=1)
    print(dataset)

    #dt = DecisionTree(attributes_map)
    #dt.fit(dataset)
    #y_pred = dt.predict(dataset)
    #print("Train accuracy: {}".format(metrics.accuracy_score(dataset['target'], y_pred)))
    #print(dt.extract_rules())
