import pm4py
import os
from time import time
import numpy as np
import copy
import pandas as pd
import argcomplete, argparse
#from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import metrics
from pm4py.algo.decision_mining import algorithm as decision_mining
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.log.importer.xes import importer as xes_importer
from utils import detect_loops, get_in_place_loop, get_next_not_silent
from utils import get_out_place_loop, get_map_place_to_events
from utils import get_place_from_transition
from utils import get_attributes_from_event, get_feature_names
from utils import extract_rules, get_map_transitions_events
from DecisionTree import DecisionTree

def ModelCompleter(**kwargs):
    return [name.split('.')[0] for name in os.listdir('models')]
# Argument (verbose and net_name)
parser = argparse.ArgumentParser()
parser.add_argument("net_name", help="name of the petri net file (without extension)", type=str).completer = ModelCompleter
argcomplete.autocomplete(parser)
# parse arguments
args = parser.parse_args()
net_name = args.net_name
k = 1

try:
    net, initial_marking, final_marking = pnml_importer.apply("models/{}.pnml".format(net_name))
except:
    raise Exception("File not found")
log = xes_importer.apply('data/log-{}.xes'.format(net_name))
net, im, fm = pm4py.discover_petri_net_inductive(log)
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

#breakpoint()
loop_vertex = detect_loops(net)
in_places_loops = get_in_place_loop(net, loop_vertex)
in_transitions_loop = list()        
for place_name in in_places_loops:
    #breakpoint()
    place = [place_net for place_net in net.places if place_net.name == place_name]
    place = place[0]
    for out_arc in place.out_arcs:
        out_transitions = get_next_not_silent(place, [], loop_vertex) 
        in_transitions_loop.extend(out_transitions)
out_places_loops = get_out_place_loop(net, loop_vertex)
#breakpoint()
# get the map of places and events
places_events_map = get_map_place_to_events(net, loop_vertex)
# get the map of transitions and events
trans_events_map = get_map_transitions_events(net)

# Fill the data
tic = time()
attributes_map = {'amount': 'continuous', 'policyType': 'categorical', 'appeal': 'boolean', 'status': 'categorical',
                  'communication': 'categorical', 'discarded': 'boolean'}

# Dictionary of attributes data for every decision point, with a target key to be used later on
decision_points_data = dict()
for place in net.places:
    if len(place.out_arcs) >= 2:
        decision_points_data[place.name] = {k: [] for k in ['target']}

event_attr = dict()

# Scanning the log to get the data
for trace in log:
    # Keep the same attributes observed in the previous traces (to keep dictionaries at the same length)
    event_attr = {k: ['NIL'] for k in event_attr.keys()}
    # Store the trace attributes (if any)
    if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
        event_attr.update(trace.attributes)
    last_event_name = 'None'
    for event in trace:
        trans_from_event = trans_events_map[event["concept:name"]]
        last_event_trans = trans_events_map[last_event_name]
        # Places that are interested in the current transition
        places_from_event = get_place_from_transition(places_events_map, event['concept:name'], loop_vertex,
                                                      last_event_trans, in_transitions_loop, out_places_loops,
                                                      in_places_loops, trans_from_event)

        for place_from_event in places_from_event:
            # Append the last attribute values to the decision point dictionary
            for a in event_attr.keys():
                # If the attribute is not present, add it as a new key, filling the previous entries with NIL
                if a not in decision_points_data[place_from_event[0]]:
                    entries = len(decision_points_data[place_from_event[0]]['target'])
                    decision_points_data[place_from_event[0]][a] = ['NIL'] * entries
                # In every case, append the new value
                decision_points_data[place_from_event[0]][a].append(event_attr[a][0])   # index 0 to avoid nested lists
            # Append also the target transition label to the decision point dictionary
            decision_points_data[place_from_event[0]]['target'].append(place_from_event[1])

        # Get the attribute values dictionary containing the current event values
        event_attr.update(get_attributes_from_event(event))
        [event_attr.pop(k) for k in ['time:timestamp', 'concept:name']]
        # Update the last event name with the current event name
        last_event_name = event['concept:name']

# For each decision point, create a dataframe, fit a decision tree and print the extracted rules
for decision_point in decision_points_data.keys():
    print("\n", decision_point)
    dataset = pd.DataFrame.from_dict(decision_points_data[decision_point])
    feature_names = get_feature_names(dataset)
    dt = DecisionTree(attributes_map)
    dt.fit(dataset)
    y_pred = dt.predict(dataset.drop(columns=['target']))
    print("Train accuracy: {}".format(metrics.accuracy_score(dataset['target'], y_pred)))
    print(dt.extract_rules())

toc = time()
print("Total time: {}".format(toc-tic))
