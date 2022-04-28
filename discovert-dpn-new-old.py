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
from utils import get_place_from_event, delete_composite_loops
from utils import get_attributes_from_event, get_feature_names, update_places_map_dp_list_if_looping
from utils import extract_rules, get_map_transitions_events, get_input_near_source
from utils import get_map_events_transitions, update_dp_list, get_all_dp_from_event_to_sink
from DecisionTree import DecisionTree
from Loops import Loop

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
    #net, initial_marking, final_marking = pnml_importer.apply("models/{}.pnml".format(net_name))
    net, im, fm = pnml_importer.apply("models/{}.pnml".format(net_name))
except:
    raise Exception("File not found")
log = xes_importer.apply('data/log-{}.xes'.format(net_name))
#net, im, fm = pm4py.discover_petri_net_inductive(log)
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

trans_events_map = get_map_transitions_events(net)
events_trans_map = get_map_events_transitions(net)
#breakpoint()
loop_vertex = detect_loops(net)
#breakpoint()
loop_vertex = delete_composite_loops(loop_vertex)
loops = list()
for loop_length in loop_vertex.keys():
    for i, loop in enumerate(loop_vertex[loop_length]):
        events_loop = [events_trans_map[vertex] for vertex in loop if vertex in events_trans_map.keys()]
        loop_obj = Loop(loop, events_loop, net, "{}_{}".format(loop_length, i))
        loops.append(loop_obj)

breakpoint()
for loop in loops:
    loop.set_nearest_input(net, loops)
    loop.set_dp_forward_order_transition(net)
    #breakpoint()
    loop.set_dp_backward_order_transition(net)
# get the map of places and events
general_places_events_map = get_map_place_to_events(net, loops)
breakpoint()
# get the map of transitions and events
# get a dict of data for every decision point
decision_points_data = dict()
for place in net.places:
    if len(place.out_arcs) >= 2:
        decision_points_data[place.name] = pd.DataFrame()

# fill the data
breakpoint()
tic = time()
attributes_map = {'amount': 'continuous', 'policyType': 'categorical', 'appeal': 'boolean',
        'status': 'categorical', 'communication': 'categorical', 'discarded': 'boolean'}
for trace in log:
    breakpoint()
    places_events_map = general_places_events_map.copy()
    dp_list = list(places_events_map.keys())
    if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
        trace_attr_row = trace.attributes
        #trace_attr_row.pop('concept:name')
    last_k_events = list()
    transition_sequence = list()
    event_sequence = list()
    number_of_loops = dict()
    for loop in loops:
        loop.set_inactive()
        number_of_loops[loop.name] = 0
    breakpoint()
    for event in trace:
        trans_from_event = trans_events_map[event["concept:name"]]
        transition_sequence.append(trans_from_event)
        event_sequence.append(event['concept:name'])
        if len(last_k_events) == 0:
            last_event_name = 'None'
        else:
            last_event_name = last_k_events[-1]['concept:name']
        last_event_trans = trans_events_map[last_event_name]
        breakpoint()
        places_events_map, dp_list = update_places_map_dp_list_if_looping(
                net, dp_list, places_events_map, loops, event_sequence, number_of_loops, trans_events_map)
        places_from_event = get_place_from_event(places_events_map, event['concept:name'], dp_list)
        dp_list = update_dp_list(places_from_event, dp_list)
        breakpoint()
        if len(places_from_event) == 0:
            last_k_events.append(event)
            if len(last_k_events) > k:
                last_k_events = last_k_events[-k:]
            continue
        for place_from_event in places_from_event:
            last_k_event_dict = dict()
            for last_event in last_k_events:
                event_attr = get_attributes_from_event(last_event)
                event_attr.pop('time:timestamp')
                last_k_event_dict.update(event_attr)
            if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
                last_k_event_dict.update(trace_attr_row)
            last_k_event_dict.pop("concept:name")
            new_row = pd.DataFrame.from_dict(last_k_event_dict)
            new_row["target"] = place_from_event[1]
            decision_points_data[place_from_event[0]] = pd.concat([decision_points_data[place_from_event[0]], new_row], ignore_index=True)
        breakpoint()
        last_k_events.append(event)
        if len(last_k_events) > k:
            last_k_events = last_k_events[-k:]
    transition = [trans for trans in net.transitions if trans.label == event['concept:name']][0]
    breakpoint()
    places_from_event = get_all_dp_from_event_to_sink(transition, loops, [])
    #breakpoint()
    if len(places_from_event) > 0:
        #breakpoint()
        for place_from_event in places_from_event:
            last_k_event_dict = dict()
            for last_event in last_k_events:
                event_attr = get_attributes_from_event(last_event)
                event_attr.pop('time:timestamp')
                last_k_event_dict.update(event_attr)
            if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
                last_k_event_dict.update(trace_attr_row)
            last_k_event_dict.pop("concept:name")
            new_row = pd.DataFrame.from_dict(last_k_event_dict)
            new_row["target"] = place_from_event[1]
            decision_points_data[place_from_event[0]] = pd.concat([decision_points_data[place_from_event[0]], new_row], ignore_index=True)


breakpoint()

for decision_point in decision_points_data.keys():
    print("")
    print(decision_point)
    dataset = decision_points_data[decision_point]
    feature_names = get_feature_names(dataset)
    dt = DecisionTree(attributes_map)
    dt.fit(dataset)
    y_pred = dt.predict(dataset.drop(columns=['target']))
    print("Train accuracy: {}".format(metrics.accuracy_score(dataset['target'], y_pred)))
    print(dt.extract_rules())
toc = time()
print("Total time: {}".format(toc-tic))
