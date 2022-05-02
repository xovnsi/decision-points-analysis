import pm4py
import os
import argcomplete, argparse
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.statistics.variants.log import get as variants_module
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.decision_mining.algorithm import simplify_token_replay
from utils import *
from Loops import Loop
from pm4py.algo.conformance.alignments.petri_net import algorithm as ali
from DecisionTree import DecisionTree
from sklearn import metrics

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
event_log = xes_importer.apply('data/log-{}.xes'.format(net_name))
#net, im, fm = pm4py.discover_petri_net_inductive(log)
for trace in event_log:
    for event in trace:
        for attr in event.keys():
           # if attr == 'appeal':
           #     breakpoint()
            if not isinstance(event[attr], bool):
                try:
                    event[attr] = float(event[attr])
                except:
                    pass

# Extract the process model
#net, im, fm = alpha_miner.apply(event_log)
net, im, fm = pm4py.discover_petri_net_inductive(event_log)

# Get all the useful data structures to handle loops etc.
trans_events_map = get_map_transitions_events(net)
events_trans_map = get_map_events_transitions(net)
loop_vertex = detect_loops(net)
loop_vertex = delete_composite_loops(loop_vertex)
loops = list()
for loop_length in loop_vertex.keys():
    for i, loop in enumerate(loop_vertex[loop_length]):
        events_loop = [events_trans_map[vertex] for vertex in loop if vertex in events_trans_map.keys()]
        loop_obj = Loop(loop, events_loop, net, "{}_{}".format(loop_length, i))
        loops.append(loop_obj)

for loop in loops:
    loop.set_nearest_input(net, loops)
    loop.set_dp_forward_order_transition(net)
    loop.set_dp_backward_order_transition(net)

# Get the map of places and events
general_places_events_map = get_map_place_to_events(net, loops)

# get the map of transitions and events
# get a dict of data for every decision point
decision_points_data = dict()
for place in net.places:
    if len(place.out_arcs) >= 2:
        decision_points_data[place.name] = pd.DataFrame()

# Get the variants
variants_idxs = variants_module.get_variants_from_log_trace_idx(event_log)
one_variant = []
for variant in variants_idxs:
    one_variant.append(variant)

replay_result = token_replay.apply(event_log, net, im, fm)
replay_result = simplify_token_replay(replay_result)

# Get the data using alignments when needed
count = 0   # variants counter

for variant in replay_result:
    if variant['trace_fitness'] == 1.0:
        for trace_index in variants_idxs[one_variant[count]]:
            trace = event_log[trace_index]
            # breakpoint()
            places_events_map = general_places_events_map.copy()
            dp_list = list(places_events_map.keys())
            if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
                trace_attr_row = trace.attributes
                # trace_attr_row.pop('concept:name')
            last_k_events = list()
            transition_sequence = list()
            event_sequence = list()
            number_of_loops = dict()
            for loop in loops:
                loop.set_inactive()
                number_of_loops[loop.name] = 0
            # breakpoint()

            for event in trace:
                trans_from_event = trans_events_map[event["concept:name"]]
                transition_sequence.append(trans_from_event)
                event_sequence.append(event['concept:name'])
                if len(last_k_events) == 0:
                    last_event_name = 'None'
                else:
                    last_event_name = last_k_events[-1]['concept:name']
                last_event_trans = trans_events_map[last_event_name]
                # breakpoint()
                places_events_map, dp_list = update_places_map_dp_list_if_looping(
                    net, dp_list, places_events_map, loops, event_sequence, number_of_loops, trans_events_map)
                places_from_event = get_place_from_event(places_events_map, event['concept:name'], dp_list)
                dp_list = update_dp_list(places_from_event, dp_list)
                # breakpoint()
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
                    decision_points_data[place_from_event[0]] = pd.concat(
                        [decision_points_data[place_from_event[0]], new_row], ignore_index=True)
                # breakpoint()
                last_k_events.append(event)
                if len(last_k_events) > k:
                    last_k_events = last_k_events[-k:]

            transition = [trans for trans in net.transitions if trans.label == event['concept:name']][0]
            # breakpoint()
            places_from_event = get_all_dp_from_event_to_sink(transition, loops, [])
            # breakpoint()
            if len(places_from_event) > 0:
                # breakpoint()
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

    else:
        example_trace = event_log[variants_idxs[one_variant[count]][0]]
        alignment = ali.apply(example_trace, net, im, fm)['alignment']
        for trace_index in variants_idxs[one_variant[count]]:
            trace = event_log[trace_index]
            # breakpoint()
            places_events_map = general_places_events_map.copy()
            dp_list = list(places_events_map.keys())
            if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
                trace_attr_row = trace.attributes
                # trace_attr_row.pop('concept:name')
            last_k_events = list()
            transition_sequence = list()
            event_sequence = list()
            number_of_loops = dict()
            last_k_event_dict = dict()
            for loop in loops:
                loop.set_inactive()
                number_of_loops[loop.name] = 0
            # breakpoint()
            j = 0   # log event index

            for el in alignment:
                if el[1] != '>>':   # If move in model
                    trans_from_event = trans_events_map[el[1]]
                    transition_sequence.append(trans_from_event)
                    event_sequence.append(el[1])
                    if len(last_k_events) == 0:
                        last_event_name = 'None'
                    else:
                        last_event_name = last_k_events[-1]['concept:name']
                    last_event_trans = trans_events_map[last_event_name]
                    # breakpoint()
                    places_events_map, dp_list = update_places_map_dp_list_if_looping(net, dp_list, places_events_map, loops, event_sequence, number_of_loops, trans_events_map)
                    places_from_event = get_place_from_event(places_events_map, el[1], dp_list)
                    dp_list = update_dp_list(places_from_event, dp_list)
                    for place_from_event in places_from_event:
                        if len(last_k_event_dict) > 0:
                            new_row = pd.DataFrame.from_dict(last_k_event_dict)
                            new_row["target"] = place_from_event[1]
                            decision_points_data[place_from_event[0]] = pd.concat([decision_points_data[place_from_event[0]], new_row], ignore_index=True)
                if el[0] != '>>' and el[1] != '>>':     # If move in log and model
                    last_k_event_dict = dict()
                    for last_event in last_k_events:
                        event_attr = get_attributes_from_event(last_event)
                        event_attr.pop('time:timestamp')
                        last_k_event_dict.update(event_attr)
                    if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
                        last_k_event_dict.update(trace_attr_row)
                    last_k_event_dict.pop("concept:name")
                if el[0] != '>>':   # If move in log, go to the next log event
                    last_event_name = trace[j]['concept:name']
                    j += 1
            # Final update of the current trace (from last event to sink)
            transition = [trans for trans in net.transitions if trans.label == el[1]][0]
            # breakpoint()
            places_from_event = get_all_dp_from_event_to_sink(transition, loops, [])
            # breakpoint()
            if len(places_from_event) > 0:
                # breakpoint()
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

    count += 1

attributes_map = {'lifecycle.transition': 'categorical', 'expense': 'continuous',
                  'totalPaymentAmount': 'continuous', 'paymentAmount': 'continuous', 'amount': 'continuous',
                  'org.resource': 'categorical', 'dismissal': 'categorical', 'vehicleClass': 'categorical',
                  'article': 'categorical', 'points': 'continuous', 'notificationType': 'categorical',
                  'lastSent': 'categorical'}

attributes_map = {'amount': 'continuous', 'policyType': 'categorical', 'appeal': 'boolean', 'status': 'categorical',
                  'communication': 'categorical', 'discarded': 'boolean'}

# For each decision point (with values for at least one attribute, apart from the 'target' attribute)
# create a dataframe, fit a decision tree and print the extracted rules
for decision_point in decision_points_data.keys():
    print("\n", decision_point)
    dataset = pd.DataFrame.from_dict(decision_points_data[decision_point]).fillna('?')
    dataset.columns = dataset.columns.str.replace(':', '.')
    feature_names = get_feature_names(dataset)
    dt = DecisionTree(attributes_map)
    dt.fit(dataset)
    if not len(dt.get_nodes()) == 1:
        y_pred = dt.predict(dataset.drop(columns=['target']))
        print("Train accuracy: {}".format(metrics.accuracy_score(dataset['target'], y_pred)))
        print(dt.extract_rules())
