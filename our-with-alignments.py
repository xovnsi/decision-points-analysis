import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.statistics.variants.log import get as variants_module
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from utils import *
from pm4py.algo.conformance.alignments.petri_net import algorithm as ali
from DecisionTree import DecisionTree
from sklearn import metrics

# Import the log
event_log = pm4py.read_xes('./data/roadtraffic100traces.xes')

# Extract the process model
net, im, fm = alpha_miner.apply(event_log)

# Get all the useful data structures to handle loops etc.
loop_vertex = detect_loops(net)
in_places_loops = get_in_place_loop(net, loop_vertex)
in_transitions_loop = list()
for place_name in in_places_loops:
    place = [place_net for place_net in net.places if place_net.name == place_name]
    place = place[0]
    for out_arc in place.out_arcs:
        out_transitions = get_next_not_silent(place, [], loop_vertex)
        in_transitions_loop.extend(out_transitions)
out_places_loops = get_out_place_loop(net, loop_vertex)
trans_events_map = get_map_transitions_events(net)
places_events_map = get_map_place_to_events(net, loop_vertex)

# Attributes map
attributes_map = {'amount': 'continuous', 'policyType': 'categorical', 'appeal': 'boolean', 'status': 'categorical',
                  'communication': 'categorical', 'discarded': 'boolean'}

# Dictionary of attributes data for every decision point, with a target key to be used later on
decision_points_data = dict()
for place in net.places:
    if len(place.out_arcs) >= 2:
        decision_points_data[place.name] = {k: [] for k in ['target']}

# Get the variants
variants_idxs = variants_module.get_variants_from_log_trace_idx(event_log)
one_variant = []
for variant in variants_idxs:
    one_variant.append(variant)

replay_result = token_replay.apply(event_log, net, im, fm)

variant = {}
for element in replay_result:
    if tuple(element['activated_transitions']) not in variant:
        variant[tuple(element['activated_transitions'])] = True
smaller_replay = []
for element in replay_result:
    if variant[tuple(element['activated_transitions'])]:
        smaller_replay.append(element)
        variant[tuple(element['activated_transitions'])] = False

# Get the data using alignments when needed
count = 0   # variants counter
event_attr = dict()

for variant in smaller_replay:
    if variant['trace_fitness'] == 1.0:
        for trace_index in variants_idxs[one_variant[count]]:
            trace = event_log[trace_index]
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
                        decision_points_data[place_from_event[0]][a].append(event_attr[a][0])  # 0 to avoid nested lists
                    # Append also the target transition label to the decision point dictionary
                    decision_points_data[place_from_event[0]]['target'].append(place_from_event[1])

                # Get the attribute values dictionary containing the current event values
                event_attr.update(get_attributes_from_event(event))
                [event_attr.pop(k) for k in ['time:timestamp', 'concept:name']]
                # Update the last event name with the current event name
                last_event_name = event['concept:name']
    else:
        example_trace = event_log[variants_idxs[one_variant[count]][0]]
        alignment = ali.apply(example_trace, net, im, fm)['alignment']
        for trace_index in variants_idxs[one_variant[count]]:
            trace = event_log[trace_index]
            # Keep the same attributes observed in the previous traces (to keep dictionaries at the same length)
            event_attr = {k: ['NIL'] for k in event_attr.keys()}
            # Store the trace attributes (if any)
            if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
                event_attr.update(trace.attributes)
            last_event_name = 'None'
            j = 0   # log event index
            for el in alignment:
                if el[1] != '>>':   # If move in model
                    trans_from_event = trans_events_map[el[1]]
                    last_event_trans = trans_events_map[last_event_name]
                    # Places that are interested in the current transition
                    places_from_event = get_place_from_transition(places_events_map, el[1], loop_vertex,
                                                                  last_event_trans, in_transitions_loop,
                                                                  out_places_loops,
                                                                  in_places_loops, trans_from_event)

                    for place_from_event in places_from_event:
                        # Append the last attribute values to the decision point dictionary
                        for a in event_attr.keys():
                            # If attribute is not present, add it as a new key, filling the previous entries with NIL
                            if a not in decision_points_data[place_from_event[0]]:
                                entries = len(decision_points_data[place_from_event[0]]['target'])
                                decision_points_data[place_from_event[0]][a] = ['NIL'] * entries
                            # In every case, append the new value
                            decision_points_data[place_from_event[0]][a].append(event_attr[a][0])  # 0 avoid nested list
                        # Append also the target transition label to the decision point dictionary
                        decision_points_data[place_from_event[0]]['target'].append(place_from_event[1])
                if el[0] != '>>' and el[1] != '>>':     # If move in log and model
                    # Get the attribute values dictionary containing the current event values
                    event_attr.update(get_attributes_from_event(trace[j]))
                    [event_attr.pop(k) for k in ['time:timestamp', 'concept:name']]
                if el[0] != '>>':   # If move in log, go to the next log event
                    last_event_name = trace[j]['concept:name']
                    j += 1
    count += 1

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
