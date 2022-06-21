import pm4py
import os
from time import time
from tqdm import tqdm
import numpy as np
import copy
import pandas as pd
import argcomplete, argparse
import copy
import json
#from sklearn.tree import DecisionTreeClassifier, export_text
from pm4py.objects.petri_net.obj import Marking
from sklearn import metrics
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.decision_mining import algorithm as decision_mining
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.log.importer.xes import importer as xes_importer
from loop_utils import detect_loops, get_in_place_loop, get_out_place_loop
from loop_utils import simplify_net, delete_composite_loops
from utils import get_next_not_silent, get_map_place_to_events, get_place_from_event, shorten_rules_manually, \
    discover_overlapping_rules
from utils import get_attributes_from_event, get_feature_names, update_places_map_dp_list_if_looping
from utils import extract_rules, get_map_transitions_events, get_decision_points_and_targets
from utils import get_map_events_transitions, update_dp_list, get_all_dp_from_event_to_sink
from DecisionTree import DecisionTree
from Loops import Loop


def main():
    def ModelCompleter(**kwargs):
        return [name.split('.')[0] for name in os.listdir('models')]
    # Argument (verbose and net_name)
    parser = argparse.ArgumentParser()
    parser.add_argument("net_name", help="name of the petri net file (without extension)", type=str).completer = ModelCompleter
    argcomplete.autocomplete(parser)
    # parse arguments
    args = parser.parse_args()
    net_name = args.net_name
    attributes_map_file = f"{net_name}.attr"
    if attributes_map_file in os.listdir('dt-attributes'):
        with open(os.path.join('dt-attributes', attributes_map_file), 'r') as f:
            json_string = json.load(f)
            attributes_map = json.loads(json_string)
    else:
        raise FileNotFoundError('Create a configuration file for the decision tree before fitting it.')
    #try:
    #    #net, initial_marking, final_marking = pnml_importer.apply("models/{}.pnml".format(net_name))
    #    net, im, fm = pnml_importer.apply("models/{}.pnml".format(net_name))
    #except:
    #    print("Model not found")
    log = xes_importer.apply('data/log-{}.xes'.format(net_name))
    trace_attributes = dict()
    event_attributes = dict()
    for trace in log:
        for attribute in trace.attributes:
            if attribute in trace_attributes.keys():
                trace_attributes[attribute].append(trace.attributes[attribute])
            else:
                trace_attributes[attribute] = [trace.attributes[attribute]]
        for event in trace:
            for attribute in event.keys():
                if attribute in event_attributes.keys():
                    event_attributes[attribute].append(event[attribute])
                else:
                    event_attributes[attribute] = [event[attribute]]
    net, im, fm = pm4py.discover_petri_net_inductive(log)
    sink_complete_net = [place for place in net.places if place.name == 'sink'][0]
    sim_net = copy.deepcopy(net)
    sim_net, sim_map, last_collapsed_left, last_collapsed_right, parallel_branches = simplify_net(sim_net)
    im_sim = Marking()
    fm_sim = Marking()
    for place in sim_net.places:
        if place.name == 'source':
            source = place
        elif place.name == 'sink':
            sink = place
        else:
            if place.name in sim_map.keys():
                if 'sink' in sim_map[place.name]:
                    sink = place
                elif 'source' in sim_map[place.name]:
                    source = place
    im_sim[source] = 1
    fm_sim[sink] = 1
    gviz = pn_visualizer.apply(net, im, fm)
    pn_visualizer.view(gviz)
    gviz = pn_visualizer.apply(sim_net, im_sim, fm_sim)
    pn_visualizer.view(gviz)
    # make sure that continuous data are float
    # TODO do it following the conf file of the tree
    for trace in tqdm(log):
        for event in trace:
            for attr in event.keys():
                if not isinstance(event[attr], bool):
                    try:
                        event[attr] = float(event[attr])
                    except:
                        pass

    trans_events_map = get_map_transitions_events(net)
    events_trans_map = get_map_events_transitions(net)

    # detect loops and create loop objects
    loop_vertex = detect_loops(sim_net)
    loop_vertex = delete_composite_loops(loop_vertex)

    loops = list()
    for loop_length in loop_vertex.keys():
        for i, loop in enumerate(loop_vertex[loop_length]):
            events_loop = [events_trans_map[vertex] for vertex in loop if vertex in events_trans_map.keys()]
            loop_obj = Loop(loop, events_loop, sim_net, "{}_{}".format(loop_length, i))
            loop_obj.set_complete_net_input(last_collapsed_left)
            loop_obj.set_complete_net_output(last_collapsed_right)
            loop_obj.set_complete_net_loop_nodes(sim_map)
            loops.append(loop_obj)

    # join different parts of the same loop (different loops with same inputs and outputs)
    not_joinable = False
    while not not_joinable:
        old_loops = copy.copy(loops)
        new_loop_nodes = None
        new_loop_events = None
        for loop in old_loops:
            new_loop_nodes = set()
            new_loop_events = set()
            for loop_inner in old_loops:
                if loop != loop_inner:
                    if loop.input_places_complete == loop_inner.input_places_complete and loop.output_places_complete == loop_inner.output_places_complete:
                        new_loop_nodes = new_loop_nodes.union(set(loop.vertex))
                        new_loop_nodes = new_loop_nodes.union(set(loop_inner.vertex))
                        new_loop_events = new_loop_events.union(set(loop.events))
                        new_loop_events = new_loop_events.union(set(loop_inner.events))
                        loops.remove(loop_inner)

            if len(new_loop_nodes) > 0:
                new_loop = Loop(list(new_loop_nodes), list(new_loop_events), sim_net, loop.name)
                new_loop.set_complete_net_input(last_collapsed_left)
                new_loop.set_complete_net_output(last_collapsed_right)
                new_loop.set_complete_net_loop_nodes(sim_map)
                new_loop.set_complete_net_loop_events(events_trans_map)
                loops.append(new_loop)
                loops.remove(loop)
                break
        if len(new_loop_nodes) == 0:
            not_joinable = True

    for loop in loops:
        loop.set_nearest_input_complete_net(sim_net, sim_map)

    tic = time()

    # Dictionary of attributes data for every decision point, with a target key to be used later on
    decision_points_data = dict()
    for place in net.places:
        if len(place.out_arcs) >= 2:
            decision_points_data[place.name] = {k: [] for k in ['target']}

    #breakpoint()
    event_attr = dict()
    stored_dicts = dict()

    # Scanning the log to get the data
    print('Extracting training data from Event Log')
    #for trace in tqdm(log):
    variants = variants_filter.get_variants(log)
    # searching only the variants
    for variant in tqdm(variants):
        transitions_sequence = list()
        events_sequence = list()
        dp_events_sequence = dict()
        for i, event_name in enumerate(variant.split(',')):
            trans_from_event = trans_events_map[event_name]
            transitions_sequence.append(trans_from_event)
            events_sequence.append(event_name)
            dp_dict = dict()
            if len(transitions_sequence) > 1:
                dp_dict, stored_dicts = get_decision_points_and_targets(transitions_sequence, loops, net, parallel_branches, stored_dicts)
            old_keys = list(dp_dict.keys()).copy()
            for dp in old_keys:
                if len(dp_dict[dp]) == 0:
                    del dp_dict[dp]
            '''
            if len(transitions_sequence) > 1:
                print("Previous event: {}".format(events_sequence[-2]))
                print("Current event: {}".format(events_sequence[-1]))
                print("DPs")
                for key in dp_dict.keys():
                    print(" - {}".format(key))
                    for inn_key in dp_dict[key]:
                        if inn_key in events_trans_map.keys():
                            print("   - {}".format(events_trans_map[inn_key]))
                        else:
                            print("   - {}".format(inn_key))
                #breakpoint()
            '''
            dp_events_sequence['Event_{}'.format(i+1)] = dp_dict
        # Final update of the current trace (from last event to sink)
        transition = [trans for trans in net.transitions if trans.label == event_name][0]

        places_from_event = get_all_dp_from_event_to_sink(transition, sink_complete_net)
        dp_events_sequence['End'] = places_from_event
        transition_sequence = list()
        event_sequence = list()
        # Keep the same attributes observed in the previous traces (to keep dictionaries at the same length)
        event_attr = {k: [np.nan] for k in event_attr.keys()}
        # Store the trace attributes (if any)
        if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
            event_attr.update(trace.attributes)
        last_event_name = 'None'

        for trace in variants[variant]:
            for i, event in enumerate(trace):
                trans_from_event = trans_events_map[event["concept:name"]]
                transition_sequence.append(trans_from_event)
                event_sequence.append(event['concept:name'])
                if len(transition_sequence) > 1:
                    dp_dict = dp_events_sequence['Event_{}'.format(i+1)]
                    for dp in dp_dict.keys():
                        # Append the last attribute values to the decision point dictionary
                        for dp_target in dp_dict[dp]:
                            for a in event_attr.keys():
                                # If attribute is not present, and it is not nan, add it as a new key, filling previous entries with nan
                                #if a not in decision_points_data[place_from_event[0]] and event_attr[a][0] is not np.nan:
                                if a not in decision_points_data[dp] and event_attr[a][0] is not np.nan:
                                    entries = len(decision_points_data[dp]['target'])
                                    #entries = len(decision_points_data[place_from_event[0]]['target'])
                                    #decision_points_data[place_from_event[0]][a] = [np.nan] * entries
                                    decision_points_data[dp][a] = [np.nan] * entries
                                    #decision_points_data[place_from_event[0]][a].append(event_attr[a][0])
                                    decision_points_data[dp][a].append(event_attr[a][0])
                                # Else, if attribute is present, just append it to the existing list
                                #elif a in decision_points_data[place_from_event[0]]:
                                elif a in decision_points_data[dp]:
                                    #decision_points_data[place_from_event[0]][a].append(event_attr[a][0])
                                    decision_points_data[dp][a].append(event_attr[a][0])
                            # Append also the target transition label to the decision point dictionary
                            decision_points_data[dp]['target'].append(dp_target)

                # Get the attribute values dictionary containing the current event values
                event_attr.update(get_attributes_from_event(event))
                [event_attr.pop(k) for k in ['time:timestamp', 'concept:name']]
                # Update the last event name with the current event name
                last_event_name = event['concept:name']

            places_from_event = dp_events_sequence['End']
            if len(places_from_event) > 0:
                for place_from_event in places_from_event.keys():
                    for target_act in places_from_event[place_from_event]:
                        for a in event_attr.keys():
                            if a not in decision_points_data[place_from_event] and event_attr[a][0] is not np.nan:
                                entries = len(decision_points_data[place_from_event]['target'])
                                decision_points_data[place_from_event][a] = [np.nan] * entries
                                decision_points_data[place_from_event][a].append(event_attr[a][0])
                            elif a in decision_points_data[place_from_event]:
                                decision_points_data[place_from_event][a].append(event_attr[a][0])
                        decision_points_data[place_from_event]['target'].append(target_act)

#    if 'road' in net_name or 'Road' in net_name:
#        attributes_map = {'lifecycle:transition': 'categorical', 'expense': 'continuous',
#                          'totalPaymentAmount': 'continuous', 'paymentAmount': 'continuous', 'amount': 'continuous',
#                          'org:resource': 'categorical', 'dismissal': 'categorical', 'vehicleClass': 'categorical',
#                          'article': 'categorical', 'points': 'continuous', 'notificationType': 'categorical',
#                          'lastSent': 'categorical', 'matricola': 'categorical'}
#    else:
#        attributes_map = {'amount': 'continuous', 'policyType': 'categorical', 'appeal': 'boolean', 'status': 'categorical',
#                          'communication': 'categorical', 'discarded': 'boolean'}

    # For each decision point, create a dataframe, fit a decision tree and print the extracted rules
    file_name = 'results.txt'
    for decision_point in decision_points_data.keys():
        print("\nDecision point: {}".format(decision_point))
        dataset = pd.DataFrame.from_dict(decision_points_data[decision_point])

        # Replacing ':' with '_' both in the dataset columns and in the attributes map since ':' creates problems
        dataset.columns = dataset.columns.str.replace(':', '_')
        attributes_map = {k.replace(':', '_'): attributes_map[k] for k in attributes_map}

        # TODO conversion of the dataset should be done only once at the beginning: check if it is done in other places
        for attr in dataset.columns:
            if attr != 'target':
                if attributes_map[attr] == 'continuous':
                    dataset[attr] = dataset[attr].astype(float)
                elif attributes_map[attr] == 'boolean':
                    dataset[attr] = dataset[attr].astype(bool)
                elif attributes_map[attr] == 'categorical':
                    dataset[attr] = dataset[attr].astype(pd.StringDtype())

        # Discovering branching conditions with Daikon - comment these four lines to go back to decision tree + pruning
        # rules = discover_branching_conditions(dataset)
        # rules = {k: rules[k].replace('_', ':') for k in rules}
        # print(rules)
        # continue

        print("Fitting a decision tree on the decision point's dataset...")
        dt = DecisionTree(attributes_map)
        dt.fit(dataset)

        if not len(dt.get_nodes()) == 1:
            print("Training complete. Extracting rules...")
            with open(file_name, 'a') as f:
                f.write('{} - SUCCESS\n'.format(decision_point))
                lf = len(dataset[dataset['target'].str.startswith(('skip', 'tauJoin', 'tauSplit', 'init_loop'))])
                f.write('Rows with invisible activity as target: {}/{}\n'.format(str(lf), str(len(dataset))))

                # Predict (just to see the accuracy)
                # y_pred = dt.predict(dataset.drop(columns=['target']))
                # print("Train accuracy: {}".format(metrics.accuracy_score(dataset['target'], y_pred)))

                rules = dt.extract_rules()

                # Rule extraction with pruning and overlapping rules discovery
                # rules = dt.extract_rules_with_pruning(dataset)
                # rules = discover_overlapping_rules(dt, dataset, attributes_map, rules)

                rules = shorten_rules_manually(rules, attributes_map)
                rules = {k: rules[k].replace('_', ':') for k in rules}

                f.write('Rules:\n')
                for k in rules:
                    f.write('{}: {}\n'.format(k, rules[k]))
                f.write('\n')
            print(rules)
        else:
            with open(file_name, 'a') as f:
                f.write('{} - FAIL\n'.format(decision_point))
                lf = len(dataset[dataset['target'].str.startswith(('skip', 'tauJoin', 'tauSplit', 'init_loop'))])
                f.write('Rows with actual activity as target: {}/{}\n\n'.format(str(lf), str(len(dataset))))

    toc = time()
    print("\nTotal time: {}".format(toc-tic))


if __name__ == '__main__':
    main()
