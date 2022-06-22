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
from pm4py.objects.petri_net.obj import Marking
from sklearn import metrics
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.log.importer.xes import importer as xes_importer
from loop_utils import detect_loops
from loop_utils import simplify_net, delete_composite_loops
from utils import shorten_rules_manually, discover_overlapping_rules
from utils import get_attributes_from_event
from utils import extract_rules, get_map_transitions_events, get_decision_points_and_targets
from utils import get_map_events_transitions, get_all_dp_from_event_to_sink
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

    # Importing the log
    log = xes_importer.apply('data/log-{}.xes'.format(net_name))
    # TODO this dictionaries are not used in the end: could they be removed?
    trace_attributes, event_attributes = dict(), dict()
    for trace in log:
        for attribute in trace.attributes:
            if attribute not in trace_attributes.keys():
                trace_attributes[attribute] = list()
            trace_attributes[attribute].append(trace.attributes[attribute])
        for event in trace:
            for attribute in event.keys():
                if attribute not in event_attributes.keys():
                    event_attributes[attribute] = list()
                event_attributes[attribute].append(event[attribute])

    # Importing the attributes_map file
    attributes_map_file = f"{net_name}.attr"
    if attributes_map_file in os.listdir('dt-attributes'):
        with open(os.path.join('dt-attributes', attributes_map_file), 'r') as f:
            json_string = json.load(f)
            attributes_map = json.loads(json_string)
    else:
        raise FileNotFoundError('Create a configuration file for the decision tree before fitting it.')

    # Converting attributes types according to the attributes_map file
    for trace in log:
        for event in trace:
            for attribute in event.keys():
                if attribute in attributes_map:
                    if attributes_map[attribute] == 'continuous':
                        event[attribute] = float(event[attribute])
                    elif attributes_map[attribute] == 'boolean':
                        event[attribute] = bool(event[attribute])

    # Importing the Petri net model, if it exists
    try:
        net, im, fm = pnml_importer.apply("models/{}.pnml".format(net_name))
    except FileNotFoundError:
        print("Model not found")

    # Otherwise, extract the Petri net from the log using the inductive miner #TODO alternative if model not present?
    #net, im, fm = pm4py.discover_petri_net_inductive(log)

    # Generating the simplified Petri net
    sink_complete_net = [place for place in net.places if place.name == 'sink'][0]
    sim_net = copy.deepcopy(net)
    sim_net, sim_map, last_collapsed_left, last_collapsed_right, parallel_branches = simplify_net(sim_net)
    im_sim, fm_sim = Marking(), Marking()
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
    #pn_visualizer.view(gviz)
    gviz = pn_visualizer.apply(sim_net, im_sim, fm_sim)
    #pn_visualizer.view(gviz)

    # Dealing with loops and other stuff... needs cleaning
    trans_events_map = get_map_transitions_events(net)
    events_trans_map = get_map_events_transitions(net)

    # Reachable activities MANUALLY for running-example-paper
    reachable_activities = dict()
    if net_name == 'running-example-paper':
        reachable_activities['Request loan'] = {'Register', 'Check', 'Prepare documents', 'Final check', "Don't authorize", 'Authorize'}
        reachable_activities['Register'] = {'Check', 'Prepare documents', 'Final check', "Don't authorize", 'Authorize'}
        reachable_activities['Check'] = {'Check', 'Final check', "Don't authorize", 'Authorize'}
        reachable_activities['Prepare documents'] = {'Prepare documents', 'Final check', "Don't authorize", 'Authorize'}
        reachable_activities['Final check'] = {'Check', 'Prepare documents', 'Final check', "Don't authorize", 'Authorize'}
        reachable_activities["Don't authorize"] = {}
        reachable_activities['Authorize'] = {}
    elif net_name == 'Road_Traffic_Fine_Management_Process':
        reachable_activities['Create Fine'] = {'Send Appeal to Prefecture', 'Payment', 'Insert Fine Notification', 'Send Fine',
                                               'Insert Date Appeal to Prefecture', 'Appeal to Judge', 'Receive Result Appeal from Prefecture',
                                               'Notify Result Appeal to Offender', 'Add penalty', 'Send for Credit Collection'}
        reachable_activities['Send Appeal to Prefecture'] = {}
        reachable_activities['Payment'] = {'Payment', 'Send for Credit Collection'}
        reachable_activities['Insert Fine Notification'] = {'Appeal to Judge', 'Receive Result Appeal from Prefecture',
                                                            'Notify Result Appeal to Offender', 'Add penalty', 'Send for Credit Collection'}
        reachable_activities['Send Fine'] = {'Send for Credit Collection'}
        reachable_activities['Insert Date Appeal to Prefecture'] = {'Send for Credit Collection'}
        reachable_activities['Appeal to Judge'] = {'Send for Credit Collection'}
        reachable_activities['Receive Result Appeal from Prefecture'] = {'Send for Credit Collection'}
        reachable_activities['Notify Result Appeal to Offender'] = {'Send for Credit Collection'}
        reachable_activities['Add penalty'] = {'Send for Credit Collection'}
        reachable_activities['Send for Credit Collection'] = {}

    #    # detect loops and create loop objects
#    loop_vertex = detect_loops(sim_net)
#    loop_vertex = delete_composite_loops(loop_vertex)
#
#    loops = list()
#    for loop_length in loop_vertex.keys():
#        for i, loop in enumerate(loop_vertex[loop_length]):
#            events_loop = [events_trans_map[vertex] for vertex in loop if vertex in events_trans_map.keys()]
#            loop_obj = Loop(loop, events_loop, sim_net, "{}_{}".format(loop_length, i))
#            loop_obj.set_complete_net_input(last_collapsed_left)
#            loop_obj.set_complete_net_output(last_collapsed_right)
#            loop_obj.set_complete_net_loop_nodes(sim_map)
#            loops.append(loop_obj)
#
#    # join different parts of the same loop (different loops with same inputs and outputs)
#    not_joinable = False
#    while not not_joinable:
#        old_loops = copy.copy(loops)
#        new_loop_nodes = None
#        new_loop_events = None
#        for loop in old_loops:
#            new_loop_nodes = set()
#            new_loop_events = set()
#            for loop_inner in old_loops:
#                if loop != loop_inner:
#                    if loop.input_places_complete == loop_inner.input_places_complete and loop.output_places_complete == loop_inner.output_places_complete:
#                        new_loop_nodes = new_loop_nodes.union(set(loop.vertex))
#                        new_loop_nodes = new_loop_nodes.union(set(loop_inner.vertex))
#                        new_loop_events = new_loop_events.union(set(loop.events))
#                        new_loop_events = new_loop_events.union(set(loop_inner.events))
#                        loops.remove(loop_inner)
#
#            if len(new_loop_nodes) > 0:
#                new_loop = Loop(list(new_loop_nodes), list(new_loop_events), sim_net, loop.name)
#                new_loop.set_complete_net_input(last_collapsed_left)
#                new_loop.set_complete_net_output(last_collapsed_right)
#                new_loop.set_complete_net_loop_nodes(sim_map)
#                new_loop.set_complete_net_loop_events(events_trans_map)
#                loops.append(new_loop)
#                loops.remove(loop)
#                break
#        if len(new_loop_nodes) == 0:
#            not_joinable = True
#
#    for loop in loops:
#        loop.set_nearest_input_complete_net(sim_net, sim_map)

    tic = time()
    # Scanning the log to get the data related to decision points
    print('Extracting training data from Event Log...')
    decision_points_data, event_attr, stored_dicts = dict(), dict(), dict()
    variants = variants_filter.get_variants(log)
    # Decision points of interest are searched considering the variants only
    for variant in tqdm(variants):
        transitions_sequence, events_sequence = list(), list()
        dp_events_sequence = dict()
        for i, event_name in enumerate(variant.split(',')):
            trans_from_event = trans_events_map[event_name]
            transitions_sequence.append(trans_from_event)
            events_sequence.append(event_name)
            if len(transitions_sequence) > 1:
                #dp_dict, stored_dicts = get_decision_points_and_targets(transitions_sequence, loops, net, parallel_branches, stored_dicts)
                dp_dict, stored_dicts = get_decision_points_and_targets(transitions_sequence, None, net, reachable_activities, stored_dicts)
                dp_events_sequence['Event_{}'.format(i+1)] = dp_dict

        # Final update of the current trace (from last event to sink)
        transition = [trans for trans in net.transitions if trans.label == event_name][0]
        dp_events_sequence['End'] = get_all_dp_from_event_to_sink(transition, sink_complete_net, dp_events_sequence)

        for trace in variants[variant]:
            # Storing the trace attributes (if any)
            if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
                event_attr.update(trace.attributes)

            # Keeping the same attributes observed in previously (to keep dictionaries at the same length)
            event_attr = {k: np.nan for k in event_attr.keys()}

            transitions_sequence = list()
            for i, event in enumerate(trace):
                trans_from_event = trans_events_map[event["concept:name"]]
                transitions_sequence.append(trans_from_event)

                # Appending the last attribute values to the decision point dictionary
                if len(transitions_sequence) > 1:
                    dp_dict = dp_events_sequence['Event_{}'.format(i+1)]
                    for dp in dp_dict.keys():
                        # Adding the decision point to the total dictionary if it is not already there
                        if dp not in decision_points_data.keys():
                            decision_points_data[dp] = {k: [] for k in ['target']}
                        for dp_target in dp_dict[dp]:
                            for a in event_attr.keys():
                                # Attribute not present and not nan: add it as new key and fill previous entries as nan
                                if a not in decision_points_data[dp] and event_attr[a] is not np.nan:
                                    n_entries = len(decision_points_data[dp]['target'])
                                    decision_points_data[dp][a] = [np.nan] * n_entries
                                    decision_points_data[dp][a].append(event_attr[a])
                                # Attribute present: just append it to the existing list
                                elif a in decision_points_data[dp]:
                                    decision_points_data[dp][a].append(event_attr[a])
                            # Appending also the target transition label to the decision point dictionary
                            decision_points_data[dp]['target'].append(dp_target)

                # Updating the attribute values dictionary with the values from the current event
                event_attr.update(get_attributes_from_event(event))

            # Appending the last attribute values to the decision point dictionary (from last event to sink)
            if len(dp_events_sequence['End']) > 0:
                for dp in dp_events_sequence['End'].keys():
                    if dp not in decision_points_data.keys():
                        decision_points_data[dp] = {k: [] for k in ['target']}
                    for dp_target in dp_events_sequence['End'][dp]:
                        for a in event_attr.keys():
                            if a not in decision_points_data[dp] and event_attr[a] is not np.nan:
                                n_entries = len(decision_points_data[dp]['target'])
                                decision_points_data[dp][a] = [np.nan] * n_entries
                                decision_points_data[dp][a].append(event_attr[a])
                            elif a in decision_points_data[dp]:
                                decision_points_data[dp][a].append(event_attr[a])
                        decision_points_data[dp]['target'].append(dp_target)

    # Data has been gathered. For each decision point, fitting a decision tree on its data and extracting the rules
    file_name = 'results.txt'
    for decision_point in decision_points_data.keys():
        print("\nDecision point: {}".format(decision_point))
        dataset = pd.DataFrame.from_dict(decision_points_data[decision_point])
        # Replacing ':' with '_' both in the dataset columns and in the attributes map since ':' creates problems
        dataset.columns = dataset.columns.str.replace(':', '_')
        attributes_map = {k.replace(':', '_'): attributes_map[k] for k in attributes_map}

        # Sampling to get a balanced dataset (in terms of target value)
        groups = list()
        grouped_df = dataset.groupby('target')
        for target_value in dataset['target'].unique():
            groups.append(grouped_df.get_group(target_value))
        groups.sort(key=len)
        # Groups is a list containing a dataset for each target value, ordered by length
        # If the smaller datasets are less than the 35% of the total dataset length, then apply the sampling
        if sum(len(group) for group in groups[:-1]) / len(dataset) <= 0.35:
            samples = list()
            # Each smaller dataset is appended to the 'samples' list, along with a sampled dataset from the largest one
            for group in groups[:-1]:
                samples.append(group)
                samples.append(groups[-1].sample(len(group)))
            # The datasets in the 'samples' list are then concatenated together
            dataset = pd.concat(samples)

        # Discovering branching conditions with Daikon - comment these four lines to go back to decision tree + pruning
        # rules = discover_branching_conditions(dataset)
        # rules = {k: rules[k].replace('_', ':') for k in rules}
        # print(rules)
        # continue

        print("Fitting a decision tree on the decision point's dataset...")
        dt = DecisionTree(attributes_map)
        dt.fit(dataset)

        if len(dt.get_nodes()) > 1:
            print("Training complete. Extracting rules...")
            with open(file_name, 'a') as f:
                f.write('{} - SUCCESS\n'.format(decision_point))
                lf = len(dataset[dataset['target'].str.startswith(('skip', 'tauJoin', 'tauSplit', 'init_loop'))])
                f.write('Rows with invisible activity as target: {}/{}\n'.format(str(lf), str(len(dataset))))

                # Predict (just to see the accuracy)
                y_pred = dt.predict(dataset.drop(columns=['target']))
                print("Train accuracy: {}".format(metrics.accuracy_score(dataset['target'], y_pred)))

                # Rule extraction without pruning
                # rules = dt.extract_rules()

                # Rule extraction with pruning
                rules = dt.extract_rules_with_pruning(dataset)

                # Overlapping rules discovery
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
                f.write('Rows with invisible activity as target: {}/{}\n\n'.format(str(lf), str(len(dataset))))

    toc = time()
    print("\nTotal time: {}".format(toc-tic))


if __name__ == '__main__':
    main()
