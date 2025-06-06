import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argcomplete
import argparse
import json
import copy
from sklearn import metrics

import pm4py
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.bpmn.importer import importer as bpmn_importer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.bpmn import converter as bpmn_to_petri_converter
from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter

from backward_search import get_decision_points_and_targets, get_all_dp_from_sink_to_last_event
from rules_extraction import sampling_dataset, shorten_rules_manually, discover_overlapping_rules
from utils import get_attributes_from_event, get_map_events_to_transitions
from DecisionTreeC45 import DecisionTree

"""
A python script for BPMN model decision points analysis, based on the streamlit webapp.
BPMN model is loaded from file or mined from provided logs. Then, based on the logs and the attribute types,
it discovers decision point rules and updates the BPMN model file if exists, or creates a new one.

Script still relies on the decision-points-analysis project.

Location:
decision-points-analysis/decision-points-analysis-no-streamlit.py

Instructions:
 > python decision_points_analysis_no_streamlit.py -i <name> -n <number-of-trees>

where:
<name> is the name of the dataset:
 - ./models/<name>.bpmn – BPMN 2.0 XML model (optional)
 - ./logs/log-<name>.xes – logs in XES format
 - ./dt-attributes/<name>.attr – attribute categories in JSON format
<number-of-trees> is the number of trees to be trained for each decision point.
"""

print("START")

MODEL_FOLDER_PATH = "models/"
LOGS_FOLDER_PATH = "logs/"
ATTR_FOLDER_PATH = "dt-attributes/"


def parse_args():
    # Argument (verbose and net_name)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_name", help="Name of the bpmn model file (without extension)", type=str) \
        .completer = lambda **kwargs: [name.split('.')[0]
                                       for name in os.listdir(MODEL_FOLDER_PATH.rstrip('/')) if name.endswith('.bpmn')]
    parser.add_argument("-n", "--num_trees", help="Number of decision trees (default: 10)", type=int, default=10)
    argcomplete.autocomplete(parser)
    return parser.parse_args()


def main():
    # Parse args
    args = parse_args()
    input_name = args.input_name
    num_trees = args.num_trees


    # Import input files
    log = xes_importer.apply(f'{LOGS_FOLDER_PATH}/log-{input_name}.xes')

    attributes_map_file = f'{input_name}.attr'
    if attributes_map_file in os.listdir(ATTR_FOLDER_PATH):
        with open(os.path.join(ATTR_FOLDER_PATH, attributes_map_file), 'r') as f:
            attributes_map = json.load(f)
    else:
        raise FileNotFoundError('Create a configuration file for the decision tree before fitting it.')


    # Convert attribute types according to the attributes_map file
    for trace in log:
        for event in trace:
            for attribute in event.keys():
                if attribute in attributes_map:
                    if attributes_map[attribute] == 'continuous':
                        event[attribute] = float(event[attribute])
                    elif attributes_map[attribute] == 'boolean':
                        event[attribute] = bool(event[attribute])


    # Import the BPMN model and convert to Petri net
    # TODO
    bpmn_net = None
    try:
        bpmn_net = bpmn_importer.apply(f"{MODEL_FOLDER_PATH}/{input_name}.bpmn")
        net, im, fm = bpmn_to_petri_converter.apply(bpmn_net)
        print("Petri Net model converted from BPMN file")
    except FileNotFoundError:
        print("Existing Petri Net model not found. Extracting one using the Inductive Miner...")
        net, im, fm = pm4py.discover_petri_net_inductive(log)
    sink_complete_net = [place for place in net.places if place.name == 'sink'][0]


    # Map observed events to corresponding transitions in the Petri net
    # (Note: additional cleanup may be needed to handle loops or complex structures)
    events_to_trans_map = get_map_events_to_transitions(net)


    # Scanning the log to get the logs related to decision points
    print('Extracting training logs from Event Log...')
    decision_points_data, event_attr, stored_dicts = dict(), dict(), dict()
    variants = variants_filter.get_variants(log)


    # Decision points of interest are searched considering the variants only
    for variant_key, traces in tqdm(variants.items()):
        transitions_sequence, events_sequence = list(), list()
        dp_events_sequence = dict()
        for i, event_name in enumerate(variant_key):
            trans_from_event = events_to_trans_map[event_name]
            transitions_sequence.append(trans_from_event)
            events_sequence.append(event_name)
            if len(transitions_sequence) > 1:
                dp_dict, stored_dicts = get_decision_points_and_targets(transitions_sequence, net, stored_dicts)
                dp_events_sequence['Event_{}'.format(i + 1)] = dp_dict

        # Final update of the current trace (from last event to sink)
        transition = [trans for trans in net.transitions if trans.label == event_name][0]
        dp_events_sequence['End'] = get_all_dp_from_sink_to_last_event(transition, sink_complete_net,
                                                                       dp_events_sequence)

        for trace in traces:
            # Storing the trace attributes (if any)
            if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
                event_attr.update(trace.attributes)

            # Keeping the same attributes observed in previously (to keep dictionaries at the same length)
            event_attr = {k: np.nan for k in event_attr.keys()}

            transitions_sequence = list()
            for i, event in enumerate(trace):
                trans_from_event = events_to_trans_map[event["concept:name"]]
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


    # For each decision point, fit decision trees and extract rules
    all_rules = {}
    for decision_point in decision_points_data.keys():
        print("\nDecision point: {}".format(decision_point))
        dataset = pd.DataFrame.from_dict(decision_points_data[decision_point])
        # Replacing ':' with '_' both in the dataset columns and in the attributes map since ':' creates problems
        dataset.columns = dataset.columns.str.replace(':', '_')
        attributes_map = {k.replace(':', '_'): attributes_map[k] for k in attributes_map}

        accuracies, f_scores = list(), list()
        best_dt = None
        best_score = 0

        # Train trees to find decision point's rules
        for _ in tqdm(range(num_trees)):
            dataset = sampling_dataset(dataset)
            dt = DecisionTree.DecisionTree(attributes_map)
            dt.fit(dataset)

            y_pred = dt.predict(dataset.drop(columns=['target']))
            if y_pred is None:
                raise ValueError(
                    "Prediction failed: 'y_pred' is None. Check if the decision tree is trained and input is valid.")

            accuracy = metrics.accuracy_score(dataset['target'], y_pred)
            accuracies.append(accuracy)

            if len(dataset['target'].unique()) > 2:
                f1_score = metrics.f1_score(dataset['target'], y_pred, average='weighted')
            else:
                f1_score = metrics.f1_score(dataset['target'], y_pred, pos_label=dataset['target'].unique()[0])

            f_scores.append(f1_score)
            if f1_score > best_score:
                best_score = f1_score
                best_dt = copy.deepcopy(dt)

        # If no rules are found, return
        if len(best_dt.get_nodes()) <= 1:
            print('{} - FAIL\n'.format(decision_point))
            return

        print(f"Avg acc: {sum(accuracies) / len(accuracies)}")
        print(f"Avg F1: {sum(f_scores) / len(f_scores)}")
        print(f"Best model F1: {best_score}")
        print("Extracting rules from best model...")

        # Rule extraction without pruning
        rules = best_dt.extract_rules()
        rules = discover_overlapping_rules(best_dt, dataset, attributes_map, rules)

        # Shorten rules and append to dict
        rules = shorten_rules_manually(rules, attributes_map)
        rules = {k: rules[k].replace('_', ':') for k in rules}
        all_rules = {**all_rules, **rules}
        print(rules)

    # Combine rules with corresponding nodes
    rule_labels = {k: all_rules[v] for k, v in events_to_trans_map.items() if v in all_rules}
    print(rule_labels)

    # Save BPMN model on the net
    output_path = f"{MODEL_FOLDER_PATH}/{input_name}.bpmn"

    if bpmn_net is None:
        # Model was not loaded so the file may not exist - save model
        bpmn_net = pm4py.convert_to_bpmn(net, im, fm)
        bpmn_exporter.apply(bpmn_net, output_path)
        print(f"BPMN model saved to {output_path}")

    # Edit saved model by adding discovered rules to flows
    import xml.etree.ElementTree as ET
    tree = ET.parse(output_path)
    root = tree.getroot()
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

    for node in root.findall('.//bpmn:*', ns):
        name = node.attrib.get('name')
        if name in rule_labels:
            flow_id = node.find('bpmn:incoming', ns).text
            flow = root.find(f".//bpmn:sequenceFlow[@id='{flow_id}']", ns)
            if flow is not None:
                flow.set('name', rule_labels[name])

    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"BPMN model edited with rule labels")



main()
