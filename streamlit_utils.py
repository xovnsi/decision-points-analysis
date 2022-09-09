import os
import json
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from tqdm import tqdm
from pm4py.algo.filtering.log.variants import variants_filter
from backward_search import get_decision_points_and_targets, get_all_dp_from_sink_to_last_event
from rules_extraction import sampling_dataset, extract_rules_with_pruning, pessimistic_pruning,\
    discover_overlapping_rules, shorten_rules_manually
from utils import get_attributes_from_event, get_map_events_to_transitions
from daikon_utils import discover_branching_conditions
from DecisionTreeC45.DecisionTree import DecisionTree
from sklearn import metrics


def get_unique_values_log():
    unique_values_trace, unique_values_event = dict(), dict()
    for log_trace in st.session_state.log:
        for trace_attr in log_trace.attributes:
            if trace_attr not in unique_values_trace.keys():
                unique_values_trace[trace_attr] = set()
            unique_values_trace[trace_attr].add(log_trace.attributes[trace_attr])
        for trace_event in log_trace:
            for event_attr in trace_event:
                if event_attr not in unique_values_event.keys():
                    unique_values_event[event_attr] = set()
                unique_values_event[event_attr].add(trace_event[event_attr])

    df_trace = pd.DataFrame.from_dict(unique_values_trace, orient='index').T
    df_event = pd.DataFrame.from_dict(unique_values_event, orient='index').T

    return df_trace, df_event, unique_values_trace, unique_values_event


def create_dict():
    dict_conf = dict()
    for name in st.session_state:
        if name in st.session_state['list_event_attr'] or name in st.session_state['list_trace_attr']:
            # remove initial 'e_' or 't_' from the name
            dict_conf["_".join(name.split('_')[1:])] = st.session_state[name]
    return dict_conf


def save_json(dict_conf, data_dir='streamlitTemp'):
    json_string = json.dumps(dict_conf)
    file_name = '{}.attr'.format(st.session_state['uploaded_log_name'])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(os.path.join(data_dir, file_name), 'w') as file:
        json.dump(json_string, file)


def build_datasets():
    sink_complete_net = [place for place in st.session_state['net'].places if place.name == 'sink'][0]
    events_to_trans_map = get_map_events_to_transitions(st.session_state['net'])

    # Scanning the log to get the logs related to decision points
    print('Extracting training logs from Event Log...')
    decision_points_data, event_attr, stored_dicts = dict(), dict(), dict()
    variants = variants_filter.get_variants(st.session_state['log'])
    # Decision points of interest are searched considering the variants only
    for variant in tqdm(variants):
        transitions_sequence, events_sequence = list(), list()
        dp_events_sequence = dict()
        for i, event_name in enumerate(variant.split(',')):
            trans_from_event = events_to_trans_map[event_name]
            transitions_sequence.append(trans_from_event)
            events_sequence.append(event_name)
            if len(transitions_sequence) > 1:
                dp_dict, stored_dicts = get_decision_points_and_targets(transitions_sequence, st.session_state['net'],
                                                                        stored_dicts)
                dp_events_sequence['Event_{}'.format(i + 1)] = dp_dict

        # Final update of the current trace (from last event to sink)
        transition = [trans for trans in st.session_state['net'].transitions if trans.label == event_name][0]
        dp_events_sequence['End'] = get_all_dp_from_sink_to_last_event(transition, sink_complete_net, dp_events_sequence)

        for trace in variants[variant]:
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
                    dp_dict = dp_events_sequence['Event_{}'.format(i + 1)]
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

    return decision_points_data


def rules_computation():
    file_name = 'results_{}_{}.txt'.format(st.session_state['uploaded_log_name'], datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    for decision_point in st.session_state.decision_points_data.keys():
        st.write("\nDecision point: {}".format(decision_point))
        dataset = pd.DataFrame.from_dict(st.session_state.decision_points_data[decision_point])
        # Replacing ':' with '_' both in the dataset columns and in the attributes map since ':' creates problems
        dataset.columns = dataset.columns.str.replace(':', '_')
        attributes_map = {k.replace(':', '_'): st.session_state['attributes_map'][k] for k in st.session_state['attributes_map']}

        # Discovering branching conditions with Daikon
        if st.session_state.method == 'Daikon':
            st.write("Discovering branching conditions with Daikon...")
            rules = discover_branching_conditions(dataset)
            rules = {k: rules[k].replace('_', ':') for k in rules}
            st.write(rules)
        else:
            st.write("Fitting a decision tree on the decision point's dataset...")
            accuracies, f_scores = list(), list()
            for i in tqdm(range(10)):
                # Sampling
                dataset = sampling_dataset(dataset)

                # Fitting
                dt = DecisionTree(attributes_map)
                dt.fit(dataset)

                # Predict
                y_pred = dt.predict(dataset.drop(columns=['target']))

                # Accuracy
                accuracy = metrics.accuracy_score(dataset['target'], y_pred)
                accuracies.append(accuracy)

                # F1-score
                if len(dataset['target'].unique()) > 2:
                    f1_score = metrics.f1_score(dataset['target'], y_pred, average='weighted')
                else:
                    f1_score = metrics.f1_score(dataset['target'], y_pred, pos_label=dataset['target'].unique()[0])
                f_scores.append(f1_score)

            # Rules extraction
            if len(dt.get_nodes()) > 1:
                st.write("Training complete. Extracting rules...")
                with open(file_name, 'a') as f:
                    f.write('{} - SUCCESS\n'.format(decision_point))
                    f.write('Dataset target values counts:\n {}\n'.format(dataset['target'].value_counts()))

                    st.write("Accuracy: {}".format(sum(accuracies) / len(accuracies)))
                    f.write('Accuracy: {}\n'.format(sum(accuracies) / len(accuracies)))
                    st.write("F1 score: {}".format(sum(f_scores) / len(f_scores)))
                    f.write('F1 score: {}\n'.format(sum(f_scores) / len(f_scores)))

                    if st.session_state.pruning == 'No Pruning':
                        # Rule extraction without pruning
                        rules = dt.extract_rules()
                    elif st.session_state.pruning == 'Pessimistic':
                        # Alternative pruning (directly on tree)
                        pessimistic_pruning(dt, dataset)
                        rules = dt.extract_rules()
                    elif st.session_state.pruning == 'Rules simplification':
                        # Rule extraction with pruning
                        rules = extract_rules_with_pruning(dt, dataset)

                    if st.session_state.overlapping == 'Yes':
                        # Overlapping rules discovery
                        rules = discover_overlapping_rules(dt, dataset, attributes_map, rules)

                    rules = shorten_rules_manually(rules, attributes_map)
                    rules = {k: rules[k].replace('_', ':') for k in rules}

                    f.write('Rules:\n')
                    for k in rules:
                        f.write('{}: {}\n'.format(k, rules[k]))
                    f.write('\n')
                st.write(rules)
            else:
                st.write("Could not fit a Decision Tree on this dataset.")
                with open(file_name, 'a') as f:
                    f.write('{} - FAIL\n'.format(decision_point))
                    f.write('Dataset target values counts: {}\n'.format(dataset['target'].value_counts()))
    return file_name