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
#breakpoint()
# get the map of places and events
places_events_map = get_map_place_to_events(net, loop_vertex)
# get the map of transitions and events
trans_events_map = get_map_transitions_events(net)
# get a dict of data for every decision point
decision_points_data = dict()
for place in net.places:
    if len(place.out_arcs) >= 2:
        decision_points_data[place.name] = pd.DataFrame()

# fill the data
tic = time()
attributes_map = {'amount': 'continuous', 'policyType': 'categorical', 'appeal': 'boolean',
        'status': 'categorical', 'communication': 'categorical', 'discarded': 'boolean'}
for trace in log:
    #breakpoint()
    if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
        trace_attr_row = trace.attributes
        #trace_attr_row.pop('concept:name')
    last_k_events = list()
    for event in trace:
        trans_from_event = trans_events_map[event["concept:name"]]
        if len(last_k_events) == 0:
            last_event_name = 'None'
        else:
            last_event_name = last_k_events[-1]['concept:name']
        last_event_trans = trans_events_map[last_event_name]
        places_from_event = get_place_from_transition(places_events_map, event['concept:name'], loop_vertex, last_event_trans, 
                in_transitions_loop, out_places_loops, in_places_loops, trans_from_event)  
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
        last_k_events.append(event)
        if len(last_k_events) > k:
            last_k_events = last_k_events[-k:]

from sklearn import linear_model, preprocessing
for decision_point in decision_points_data.keys():
    print("\n", decision_point)

    dataset = decision_points_data[decision_point]
    X = pd.get_dummies(dataset[dataset.columns.difference(['target'])])
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y = dataset[['target']].values.ravel()

    logistic_mn = linear_model.LogisticRegression(C=0.1, penalty='l1', fit_intercept=False, multi_class='multinomial', solver='saga', max_iter=200)
    logistic_mn.fit(X_scaled, y)

    print("Train accuracy: {}".format(logistic_mn.score(X_scaled, y)))
    print("Input features: {}. Output features: {}".format(X.columns.values, logistic_mn.classes_))
    # print(logistic_mn.coef_, logistic_mn.intercept_)
    # prob = [(1 / (1 + np.exp(c))) for c in logistic_mn.coef_]
    # print(prob)
    # print(logistic_mn.predict(X))

    # decision boundary: w1x1 + w2x2 + w0 >= 0 -> class 1
    #                    w1x2 + w2x2 + w0 < 0  -> class 0
    # to extend for more than two input variables and target classes
    rounded_coefs = [[round(x, 3) for x in lst] for lst in logistic_mn.coef_]
    equations_df = pd.DataFrame(rounded_coefs, columns=X.columns)
    equations_df['intercept'] = [round(x, 3) for x in logistic_mn.intercept_]
    equations_df['class'] = logistic_mn.classes_[1:]
    print("Weights:\n", equations_df)

    decisions = dict()
    for row_ind in range(len(equations_df)):
        rule = ""
        for attribute in equations_df.drop(columns=['class', 'intercept']):
            if equations_df.loc[row_ind, attribute] < 0:
                rule = rule + "- " + str(abs(equations_df.loc[row_ind, attribute])) + "*" + attribute + " "
            else:
                rule = rule + "+ " + str(equations_df.loc[row_ind, attribute]) + "*" + attribute + " "
        if equations_df.loc[row_ind, 'intercept'] < 0:
            rule = rule + "- " + str(abs(equations_df.loc[row_ind, 'intercept']))
        else:
            rule = rule + "+ " + str(equations_df.loc[row_ind, 'intercept'])
        rule = rule + " >= 0"
        decisions[equations_df.loc[row_ind, 'class']] = rule
    print("Rule using weights: ", decisions)

    final_df = pd.DataFrame(columns=dataset.columns)
    for row_ind in range(len(equations_df)):
        new_row = dict.fromkeys(dataset.columns)
        for attribute in dataset.columns.difference(['target']):
            if attributes_map[attribute] == 'categorical':
                attr_cols = [c for c in X.columns if c.startswith(attribute)]
                attr_df = pd.DataFrame(data=pd.to_numeric(equations_df.loc[row_ind, attr_cols])).transpose()
                chosen_attr = attr_df.idxmax(axis=1).values[0]
                new_row[attribute] = chosen_attr.split("_")[1]
            else:
                new_row[attribute] = equations_df.loc[row_ind, attribute]
        new_row['target'] = equations_df.loc[row_ind, 'class']
        final_df = pd.concat([final_df, pd.DataFrame([new_row])])
    print("From weights to categorical:\n", final_df)


print("\n=============== DECISION TREES ===============")

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
