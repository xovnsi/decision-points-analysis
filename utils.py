import copy
import pandas as pd

from DecisionTree import DecisionTree
from decision_tree_utils import extract_rules_from_leaf


def get_decision_points_and_targets(sequence, net, stored_dicts) -> [dict, dict]:
    """ Returns a dictionary containing decision points and their targets.

    Starting from the last activity in the sequence, the algorithm selects the previous not
    parallel activity. Exploring the net backward, it returns all the decision points with their targets
    encountered on the path leading to the last activity in the sequence.
    """

    # Current activity
    current_act = [trans for trans in net.transitions if trans.name == sequence[-1]][0]

    # Backward decision points search towards the previous reachable activity
    dp_dict = dict()
    for previous_act in reversed(sequence[:-1]):
        prev_curr_key = ', '.join([previous_act, current_act.name])
        if prev_curr_key not in stored_dicts.keys():
            dp_dict, event_found = _new_get_dp_to_previous_event(previous_act, current_act)
            if event_found:
                # ------------------------------ DEBUGGING ------------------------------
                events_trans_map = get_map_events_transitions(net)
                print("\nPrevious event: {}".format(events_trans_map[previous_act]))
                print("Current event: {}".format(current_act.label))
                print("DPs")
                for key in dp_dict.keys():
                    print(" - {}".format(key))
                    for inn_key in dp_dict[key]:
                        if inn_key in events_trans_map.keys():
                            print("   - {}".format(events_trans_map[inn_key]))
                        else:
                            print("   - {}".format(inn_key))
                # ------------------------------ DEBUGGING ------------------------------
                stored_dicts[prev_curr_key] = dp_dict
                break
        else:
            dp_dict = stored_dicts[prev_curr_key]
            break

    return dp_dict, stored_dicts


def _new_get_dp_to_previous_event(previous, current, decision_points=None, passed_inn_arcs=None) -> [dict, bool]:
    """ Extracts all the decision points that are traversed between two activities (previous and current), reporting the
    decision(s) that has been taken for each of them.

    Starting from the 'current' activity, the algorithm proceeds backwards on each incoming arc (in_arc). Then, it saves
    the so-called 'inner_arcs', which are the arcs between the previous place (in_arc.source) and its previous
    activities (in_arc.source.in_arcs).
    If the 'previous' activity is immediately before the previous place (so it is the source of one of the inner_arcs),
    then the algorithm sets a boolean variable 'target_found' to True to signal that the target has been found, and it
    adds the corresponding decision point (in_arc.source) to the dictionary.
    In any case, all the other backward paths containing an invisible activity are explored recursively.
    Every time an 'inner_arc' is traversed for exploration, it is added to the 'passed_inn_arcs' list, to avoid looping
    endlessly. Indeed, before a new recursion on an 'inner_arc', the algorithm checks if it is present in that list: if
    it is not present, it simply goes on with the recursion, since it means that the specific path has not been explored
    yet.
    Note that decision points are then added to the dictionary in a forward way: whenever the recursion cannot go on (no
    more invisible activities backward) it returns also signalling the 'target_found' value. This is True if the current
    path found the target activity (or an already explored 'inner_arc' during the same path, as explained before). The
    returned value is then put in disjunction with the current value of 'target_found' in case the target activity has
    been found by the actual instance and not by the recursive one.
    It returns a 'decision_points' dictionary which contains, for each decision point on every possible path between
    the 'current' activity and the 'previous' activity, the target value(s) to be followed.
    """

    if decision_points is None:
        decision_points = dict()
    if passed_inn_arcs is None:
        passed_inn_arcs = set()
    target_found = False
    for in_arc in current.in_arcs:
        # Preparing the lists containing inner_arcs towards invisible and non-invisible transitions
        inner_inv_acts, inner_in_arcs_names = set(), set()
        for inner_in_arc in in_arc.source.in_arcs:
            if inner_in_arc.source.label is None:
                inner_inv_acts.add(inner_in_arc)
            else:
                inner_in_arcs_names.add(inner_in_arc.source.name)

        # Base case: the target activity (previous) is one of the activities immediately before the current one
        if previous in inner_in_arcs_names:
            target_found = True
            decision_points = _add_dp_target(decision_points, in_arc.source, current.name, target_found)
        # Recursive case: follow every invisible activity backward
        for inner_in_arc in inner_inv_acts:
            if inner_in_arc not in passed_inn_arcs:
                passed_inn_arcs.add(inner_in_arc)
                decision_points, previous_found = _new_get_dp_to_previous_event(previous, inner_in_arc.source,
                                                                                decision_points, passed_inn_arcs)
                passed_inn_arcs.remove(inner_in_arc)
                decision_points = _add_dp_target(decision_points, in_arc.source, current.name, previous_found)
                target_found = target_found or previous_found

    return decision_points, target_found


def _add_dp_target(decision_points, dp, target, add_dp) -> dict:
    """ Adds the decision point and its target activity to the 'decision_points' dictionary.

    Given the 'decision_points' dictionary, the place 'dp' and the target activity name, adds the target activity name
    to the set of targets related to the decision point. If not present, adds the decision point to the dictionary keys
    first. This is done if the place is an actual decision point, and if the boolean variable 'add_dp' is True.
    """

    if add_dp and len(dp.out_arcs) > 1:
        if dp.name in decision_points.keys():
            decision_points[dp.name].add(target)
        else:
            decision_points[dp.name] = {target}
    return decision_points


def get_attributes_from_event(event) -> dict:
    """ Given an event from an event log, returns a dictionary containing the attributes values of the given event.

    Each attribute is set as key of the dictionary, and its value is the actual value of the attribute in the event.
    Attributes 'time:timestamp' and 'concept:name' are ignored and therefore not added to the dictionary.
    """

    attributes = dict()
    for attribute in event.keys():
        if attribute not in ['time:timestamp', 'concept:name']:
            attributes[attribute] = event[attribute]

    return attributes


def get_map_transitions_events(net) -> dict:
    """ Compute a map of transitions name and events

    Given a Petri Net in the implementation of Pm4Py library, the function creates
    a dictionary containing for every event the corresponding transition name
    """
    # initialize
    map_trans_events = dict()
    for trans in net.transitions:
        map_trans_events[trans.label] = trans.name
    map_trans_events['None'] = 'None'
    return map_trans_events


def get_map_events_transitions(net) -> dict:
    """ Compute a map of event name and transitions name

    Given a Petri Net in the implementation of Pm4Py library, the function creates
    a dictionary containing for every transition the corresponding event name
    """
    # initialize
    map_events_trans = dict()
    for trans in net.transitions:
        if trans.label is not None:
            map_events_trans[trans.name] = trans.label
    return map_events_trans


def get_all_dp_from_event_to_sink(transition, sink, decision_points_seen) -> dict:
    """ Returns all the decision points in the path from the 'transition' activity to the sink of the Petri net, passing
    only through invisible transitions.

    Starting from the sink, extracts all the transitions connected to the sink (the ones immediately before the sink).
    If 'transition' is one of them, there are no decision points to return, so it returns an empty dictionary.
    Otherwise, for each invisible transition among them, it calls method '_new_get_dp_to_previous_event' to retrieve
    all the decision points and related targets between 'transition' and the invisible transition currently considered.
    Discovered decision points for all the backward paths are put in the same 'decision_points' dictionary.
    """

    dp_seen = set()
    for event_key in decision_points_seen:
        for dp_key in decision_points_seen[event_key]:
            dp_seen.add(dp_key)

    sink_in_acts = [in_arc.source for in_arc in sink.in_arcs]
    if transition in sink_in_acts:
        return dict()
    else:
        decision_points = dict()

        for sink_in_act in sink_in_acts:
            if sink_in_act.label is None:
                decision_points = _get_dp_to_previous_event_from_sink(transition, sink_in_act, dp_seen, decision_points)

        return decision_points


def _get_dp_to_previous_event_from_sink(previous, current, dp_seen, decision_points=None, passed_inn_arcs=None) -> dict:

    if decision_points is None:
        decision_points = dict()
    if passed_inn_arcs is None:
        passed_inn_arcs = set()
    for in_arc in current.in_arcs:
        # If decision point already seen in variant, stop following this path
        if in_arc.source.name in dp_seen:
            continue

        for inner_in_arc in in_arc.source.in_arcs:
            # If invisible activity backward, recurse only if 'inner_in_arc' has not been seen in this path yet
            if inner_in_arc.source.label is None:
                if inner_in_arc not in passed_inn_arcs:
                    passed_inn_arcs.add(inner_in_arc)
                    decision_points = _get_dp_to_previous_event_from_sink(previous, inner_in_arc.source, dp_seen,
                                                                          decision_points, passed_inn_arcs)
                    decision_points = _add_dp_target(decision_points, in_arc.source, current.name, True)
                    passed_inn_arcs.remove(inner_in_arc)
                else:
                    decision_points = _add_dp_target(decision_points, in_arc.source, current.name, True)
            # Otherwise, just add the decision point and its target activity
            else:
                decision_points = _add_dp_target(decision_points, in_arc.source, current.name, True)

    return decision_points


def discover_overlapping_rules(base_tree, dataset, attributes_map, original_rules):
    """ Discovers overlapping rules, if any.

    Given the fitted decision tree, extracts the training set instances that have been wrongly classified, i.e., for
    each leaf node, all those instances whose target is different from the leaf label. Then, it fits a new decision tree
    on those instances, builds a rules dictionary as before (disjunctions of conjunctions) and puts the resulting rules
    in disjunction with the original rules, according to the target value.
    Method taken by "Decision Mining Revisited - Discovering Overlapping Rules" by Felix Mannhardt, Massimiliano de
    Leoni, Hajo A. Reijers, Wil M.P. van der Aalst (2016).
    """

    rules = copy.deepcopy(original_rules)

    leaf_nodes = base_tree.get_leaves_nodes()
    leaf_nodes_with_wrong_instances = [ln for ln in leaf_nodes if len(ln.get_class_names()) > 1]

    for leaf_node in leaf_nodes_with_wrong_instances:
        vertical_rules = extract_rules_from_leaf(leaf_node)

        vertical_rules_query = ""
        for r in vertical_rules:
            r_attr, r_comp, r_value = r.split(' ')
            vertical_rules_query += r_attr
            if r_comp == '=':
                vertical_rules_query += ' == '
            else:
                vertical_rules_query += ' ' + r_comp + ' '
            if dataset.dtypes[r_attr] == 'float64' or dataset.dtypes[r_attr] == 'bool':
                vertical_rules_query += r_value
            else:
                vertical_rules_query += '"' + r_value + '"'
            if r != vertical_rules[-1]:
                vertical_rules_query += ' & '

        leaf_instances = dataset.query(vertical_rules_query)
        # TODO not considering missing values for now, so wrong_instances could be empty
        # This happens because all the wrongly classified instances have missing values for the query attribute(s)
        wrong_instances = (leaf_instances[leaf_instances['target'] != leaf_node._label_class]).copy()

        sub_tree = DecisionTree(attributes_map)
        sub_tree.fit(wrong_instances)

        sub_leaf_nodes = sub_tree.get_leaves_nodes()
        if len(sub_leaf_nodes) > 1:
            sub_rules = {}
            for sub_leaf_node in sub_leaf_nodes:
                new_rule = ' && '.join(vertical_rules + extract_rules_from_leaf(sub_leaf_node))
                if sub_leaf_node._label_class not in sub_rules.keys():
                    sub_rules[sub_leaf_node._label_class] = set()
                sub_rules[sub_leaf_node._label_class].add(new_rule)
            for sub_target_class in sub_rules.keys():
                sub_rules[sub_target_class] = ' || '.join(sub_rules[sub_target_class])
                if sub_target_class not in rules.keys():
                    rules[sub_target_class] = sub_rules[sub_target_class]
                else:
                    rules[sub_target_class] += ' || ' + sub_rules[sub_target_class]
        # Only root in sub_tree = could not find a suitable split of the root node -> most frequent target is chosen
        elif len(wrong_instances) > 0:  # length 0 could happen since we do not consider missing values for now
            sub_target_class = wrong_instances['target'].mode()[0]
            if sub_target_class not in rules.keys():
                rules[sub_target_class] = ' && '.join(vertical_rules)
            else:
                rules[sub_target_class] += ' || ' + ' && '.join(vertical_rules)

    return rules


def shorten_rules_manually(original_rules, attributes_map):
    """ Rewrites the final rules dictionary to compress many-valued categorical attributes equalities and continuous
    attributes inequalities.

    For example, the series "org:resource = 10 && org:resource = 144 && org:resource = 68" is rewritten as "org:resource
    one of [10, 68, 144]".
    The series "paymentAmount > 21.0 && paymentAmount <= 37.0 && paymentAmount <= 200.0 && amount > 84.0 && amount <=
    138.0 && amount > 39.35" is rewritten as "paymentAmount > 21.0 && paymentAmount <= 37.0 && amount <= 138.0 && amount
    84.0".
    The same reasoning is applied for atoms without '&&s' inside.
    """

    rules = copy.deepcopy(original_rules)

    for target_class in rules.keys():
        or_atoms = rules[target_class].split(' || ')
        new_target_rule = list()
        cat_atoms_same_attr_noand = dict()
        cont_atoms_same_attr_less_noand, cont_atoms_same_attr_greater_noand = dict(), dict()
        cont_comp_less_equal_noand, cont_comp_greater_equal_noand = dict(), dict()

        for or_atom in or_atoms:
            if ' && ' in or_atom:
                and_atoms = or_atom.split(' && ')
                cat_atoms_same_attr = dict()
                cont_atoms_same_attr_less, cont_atoms_same_attr_greater = dict(), dict()
                cont_comp_less_equal, cont_comp_greater_equal = dict(), dict()
                new_or_atom = list()

                for and_atom in and_atoms:
                    a_attr, a_comp, a_value = and_atom.split(' ')
                    # Storing information for many-values categorical attributes equalities
                    if attributes_map[a_attr] == 'categorical' and a_comp == '=':
                        if a_attr not in cat_atoms_same_attr.keys():
                            cat_atoms_same_attr[a_attr] = list()
                        cat_atoms_same_attr[a_attr].append(a_value)
                    # Storing information for continuous attributes inequalities (min/max value for each attribute and
                    # also if the inequality is strict or not)
                    elif attributes_map[a_attr] == 'continuous':
                        if a_comp in ['<', '<=']:
                            if a_attr not in cont_atoms_same_attr_less.keys() or float(a_value) <= float(cont_atoms_same_attr_less[a_attr]):
                                cont_atoms_same_attr_less[a_attr] = a_value
                                cont_comp_less_equal[a_attr] = True if a_comp == '<=' else False
                        elif a_comp in ['>', '>=']:
                            if a_attr not in cont_atoms_same_attr_greater.keys() or float(a_value) >= float(cont_atoms_same_attr_greater[a_attr]):
                                cont_atoms_same_attr_greater[a_attr] = a_value
                                cont_comp_greater_equal[a_attr] = True if a_comp == '>=' else False
                    else:
                        new_or_atom.append(and_atom)

                # Compressing many-values categorical attributes equalities
                for attr in cat_atoms_same_attr.keys():
                    if len(cat_atoms_same_attr[attr]) > 1:
                        new_or_atom.append(attr + ' one of [' + ', '.join(sorted(cat_atoms_same_attr[attr])) + ']')
                    else:
                        new_or_atom.append(attr + ' = ' + cat_atoms_same_attr[attr][0])

                # Compressing continuous attributes inequalities (< / <= and then > / >=)
                for attr in cont_atoms_same_attr_less.keys():
                    min_value = cont_atoms_same_attr_less[attr]
                    comp = ' <= ' if cont_comp_less_equal[attr] else ' < '
                    new_or_atom.append(attr + comp + min_value)

                for attr in cont_atoms_same_attr_greater.keys():
                    max_value = cont_atoms_same_attr_greater[attr]
                    comp = ' >= ' if cont_comp_greater_equal[attr] else ' > '
                    new_or_atom.append(attr + comp + max_value)

                # Or-atom analyzed: putting its new and-atoms in conjunction
                new_target_rule.append(' && ' .join(new_or_atom))

            # If the or_atom does not have &&s inside (single atom), just simplify attributes.
            # For example, the series "org:resource = 10 || org:resource = 144 || org:resource = 68" is rewritten as
            # "org:resource one of [10, 68, 144]". For continuous attributes, follows the same reasoning as before.
            else:
                a_attr, a_comp, a_value = or_atom.split(' ')
                # Storing information for many-values categorical attributes equalities
                if attributes_map[a_attr] == 'categorical' and a_comp == '=':
                    if a_attr not in cat_atoms_same_attr_noand.keys():
                        cat_atoms_same_attr_noand[a_attr] = list()
                    cat_atoms_same_attr_noand[a_attr].append(a_value)
                elif attributes_map[a_attr] == 'continuous':
                    if a_comp in ['<', '<=']:
                        if a_attr not in cont_atoms_same_attr_less_noand.keys() or float(a_value) <= float(cont_atoms_same_attr_less_noand[a_attr]):
                            cont_atoms_same_attr_less_noand[a_attr] = a_value
                            cont_comp_less_equal_noand[a_attr] = True if a_comp == '<=' else False
                    elif a_comp in ['>', '>=']:
                        if a_attr not in cont_atoms_same_attr_greater_noand.keys() or float(a_value) >= float(cont_atoms_same_attr_greater_noand[a_attr]):
                            cont_atoms_same_attr_greater_noand[a_attr] = a_value
                            cont_comp_greater_equal_noand[a_attr] = True if a_comp == '>=' else False
                else:
                    new_target_rule.append(or_atom)

        # Compressing many-values categorical attributes equalities for the 'no &&s' case
        for attr in cat_atoms_same_attr_noand.keys():
            if len(cat_atoms_same_attr_noand[attr]) > 1:
                new_target_rule.append(attr + ' one of [' + ', '.join(sorted(cat_atoms_same_attr_noand[attr])) + ']')
            else:
                new_target_rule.append(attr + ' = ' + cat_atoms_same_attr_noand[attr][0])

        # Compressing continuous attributes inequalities (< / <= and then > / >=) for the 'no &&s' case
        for attr in cont_atoms_same_attr_less_noand.keys():
            min_value = cont_atoms_same_attr_less_noand[attr]
            comp = ' <= ' if cont_comp_less_equal_noand[attr] else ' < '
            new_target_rule.append(attr + comp + min_value)

        for attr in cont_atoms_same_attr_greater_noand.keys():
            max_value = cont_atoms_same_attr_greater_noand[attr]
            comp = ' >= ' if cont_comp_greater_equal_noand[attr] else ' > '
            new_target_rule.append(attr + comp + max_value)

        # Rule for a target class analyzed: putting its new or-atoms in disjunction
        rules[target_class] = ' || '.join(new_target_rule)

    # Rules for all target classes analyzed: returning the new rules dictionary
    return rules


def sampling_dataset(dataset) -> pd.DataFrame:
    """ Performs sampling to obtain a balanced dataset, in terms of target values. """

    dataset = dataset.copy()

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
        dataset = pd.concat(samples, ignore_index=True)

    return dataset
