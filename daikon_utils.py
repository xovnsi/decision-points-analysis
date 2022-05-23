import subprocess
import os
import math
from operator import itemgetter


def discovering_branching_conditions(dataset, attributes_map) -> dict:
    """ Alternative method for discovering branching conditions, using Daikon invariant detector

    It uses the existing version of Daikon, since it supports csv files (after a conversion).
    It returns a dictionary containing the discovered rule for each branch.
    Method taken from "Discovering Branching Conditions from Business Process Execution Logs" by Massimiliano de Leoni,
    Marlon Dumas, and Luciano Garcia-Banuelos. In particular, only the CD+IG+LV approach is implemented.
    Modified in order to support not only binary decision points. If binary, the two resulting rules are modified so
    that the one with the lower information gain is the negation of the other.
    Also, invariants that do not make sense are not considered.
    """

    # Saving continuous variables and non-continuous variables names
    feature_names_cont = [c for c in dataset.columns if c != 'target' and attributes_map[c] == 'continuous']
    feature_names_cat = [c for c in dataset.columns if c != 'target' and attributes_map[c] == 'categorical']
    feature_names_bool = [c for c in dataset.columns if c != 'target' and attributes_map[c] == 'boolean']

    # Dropping missing values for continuous variables and filling the other missing values with '?'
    dataset = dataset.dropna(subset=feature_names_cont).fillna('?')
    # for col in feature_names_not_cont:
    #    dataset[col] = dataset[col].astype(pd.StringDtype())

    # Enriching the dataset with latent variables (all combinations of variables using +, -, *, /)
    for i, c1 in enumerate(feature_names_cont):
        # Non-commutative operations are computed also switching the operands
        for c2 in feature_names_cont[i+1:]:
            dataset[c1 + '_plus_' + c2] = dataset[c1] + dataset[c2]
            dataset[c1 + '_minus_' + c2] = dataset[c1] - dataset[c2]
            dataset[c2 + '_minus_' + c1] = dataset[c2] - dataset[c1]
            dataset[c1 + '_times_' + c2] = dataset[c1] * dataset[c2]
            if 0 not in dataset[c2].values:    # to avoid division by zero
                dataset[c1 + '_div_by_' + c2] = dataset[c1] / dataset[c2]
            if 0 not in dataset[c1].values:    # to avoid division by zero
                dataset[c2 + '_div_by_' + c1] = dataset[c2] / dataset[c1]

    # Adding the newly created latent continuous variables to the original list
    feature_names_cont.extend(
        x for x in dataset.columns if any(y in x for y in ['_plus_', '_minus_', '_times_', '_div_by_']))

    # Splitting the dataset according to the target value
    gb = dataset.groupby('target')
    target_datasets = [x for _, x in gb]

    # Extracting the invariants using Daikon
    invariants_list = []
    for td in target_datasets:
        invariants_list.append(_get_daikon_invariants(td))

    # Removing the wrongly reported invariants (the ones comparing continuous and non-continuous variables, or
    # categorical and boolean variables, or categorical and categorical variables, or boolean and boolean variables)
    for i, invariants in enumerate(invariants_list):
        invariants_to_remove = set()
        for inv in invariants:
            if ((any(a in inv for a in feature_names_cont) and any(b in inv for b in feature_names_cat)) or
                    (any(c in inv for c in feature_names_cont) and any(d in inv for d in feature_names_bool)) or
                    (any(e in inv for e in feature_names_cat) and any(f in inv for f in feature_names_bool)) or
                    (sum(g in inv for g in feature_names_cat) > 1) or (sum(g in inv for g in feature_names_bool) > 1)):
                invariants_to_remove.add(inv)
        invariants_list[i] = list(set(invariants) - invariants_to_remove)

    # Building the conjunctive expression for each branch of the decision point
    conj_expressions = []
    for invariants in invariants_list:
        conj_expressions.append(_build_conj_expr(target_datasets, invariants))

    # If the decision point is binary, adjusting the expressions to ensure that one is the negation of the other
    if len(target_datasets) == 2:
        conj_expressions = _adjust_conditions(target_datasets, conj_expressions)

    # Rewriting the operands for the latent variables (i.e. '_plus_' becomes '+' etc.)
    for i, ce in enumerate(conj_expressions):
        conj_expressions[i] = _clean_latent_variables(ce)

    return dict(zip(list(gb.groups.keys()), conj_expressions))


def _get_daikon_invariants(dataset) -> list:
    """ Extracting the invariants from a set of observation instances related to a branch of a decision point

    After exporting the DataFrame as a csv file, a Perl script is launched to create the input files for Daikon in the
    proper format. Then, Daikon is called to discover the invariants. Finally, the extracted invariants are cleaned
    to be used later and returned as a list.
    """

    dataset.drop(columns=['target']).to_csv(path_or_buf='dataset.csv', index=False)
    subprocess.run(['perl', 'daikon-5.8.10/scripts/convertcsv.pl', 'dataset.csv'])
    subprocess.run(['java', '-cp', 'daikon-5.8.10/daikon.jar', 'daikon.Daikon', '--nohierarchy', '-o', 'invariants.inv',
                    '--no_text_output', '--noversion', '--omit_from_output', 'r', 'dataset.dtrace', 'dataset.decls'])
    inv = subprocess.run(['java', '-cp', 'daikon-5.8.10/daikon.jar', 'daikon.PrintInvariants',
                          'invariants.inv'], capture_output=True, text=True)
    invariants = []
    for line in inv.stdout.splitlines():
        if not any(x in line for x in ["===", "aprogram.point:::POINT", "one of {"]):
            invariants.append(line)

    for file_name in ['dataset.csv', 'dataset.dtrace', 'dataset.decls', 'invariants.inv']:
        try:
            os.remove(file_name)
        except FileNotFoundError:
            continue

    return invariants


def _build_conj_expr(sets, invariants) -> str | None:
    """ Builds a conjunctive expression starting from the invariants found.

    The resulting conjunctive expression is built using a greedy approach. The first atom selected is the one with the
    highest information gain. Then, iteratively, the atom with the highest information gain is added, provided that the
    resulting conjunctive expression, adding that atom, increases the information gain.
    The resulting conjunctive expression is then returned as a string.
    """

    if len(invariants) == 0:
        return None
    else:
        atom_information_gains = []
        for inv in invariants:
            atom_information_gains.append((inv, _compute_information_gain(sets, [inv])))

        element_with_max_gain = max(atom_information_gains, key=itemgetter(1))
        resulting_expr = [element_with_max_gain[0]]
        atom_information_gains.remove(element_with_max_gain)

        while len(atom_information_gains) > 0:
            element_with_max_gain = max(atom_information_gains, key=itemgetter(1))
            new_predicate = resulting_expr + [element_with_max_gain[0]]
            if _compute_information_gain(sets, new_predicate) > _compute_information_gain(sets, resulting_expr):
                resulting_expr.append(element_with_max_gain[0])
            atom_information_gains.remove(element_with_max_gain)

        return ' && '.join(resulting_expr)


def _compute_entropy(sets) -> float:
    """ Computes the entropy of the sets of observation instances """

    if any(len(s) == 0 for s in sets):
        return 0
    else:
        fractions = []
        total_size = sum(len(s) for s in sets)
        for s in sets:
            fractions.append(len(s) / total_size)
        terms = []
        for f in fractions:
            terms.append(- f * math.log2(f))
        return sum(terms)


def _compute_information_gain(sets, predicate) -> float:
    """ Computes the information gain of predicate given the sets of observation instances. """

    predicate = ' & '.join(predicate)

    sets_pred = []
    sets_not_pred = []
    for i, s in enumerate(sets):
        sets_pred.append(s.query(predicate))
        sets_not_pred.append(s[~s.apply(tuple, 1).isin(sets_pred[i].apply(tuple, 1))])

    term_1 = (sum(len(s) for s in sets_pred) * _compute_entropy(sets_pred)) / sum(len(s) for s in sets)
    term_2 = (sum(len(s) for s in sets_not_pred) * _compute_entropy(sets_not_pred)) / sum(len(s) for s in sets)

    return _compute_entropy(sets) - term_1 - term_2


def _adjust_conditions(sets, conj_exprs) -> list:
    """ Adjusts the two conjunctive expression so that one is the negation of the other.

    If one of them is empty, then it is set to the negation of the other. If they are both non-empty, then the one with
    the lower information gain is set to the negation of the other.
    """

    conj_expr_1, conj_expr_2 = conj_exprs

    if conj_expr_1 is not None and conj_expr_2 is not None:
        expr_1_list = conj_expr_1.split(' && ')
        expr_2_list = conj_expr_2.split(' && ')
        info_gain_expr_1 = _compute_information_gain(sets, expr_1_list)
        info_gain_expr_2 = _compute_information_gain(sets, expr_2_list)

        if info_gain_expr_1 > info_gain_expr_2:
            conj_expr_2 = _negate_expr(conj_expr_1)
        else:
            conj_expr_1 = _negate_expr(conj_expr_2)
        return [conj_expr_1, conj_expr_2]
    elif conj_expr_1 is None and conj_expr_2 is not None:
        return [_negate_expr(conj_expr_2), conj_expr_2]
    elif conj_expr_1 is not None and conj_expr_2 is None:
        return [conj_expr_1, _negate_expr(conj_expr_1)]
    else:
        return ['None', 'None']


def _negate_expr(expr) -> str:
    """ Returns the negation of an expression.

    If the expression contains multiple atoms in conjunction, it places 'not' before the original expression.
    Otherwise, it negates the operand of the expression.
    """

    if ' && ' in expr:
        return 'not (' + expr + ')'
    else:
        ops = [' == ', ' != ', ' > ',  ' < ', ' >= ', ' <= ']
        neg_ops = [' != ', ' == ', ' <= ', ' >= ', ' < ', ' > ']

        for i, o in enumerate(ops):
            if o in expr:
                split = expr.split(o)
                return neg_ops[i].join(split)


def _clean_latent_variables(conj_expr) -> str:
    """ Rewrites the dummy names for latent variables using the corresponding operands. """

    ops = ['_plus_', '_minus_', '_times_', '_div_by_']
    ops_symb = [' + ', ' - ', ' * ', ' / ']
    for j, o in enumerate(ops):
        if o in conj_expr:
            conj_expr = conj_expr.replace(o, ops_symb[j])

    return conj_expr
