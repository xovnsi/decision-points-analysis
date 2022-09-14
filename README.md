# Decision Points Analysis
#### *A novel method to perform Decision Points Analysis considering multiple paths that process instances could follow inside the Petri Net model.*

Understanding and analyzing decisions inside a business process can provide valuable insights about the process itself.
The knowledge about process executions is often stored in Event Logs: each process instance corresponds to a trace, and
each trace contains the sequence of events that took place. The stored data contains not only the name of the event, but
also additional information, such as a timestamp, the resource who executed it, and any additional attribute depending
on the specific process.

Process Mining techniques allow to automatically extract a model of the process, looking at the sequence of events (and
their relationship) in the Event Log. The model can then be extended with rules steering the decisions. Each rule is a
boolean (True/False) condition on the process attributes: if the rule is satisfied, then the process instance can follow
that specific path in the model. The Decision Mining field gathers this family of methods.

Among the different modelling notations, Petri Nets are the ones we are interested in. A Petri Net is a directed
bipartite graph with two types of nodes: places and transitions. Places are represented by white circles, and they
correspond to states of the process; transitions are represented by white rectangles, and they correspond to events.
Places and transitions are connected through arcs. A Petri Net representing a process usually contains a start place and
a sink place: every process instance begins in the start place, follow some path, and then terminates in the sink place.
There could also be so-called invisible transitions, which does not correspond to any event in the Event Log: they only
serve for routing purposes in the obtained model, and they are represented by black rectangles.

In the context of Petri Nets, Decision Mining is referred to as Decision Points Analysis. A decision point is a place
with more than one outgoing arc: when a process execution finds itself at a decision point, it can follow a different
path according to the value of its attributes. The goal of Decision Points Analysis is to extract the rules that guide
these decisions.

In order to do this, state-of-the-art techniques consider every decision point as a classification problem, which can be
solved using Decision Trees. The training set related to a decision point contains an instance for each traversal of that
decision point; the features are the process attributes, while the target is the decision that has been taken, which is the
transition followed immediately after. To find decision points traversals, one must know the path followed by a trace in
the model.

This novel method allows to consider multiple paths inside the model, exploiting invisible transitions. In fact, given a
set of visible transitions recorded in the log, the model could allow multiple different paths traversing invisible
transitions. These alternative paths may or may not have been followed by the process instance, since invisible transitions
are not recorded in the Event Log.

### Can I try it out?

Sure. A working version of the algorithm is available as a web app via Streamlit at [this page](https://savoiadiego-decision-points-analysis-streamlit-dpa-28jy5l.streamlitapp.com/).

You just need to upload the Event Log file (.xes format only at the moment) and it will show the first 10 unique values
for each attribute. You must select the proper type for each attribute (categorical, continuous or boolean) through the
displayed dropdown boxes.

After a successful selection, a Petri Net model of the process is automatically extracted using the Inductive Miner
implementation in the [PM4PY library](https://pm4py.fit.fraunhofer.de/).

Finally, you can select which method the algorithm should use in order to extract rules. You can choose between the standard
Decision Tree classifiers or the Daikon invariants detector. The former can also be enhanced with pruning methodologies,
in order to avoid gigantic rules when possible. Two pruning approaches are available, both proposed by [Ross Quinlan](https://en.wikipedia.org/wiki/Ross_Quinlan).
The pessimistic one works by pruning the fitted tree, while the rule simplification one works more directly on rules.
You can also choose to discover overlapping rules, as proposed in [this paper](https://research.tue.nl/en/publications/decision-mining-revisited-discovering-overlapping-rules-2).

The discovered set of rules for each decision point is then shown. At the end, you can also download a text file containing
the results. Note that the rules extraction process may take a long time, depending on the size and dimensionality of the
uploaded Event Log.

*The C4.5 algorithm for Decision Tree classification has been implemented in a custom version in order to correctly handle
all types of attributes. It is available as a standalone version at [this GitHub page](https://github.com/piepor/C4.5-Decision-Trees).*
