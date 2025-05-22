import argparse
import json
import datetime
from datetime import timedelta


class Trace:
    def __init__(self):
        # Define possible process attributes here
        self.attribute_space = {
            "speed": [1.0, 5.0],
            "color": ["red", "blue", "black", "silver"],
            "manual": [True, False],
            "ac": ["yes", "no"]
        }

        self.attributes = {}
        self.path = []
        self.timestamp = datetime.datetime.now()

        import random
        for key, values in self.attribute_space.items():
            if len(values) == 2 and all(isinstance(v, float) for v in values):
                self.attributes[key] = random.uniform(values[0], values[1])
            else:
                self.attributes[key] = random.choice(values)


class Condition:
    def __init__(self, attribute_name, operator, value, output):
        self.attribute_name = attribute_name
        self.operator = operator  # e.g. ">", "==", "<=", etc.
        self.value = value
        self.output = output

    def is_satisfied(self, trace):
        actual = trace.attributes.get(self.attribute_name)
        if actual is None:
            return False
        return eval(f"{repr(actual)} {self.operator} {repr(self.value)}")


class ConditionAnd:
    def __init__(self, conditions, output):
        self.conditions = conditions
        self.output = output

    def is_satisfied(self, trace):
        return all(c.is_satisfied(trace) for c in self.conditions)


class Node:
    def __init__(self, name, conditions, default_output):
        self.name = name
        self.conditions = conditions
        self.default_output = default_output

    def execute(self, trace):
        for c in self.conditions:
            if c.is_satisfied(trace):
                trace.path.append(c.output)
                return
        trace.path.append(self.default_output)


def parse_args():
    parser = argparse.ArgumentParser(description="Define log generation parameters")
    parser.add_argument("-n", "--num-traces", type=int, default=100,
                        help="Number of traces to generate (default: 100)")
    parser.add_argument("-o", "--output-name", type=str, default="out",
                        help="Output base name for log and attributes files (default: out)")
    return parser.parse_args()


def trace_to_xes(trace, concept_name):
    current_time = trace.timestamp
    delta = timedelta(minutes=2)

    xes_trace = f"""  <trace>
    <string key="concept:name" value="{concept_name}" />
"""


    for node in trace.path:
        xes_trace += f"""    <event>
      <string key="concept:name" value="{node}"/>
      <date key="time:timestamp" value="{current_time.isoformat()}+01:00"/>\n"""

        for key, value in trace.attributes.items():
            value_type = "boolean" if isinstance(value, bool) else "string"
            xes_trace += f'      <{value_type} key="{key}" value="{str(value).lower()}"/>\n'

        xes_trace += "    </event>\n"
        current_time += delta

    xes_trace += "  </trace>\n"
    return xes_trace


def run_trace(node_map, start, index):
    trace = Trace()
    trace.path = [start]
    retval = ""

    while trace.path[-1] is not None:
        node_map[trace.path[-1]].execute(trace)
    trace.path = trace.path[:-1]
    retval += trace_to_xes(trace, f"{index:06}")

    return retval


def generate_logs(nodes, start, num_traces):
    node_map = {node.name: node for node in nodes}
    log = """<?xml version="1.0" encoding="utf-8" ?>
<log xes.version="1849-2016" xes.features="nested-attributes" xmlns="http://www.xes-standard.org/">\n"""

    for i in range(num_traces):
        log += run_trace(node_map, start, i)

    log += "</log>\n"

    return log


def infer_list_types(trace):
    result = {}
    for key, values in trace.attribute_space.items():
        if all(isinstance(v, bool) for v in values):
            result[key] = "boolean"
        elif all(isinstance(v, (float, int)) for v in values):
            result[key] = "continuous"
        elif all(isinstance(v, str) for v in values):
            result[key] = "categorical"
        else:
            result[key] = "unknown"
    return json.dumps(result, indent=4)





# Define model title
args = parse_args()
num_traces = args.num_traces
title = args.output_name

# Define nodes and their xor conditions
nodes = [
    Node("A", [
        ConditionAnd([
            Condition("speed", ">", 3.0, None),
            Condition("ac", "==", "yes", None),
        ], "B")
    ], "C"),
    Node("B", [
        Condition("color", "==", "black", "D"),
        Condition("color", "==", "silver", "D"),
    ], "E"),
    Node("C", [
        Condition("manual", "==", True, "G"),
    ], "H"),
    Node("D", [], "F"),
    Node("E", [], "F"),
    Node("F", [], None),
    Node("G", [], "I"),
    Node("H", [], "I"),
    Node("I", [], None),
]



print("Log file name:", title)
print("Number of traces:", num_traces)
print("Defined attributes:", Trace().attribute_space.keys())
print("Defined nodes:", [n.name for n in nodes])

# Saves log traces
with open(f"logs/log-{title}.xes", "w", encoding="utf-8") as f:
    f.write(generate_logs(nodes, "A", num_traces))

# Saves attr categories
with open(f"dt-attributes/{title}.attr", "w", encoding="utf-8") as f:
    f.write(infer_list_types(Trace()))

print("\nLogs generated.")





