import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import random
import json
import argparse
from datetime import datetime, timedelta

def parse_args():
    parser = argparse.ArgumentParser(description="Generate customizable event logs.")
    parser.add_argument("-n", "--num-traces", type=int, default=100,
                        help="Number of traces to generate (default: 100)")
    parser.add_argument("-a", "--num-attributes", type=int, default=3,
                        help="Number of attributes per event (default: 3)")
    parser.add_argument("-o", "--output-name", type=str, default="test",
                        help="Output base name for log and attributes files (default: test)")
    return parser.parse_args()

def generate_random_value(attr_type):
    if attr_type == "continuous":
        return round(random.uniform(0, 100), 2)
    elif attr_type == "boolean":
        return random.choice([True, False])
    elif attr_type == "categorical":
        return random.choice(["Category1", "Category2", "Category3"])
    else:
        return None

def generate_trace(attributes):
    # Use a balanced condition for decision_label
    if attributes.get("attr_0", 0) > 50:
        decision_label = random.choice(["A", "B"])  # randomly pick so tree sees variation
        if decision_label == "A":
            trace_events = ["Start", "Task_A1", "Task_A2"]
        else:
            trace_events = ["Start", "Task_A3", "Task_A4"]
    else:
        decision_label = random.choice(["C", "D"])
        if decision_label == "C":
            trace_events = ["Start", "Task_B1", "Task_B2"]
        else:
            trace_events = ["Start", "Task_B3", "Task_B4"]

    # Add parallel tasks and loops as before...

    trace_events.append("End")
    attributes["decision_label"] = decision_label
    return trace_events


def main():
    args = parse_args()

    NUM_TRACES = args.num_traces
    NUM_ATTRIBUTES = args.num_attributes
    OUTPUT_NAME = args.output_name

    ATTRIBUTE_TYPES = []
    types = ["continuous", "boolean", "categorical"]
    for i in range(NUM_ATTRIBUTES):
        ATTRIBUTE_TYPES.append(types[i % len(types)])

    log = EventLog()

    for trace_idx in range(NUM_TRACES):
        # Generate attributes for this trace
        attributes = {}
        for i in range(NUM_ATTRIBUTES):
            attr_name = f"attr_{i}"
            attributes[attr_name] = generate_random_value(ATTRIBUTE_TYPES[i])

        # Generate event sequence depending on attributes, also adds decision_label
        event_sequence = generate_trace(attributes)

        trace = Trace()
        trace.attributes["concept:name"] = f"Trace_{trace_idx+1}"

        timestamp = datetime.now()
        for event_name in event_sequence:
            event = Event()
            event["concept:name"] = event_name

            # Add all attributes including decision_label to each event
            for attr_name, attr_value in attributes.items():
                event[attr_name] = attr_value

            event["time:timestamp"] = timestamp
            timestamp += timedelta(minutes=1)

            trace.append(event)
        log.append(trace)

    # Export the log
    log_file_path = f"logs/log-{OUTPUT_NAME}.xes"
    xes_exporter.apply(log, log_file_path)
    print(f"Log exported to {log_file_path}")

    # Save attribute types to JSON for your decision tree input (including decision_label type)
    attr_map = {f"attr_{i}": ATTRIBUTE_TYPES[i] for i in range(NUM_ATTRIBUTES)}
    attr_map["decision_label"] = "categorical"
    attr_file_path = f"dt-attributes/{OUTPUT_NAME}.attr"
    with open(attr_file_path, "w") as f:
        json.dump(attr_map, f, indent=4)
    print(f"Attribute map saved to {attr_file_path}")

if __name__ == "__main__":
    main()
