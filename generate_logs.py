import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import random
import json
import argparse
from datetime import datetime, timedelta

def parse_args():
    parser = argparse.ArgumentParser(description="Generate complex event logs with concurrency, branching, and loops.")
    parser.add_argument("-n", "--num-traces", type=int, default=100,
                        help="Number of traces to generate (default: 100)")
    parser.add_argument("-a", "--num-attributes", type=int, default=3,
                        help="Number of attributes per event (default: 3)")
    parser.add_argument("-o", "--output-name", type=str, default="complex",
                        help="Output base name for log and attributes files (default: complex)")
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
    """
    Generates one trace with branching, loops, and concurrency based on attribute values.
    """
    trace_events = ["Start"]

    # Branching: choose between two paths based on attr_0
    if attributes.get("attr_0", 0) > 50:
        trace_events += ["Task_A1", "Task_A2"]
    else:
        trace_events += ["Task_B1", "Task_B2"]

    # Parallel tasks (interleaved randomly)
    parallel_tasks = ["Parallel_1", "Parallel_2", "Parallel_3"]
    random.shuffle(parallel_tasks)
    trace_events += parallel_tasks

    # Optional loop: repeat "Review" 0 to 2 times randomly
    for _ in range(random.randint(0, 2)):
        trace_events.append("Review")

    trace_events.append("End")

    return trace_events

def main():
    args = parse_args()

    NUM_TRACES = args.num_traces
    NUM_ATTRIBUTES = args.num_attributes
    OUTPUT_NAME = args.output_name

    # Define attribute types cyclically
    types = ["continuous", "boolean", "categorical"]
    ATTRIBUTE_TYPES = [types[i % len(types)] for i in range(NUM_ATTRIBUTES)]

    log = EventLog()

    for trace_idx in range(NUM_TRACES):
        # Generate attribute values for this trace
        attributes = {}
        for i in range(NUM_ATTRIBUTES):
            attr_name = f"attr_{i}"
            attributes[attr_name] = generate_random_value(ATTRIBUTE_TYPES[i])

        # Generate the event sequence for the trace
        event_sequence = generate_trace(attributes)

        # Build the trace
        trace = Trace()
        trace.attributes["concept:name"] = f"Trace_{trace_idx+1}"

        # Timestamps: start now, increment 1 min per event
        timestamp = datetime.now()

        for event_name in event_sequence:
            event = Event()
            event["concept:name"] = event_name
            # Add all attributes to each event
            for attr_name, attr_value in attributes.items():
                event[attr_name] = attr_value
            event["time:timestamp"] = timestamp
            timestamp += timedelta(minutes=1)

            trace.append(event)
        log.append(trace)

    # Export log to XES
    log_file_path = f"logs/log-{OUTPUT_NAME}.xes"
    xes_exporter.apply(log, log_file_path)
    print(f"Log exported to {log_file_path}")

    # Save attribute types to JSON for decision tree usage
    attr_map = {f"attr_{i}": ATTRIBUTE_TYPES[i] for i in range(NUM_ATTRIBUTES)}
    attr_file_path = f"dt-attributes/{OUTPUT_NAME}.attr"
    with open(attr_file_path, "w") as f:
        json.dump(attr_map, f, indent=4)
    print(f"Attribute map saved to {attr_file_path}")

if __name__ == "__main__":
    main()
