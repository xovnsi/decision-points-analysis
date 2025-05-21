import random
from datetime import datetime, timedelta
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import os

log = EventLog()
base_time = datetime.now()
num_traces = 100

for i in range(num_traces):
    trace = Trace()
    trace.attributes["concept:name"] = f"Trace {i}"
    start_time = base_time + timedelta(days=i)

    # Common start
    e1 = Event({
        "concept:name": "Start",
        "time:timestamp": start_time,
        "cost": random.uniform(0, 100),
        "is_manual": random.choice([True, False])
    })
    trace.append(e1)

    # Decision point based on cost
    if e1["cost"] > 50:
        e2 = Event({
            "concept:name": "Path_A",
            "time:timestamp": start_time + timedelta(minutes=5),
            "cost": e1["cost"],
            "is_manual": e1["is_manual"]
        })
    else:
        e2 = Event({
            "concept:name": "Path_B",
            "time:timestamp": start_time + timedelta(minutes=5),
            "cost": e1["cost"],
            "is_manual": e1["is_manual"]
        })

    trace.append(e2)

    # Common end
    e3 = Event({
        "concept:name": "End",
        "time:timestamp": start_time + timedelta(minutes=10),
        "cost": e1["cost"],
        "is_manual": e1["is_manual"]
    })
    trace.append(e3)

    log.append(trace)

# Ensure the logs/ directory exists
os.makedirs("logs", exist_ok=True)

# Export log to XES
xes_exporter.apply(log, "logs/log-test.xes")
print("✅ Generated logs/log-test.xes")
