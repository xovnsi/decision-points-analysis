import pandas as pd
import pm4py
from pm4py.visualization.petri_net import visualizer as pn_visualizer


def import_csv(file_path):
    event_log = pm4py.format_dataframe(pd.read_csv(file_path, sep=';'), case_id='case_id', activity_key='activity',
                                       timestamp_key='timestamp', timest_format='%Y-%m-%d %H:%M:%S%z')
    start_activities = pm4py.get_start_activities(event_log)
    end_activities = pm4py.get_end_activities(event_log)
    print("Start activities: {}\nEnd activities: {}".format(start_activities, end_activities))
    return event_log


def discover_petri_net(log):
    # Alpha Miner
    from pm4py.algo.discovery.alpha import algorithm as alpha_miner
    alpha_net, alpha_initial_marking, alpha_final_marking = alpha_miner.apply(log)
    alpha_graphviz = pn_visualizer.apply(alpha_net, alpha_initial_marking, alpha_final_marking)
    alpha_graphviz.graph_attr['bgcolor'] = 'white'
    pn_visualizer.save(alpha_graphviz, "pn_alpha_miner.png")

    # Inductive Miner
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    ind_net, ind_initial_marking, ind_final_marking = inductive_miner.apply(log)
    ind_graphviz = pn_visualizer.apply(ind_net, ind_initial_marking, ind_final_marking)
    ind_graphviz.graph_attr['bgcolor'] = 'white'
    pn_visualizer.save(ind_graphviz, "pn_inductive_miner.png")


def discover_other_models(log):
    # Process Tree
    process_tree = pm4py.discover_process_tree_inductive(log)
    pm4py.save_vis_process_tree(process_tree, "process_tree.png")

    # BPMN Model
    bpmn_model = pm4py.convert_to_bpmn(process_tree)
    pm4py.save_vis_bpmn(bpmn_model, "bpmn_model.png")

    # Process Map (Directly Follows Graph)
    dfg, start_activities, end_activities = pm4py.discover_dfg(log)
    pm4py.save_vis_dfg(dfg, start_activities, end_activities, "dfg_process_map.png")

    # Heuristic Miner
    heu = pm4py.discover_heuristics_net(log)
    pm4py.save_vis_heuristics_net(heu, "heuristics_net.png")


if __name__ == "__main__":
    e_log = import_csv("running-example.csv")
    discover_petri_net(e_log)
    # discover_other_models(e_log)
