import os
import json
import pm4py
import shutil
import streamlit as st
import streamlit.components.v1 as components
import subprocess
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from streamlit_utils import get_unique_values_log, create_dict, save_json, build_datasets, rules_computation, start_server
from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests

st.set_page_config(layout="wide")

def main():
    start_server()
    if 'uploaded_log_name' not in st.session_state:
        st.session_state['uploaded_log_name'] = None

    st.title("Decision Points Analysis")
    uploaded_event_log = st.file_uploader('Choose the event log file in .xes format', type='xes')

    if uploaded_event_log is not None:
        if st.session_state['uploaded_log_name'] != uploaded_event_log.name.split('.')[0]:
            # Initialize session state
            st.session_state['uploaded_log_name'] = uploaded_event_log.name.split('.xes')[0]
            st.session_state['list_trace_attr'], st.session_state['list_event_attr'] = list(), list()
            st.session_state['attr_config_saved'] = False
            st.session_state['net'] = None
            st.session_state['decision_points_data'] = None
            st.session_state['results_selection'] = False

            # Store uploaded event log temporarily and import it
            if not os.path.exists('streamlitTemp'):
                os.makedirs('streamlitTemp')
            with open(os.path.join('streamlitTemp', st.session_state['uploaded_log_name']), 'wb') as f:
                f.write(uploaded_event_log.getbuffer())
            with st.spinner("Parsing the log..."):
                st.session_state.log = xes_importer.apply('streamlitTemp/{}'.format(st.session_state['uploaded_log_name']))
            shutil.rmtree('./streamlitTemp')

            # Extracting unique values from the log
            with st.spinner("Analyzing dataset..."):
                st.session_state['df_trace'], st.session_state['df_event'], unique_values_trace, unique_values_event = get_unique_values_log()
                _ = [st.session_state.list_trace_attr.append('t_{}'.format(value)) for i, value in enumerate(unique_values_trace.keys())]
                _ = [st.session_state.list_event_attr.append('e_{}'.format(value)) for i, value in enumerate(unique_values_event.keys()) if value != 'time:timestamp']

        # Showing tables with unique values
        st.write('First 10 unique values of trace attributes: ', st.session_state['df_trace'].head(10),
                 'First 10 unique values of events attributes: ', st.session_state['df_event'].head(10))

        # Attributes types selection
        st.header("Attributes types configuration")
        st.write("Select the right type for each attribute.")
        col_event, col_trace = st.columns(2)
        with col_trace:
            st.subheader("Trace attributes")
            for name in st.session_state['list_trace_attr']:
                st.selectbox(name.split('t_')[1], ('categorical', 'continuous', 'boolean'), key=name)
        with col_event:
            st.subheader("Event attributes")
            for name in st.session_state['list_event_attr']:
                st.selectbox(name.split('e_')[1], ('categorical', 'continuous', 'boolean'), key=name)

        # Saving configuration button
        if st.button('Save Configuration') and not st.session_state['attr_config_saved']:
            dict_conf = create_dict()
            save_json(dict_conf)

            # Converting the attributes types in the log according to the attributes_map file
            attributes_map_file = '{}.attr'.format(st.session_state['uploaded_log_name'])
            with open(os.path.join('streamlitTemp', attributes_map_file), 'r') as f:
                json_string = json.load(f)
                st.session_state['attributes_map'] = json.loads(json_string)
            shutil.rmtree('./streamlitTemp')

            attr_conversion_error = set()
            for trace in st.session_state['log']:
                for event in trace:
                    for attribute in event.keys():
                        if attribute in st.session_state['attributes_map']:
                            if st.session_state['attributes_map'][attribute] == 'continuous':
                                try:
                                    event[attribute] = float(event[attribute])
                                except ValueError:
                                    attr_conversion_error.add(attribute)
                            elif st.session_state['attributes_map'][attribute] == 'boolean':
                                event[attribute] = bool(event[attribute])

            if attr_conversion_error:
                for attribute in attr_conversion_error:
                    st.error("Attribute '{}' does not appear to be a continuous one. Is it categorical? Please recheck the configuration.".format(attribute), icon="\U0001F6A8")
                os.remove(os.path.join('dt-attributes', attributes_map_file))
            else:
                st.session_state['attr_config_saved'] = True

        if st.session_state['attr_config_saved']:
            # Importing the Petri net model or extracting one
            st.header("Petri Net model")
            if st.session_state['net'] is None:
                try:
                    st.write("Looking for an existing Petri Net model...")
                    st.session_state['net'], st.session_state['im'], st.session_state['fm'] = pnml_importer.apply("models/{}.pnml".format(st.session_state['uploaded_log_name']))
                except (FileNotFoundError, OSError):
                    st.warning("Existing Petri Net model not found. "
                               "If you wish to use an already created model, put the .pnml file inside a 'models' "
                               "folder, using the same name as the one of the log file.", icon="\U000026A0")
                    st.write("Extracting a Petri Net model using the Inductive Miner...")
                    st.session_state['net'], st.session_state['im'], st.session_state['fm'] = pm4py.discover_petri_net_inductive(st.session_state['log'])

                st.session_state['net_graph_image'] = pn_visualizer.apply(st.session_state['net'], st.session_state['im'], st.session_state['fm'])
                st.session_state['net_graph_image'].graph_attr['bgcolor'] = 'white'
            st.graphviz_chart(st.session_state['net_graph_image'])


            # view and save initial BPMN model
            st.session_state['initial_bpmn_diagram'] = pm4py.convert_to_bpmn(
                st.session_state['net'],
                st.session_state['im'],
                st.session_state['fm'],
            )
            st.header("Initial BPMN diagram")
            st.session_state['initial_bpmn_graph_image'] = bpmn_visualizer.apply(st.session_state['initial_bpmn_diagram'], parameters={'font_size': 10})
            st.graphviz_chart(st.session_state['initial_bpmn_graph_image'])
            if not os.path.exists('tmp'):
                os.mkdir('tmp')
            bpmn_exporter.apply(st.session_state['initial_bpmn_diagram'], 'tmp/initial_in.bpmn')

            # Extracting the decision points datasets
            if st.session_state.decision_points_data is None:
                st.session_state.decision_points_data = build_datasets()

            # Method selection
            st.header("Rules extraction")
            st.text("Choose the preferred method.")
            st.session_state.method = st.selectbox('Decision Trees or Daikon?', ('Choose one...', 'Decision Trees', 'Daikon'))
            if st.session_state.method == 'Decision Trees':
                st.session_state.pruning = st.selectbox('Pruning method?', ('Choose one...', 'Rules simplification', 'Pessimistic', 'No Pruning'))
                st.session_state.overlapping = st.selectbox('Overlapping rules?', ('Choose one...', 'Yes', 'No'))
                if st.session_state.pruning != 'Choose one...' and st.session_state.overlapping != 'Choose one...':
                    st.session_state.results_selection = True
            elif st.session_state.method == 'Daikon':
                st.session_state.method = 'Daikon'
                st.session_state.results_selection = True

            # If the method has been selected, then compute the results
            if st.button('Start rules computation'):
                if not st.session_state.results_selection:
                    st.error("Please make a selection first.", icon="\U0001F6A8")
                else:
                    output_file_name = rules_computation()
                    st.success("A text file containing the results has been saved with the name '{}'".format(output_file_name), icon="\U00002705")
                    with open(output_file_name, "rb") as f:
                        st.download_button("Download results", f, file_name=output_file_name)

                # display updated bpmn with the rules
                # 
                st.components.v1.iframe("http://localhost:8000/modeler.html", height=800)



if __name__ == '__main__':
    main()
