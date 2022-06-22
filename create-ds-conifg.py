import streamlit as st
import numpy as np
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
import time
import copy
import json
import os

if 'flag_uploaded_file' not in st.session_state:
    st.session_state['flag_uploaded_file'] = False
    st.session_state['uploaded_file'] = None

uploaded_file = st.file_uploader("Choose a file")

# st.cache does not work, using experimental_memo instead
@st.experimental_memo
def import_data(file_name):
    log = xes_importer.apply('data/{}'.format(file_name))
    st.session_state.flag_uploaded_file = True
    return log

@st.experimental_memo
def analyse_ds(_log):
    unique_values_trace = dict()
    unique_values = dict()
    for trace in log:
        for trace_attr in trace.attributes:
            if trace_attr in unique_values_trace.keys():
                unique_values_trace[trace_attr].add(trace.attributes[trace_attr])
            else:
                unique_values_trace[trace_attr] = {trace.attributes[trace_attr]}
        for event in trace:
            for ev_attr in event:
                if ev_attr in unique_values.keys():
                    unique_values[ev_attr].add(event[ev_attr])
                else:
                    unique_values[ev_attr] = {event[ev_attr]}
    max_len = 0
    max_val = None
    for value in unique_values.keys():
        if len(unique_values[value]) > max_len:
            max_len = len(unique_values[value]) 
            max_val = value
    for value in unique_values.keys():
        old_values = list(copy.copy(unique_values[value]))
        old_values.extend([np.nan]*(max_len - len(unique_values[value])))
        unique_values[value] = old_values
    for value in unique_values_trace.keys():
        old_values = list(copy.copy(unique_values_trace[value]))
        old_values.extend([np.nan]*(max_len - len(unique_values_trace[value])))
        unique_values_trace[value] = old_values
    df_trace = pd.DataFrame.from_dict(unique_values_trace)
    df = pd.DataFrame.from_dict(unique_values)

    return df_trace, df, unique_values, unique_values_trace

def save_json(dict_conf, data_dir='dt-attributes'):
    json_string = json.dumps(dict_conf)
    #breakpoint()
    file_name = '{}.attr'.format(st.session_state.uploaded_file.split('log-')[1].split('.xes')[0])
    with open(os.path.join(data_dir, file_name), 'w') as file:
        json.dump(json_string, file)

def create_dict(st_session_state):
    dict_conf = dict()
    #breakpoint()
    for name in st_session_state:
        if ('e_' in name or 't_' in name) and name.split('_')[1] not in ['trace', 'event']:
            # remove initial 'e_' or 't_' from the name
            dict_conf["_".join(name.split('_')[1:])] = st.session_state[name]
    return dict_conf

if uploaded_file is not None:
    #with st.spinner('Importing dataset, have a coffee and relax!'):
    if st.session_state.uploaded_file != uploaded_file.name:
        log = import_data(uploaded_file.name)
        with st.spinner("Analysing dataset, don't worry be happy!"):
            df_trace, df, unique_values, unique_values_trace = analyse_ds(log)
            st.write('First 10 unique values of trace attributes: ', df_trace.head(10),
                    'First 10 unique values of events attributes: ', df.head(10))
        col_event, col_trace = st.columns(2)
        with col_trace:
            list_trace_attr = list()
            st.header("Trace attributes")
            for i, value in enumerate(unique_values_trace.keys()):
                st.selectbox(value, ('continuous', 'categorical', 'boolean'), key='t_{}'.format(value))
                list_trace_attr.append('t_{}'.format(value))
        with col_event:
            list_event_attr = list()
            st.header("Event attributes")
            for i, value in enumerate(unique_values.keys()):
                if value != 'time:timestamp':
                    st.selectbox(value, ('continuous', 'categorical', 'boolean'), key='e_{}'.format(value))
                    list_event_attr.append('e_{}'.format(value))
        st.session_state.df = df
        st.session_state.df_trace = df_trace
        st.session_state.list_trace_attr = list_trace_attr
        st.session_state.list_event_attr = list_event_attr
    else:
        st.write('First 10 unique values of trace attributes: ', st.session_state.df_trace.head(10),
                'First 10 unique values of events attributes: ', st.session_state.df.head(10))
        col_event, col_trace = st.columns(2)
        with col_trace:
            st.header("Trace attributes")
            for name in st.session_state.list_trace_attr:
                if 't_' in name:
                    st.selectbox(name.split('t_')[1], ('continuous', 'categorical', 'boolean'), key=name)
        with col_event:
            st.header("Event attributes")
            for name in st.session_state.list_event_attr:
                st.selectbox(name.split('e_')[1], ('continuous', 'categorical', 'boolean'), key=name)

    st.session_state.uploaded_file = uploaded_file.name
        #log = xes_importer.apply('data/{}'.format(uploaded_file.name))
    #df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    #st.write(df)
#    else:
#        unique_values = st.session_state.unique_values
#        unique_values_trace = st.session_state.unique_values_trace


    if st.button('Save Configuration'):
        dict_conf = create_dict(st.session_state)
        save_json(dict_conf)

#st.session_state.flag_uploaded_file = False
#if st.session_state.uploaded_file is None:
#    st.session_state.uploaded_file = uploaded_file
