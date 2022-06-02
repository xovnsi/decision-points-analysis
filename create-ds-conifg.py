import streamlit as st
import numpy as np
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
import time
import copy

uploaded_file = st.file_uploader("Choose a file")

# st.cache does not work, using experimental_memo instead
@st.experimental_memo
def import_data(file_name):
    log = xes_importer.apply('data/{}'.format(file_name))
    return log

#def save_json(

if uploaded_file is not None:
    #with st.spinner('Importing dataset, have a coffee and relax!'):
    log = import_data(uploaded_file.name)
        #log = xes_importer.apply('data/{}'.format(uploaded_file.name))
    #df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    #st.write(df)

    with st.spinner("Analysing dataset, don't worry be happy!"):
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
        st.write('First 10 unique values of trace attributes: ', df_trace.head(10),
                'First 10 unique values of events attributes: ', df.head(10))

    col_event, col_trace = st.columns(2)
    with col_trace:
        st.header("Trace attributes")
        for value in unique_values_trace.keys():
            st.selectbox(value, ('continuous', 'categorical', 'boolean'), key='t_{}'.format(value))
    with col_event:
        st.header("Event attributes")
        for value in unique_values.keys():
            if value != 'time:timestamp':
                st.selectbox(value, ('continuous', 'categorical', 'boolean'), key='e_{}'.format(value))


