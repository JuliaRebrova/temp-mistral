import streamlit as st
import yaml
import pandas as pd
st.set_page_config(layout="wide")

with open(f"gpt_groq_compare.yml", "r") as f:
    data = yaml.safe_load(f)


qws = list(data.keys())


for qw in qws:
    request = qw 
    st.markdown(request)
    col1, col2 = st.columns(2)

    with col1:
        model_name = list(data[request].keys())[0]
        st.markdown(model_name)
        st.json(data[request][model_name])

    with col2:
        model_name = list(data[request].keys())[1]
        st.markdown(model_name)
        st.json(data[request][model_name])