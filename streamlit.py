import streamlit as st
import pandas as pd
import re
df = pd.read_csv('final.csv')

ids = df.id.to_list()

selected_id = st.selectbox('Chose an ID', ids)

dialogue = df[df['id'] == selected_id].dialogue.values[0]
summary = df[df['id'] == selected_id].final_summary.values[0]

dialogue = re.sub(r"( )([a-zA-Z]+:)", r"\n\n\g<2> ", dialogue)


col1, col2 = st.columns(2)

with col1:
   st.header("Dialogue")
   dialogue

with col2:
   st.header("Summary")
   summary