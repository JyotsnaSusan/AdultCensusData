import os
import base64
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as pximport 

df_adult_new= pd.read_excel('Cleaned_Census_Data.xlsx')

st.set_page_config(layout="wide")

plt.figure(figsize=(12, 4))
sns.boxplot(x='workclass', y='age', hue='workclass', data=df_adult_new, palette='Set3')
st.pyplot(plt.gcf())
