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

#plt.figure(figsize=(12, 4))
#sns.boxplot(x='workclass', y='age', hue='workclass', data=df_adult_new, palette='Set3')
#st.pyplot(plt.gcf())

# tab 1: EDA summary-2-3 graphs
# tab 2: Correlation plot, chloropeth map, any other insight related graph, show outliers
# tab 3: Scikit learn model



# Tejashree's Trial 

#df_adult_new.hist( bins=10, sharey=True, figsize=[50,50],xlabelsize=20,ylabelsize=20)
#st.pyplot(plt.gcf())

#st.write("Box Plot")
#df_adult_new.boxplot(column='education', by='income')
#st.pyplot(plt.gcf())

styled_htmlTS = f"<div style='font-family: Roboto, sans-serif; font-size: 16px; text-align: center;'></div><br><br>"
st.write(styled_htmlTS, unsafe_allow_html=True)
st.write("Outlier Detection")
st.write("Age and Income Analysis")
plt.figure(figsize=(5, 4))
#df_adult_new.boxplot(column='age', by='income')
sns.boxplot(x='income', y='age', hue='income', data=df_adult_new, palette='Set2',width=0.5)
st.pyplot(plt.gcf())

st.write("Race and Hours Per week")
plt.figure(figsize=(10, 5))
plt.scatter('race', 'hours-per-week', data=df_adult_new)
st.pyplot(plt.gcf())


st.write("Occupation and Hours Per week")
plt.figure(figsize=(20, 5))
plt.scatter('occupation', 'hours-per-week', data=df_adult_new)
st.pyplot(plt.gcf())

st.write("Corelation")
sns.heatmap(df_adult_new.corr(method='pearson'),cmap='coolwarm', annot=True)
st.pyplot(plt.gcf())