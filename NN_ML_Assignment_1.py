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

# tab 1: EDA summary-2-3 graphs
# tab 2: Correlation plot, chloropeth map, any other insight related graph, show outliers
# tab 3: Scikit learn model

plt.figure(figsize=(12, 4))
sns.boxplot(x='education', y='age', hue='income', data=df_adult_new,palette='twilight')
plt.xticks(rotation=45)
plt.title('Age Distribution by Education and Income')
st.pyplot(plt.gcf())

plt.figure(figsize=(12, 6))
sns.boxplot(x='income', y='age', data=df_adult_new,hue = 'sex')
plt.title('Age vs Income')
st.pyplot(plt.gcf())

plt.figure(figsize=(12, 4))
sns.barplot(x='education', y='hours-per-week', hue='income', data=df_adult_new,palette ='Purples')
plt.xticks(rotation=45)
plt.title('Hours per Week by Education and Income')
st.pyplot(plt.gcf())

edu_income = df_adult_new.groupby('education')['income'].value_counts(normalize=True).unstack()
plt.figure(figsize=(12, 6))
edu_income.plot(kind='bar', stacked=True, color=['grey', 'purple'])
plt.title('Income Levels by Education')
plt.xlabel('Education')
plt.legend(title='Income')
st.pyplot(plt.gcf())

sns.set_palette(sns.color_palette("Dark2", 8))
sns.pairplot(data=df_adult_new,hue='income',corner=True)
st.pyplot(plt.gcf())

workclass_income = df_adult_new.groupby('workclass')['income'].value_counts(normalize=True).unstack()
plt.figure(figsize=(12, 6))
workclass_income.plot(kind='bar',color=['grey', 'blue'])
plt.title('Income Levels by Workclass')
plt.xlabel('Workclass')
plt.legend(title='Income')
st.pyplot(plt.gcf())
