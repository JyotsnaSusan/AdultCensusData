import os
import base64
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as pximport 
import folium
from streamlit.components.v1 import html as components_html
from sklearn import preprocessing
import pickle

df_adult_new = pd.read_excel('Cleaned_Census_Data.xlsx')

st.set_page_config(layout="wide")

# tab 1: EDA summary-2-3 graphs
# tab 2: Correlation plot, chloropeth map, any other insight related graph, show outliers
# tab 3: Scikit learn model

tab1, tab2, tab3 = st.tabs(["Feature Importance", "Key Feature", "Prediction Model"])

with tab1:
    sns.set_palette(sns.color_palette("Dark2", 8))
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
    sns.boxplot(x='education', y='age', hue='income', data=df_adult_new, palette='twilight', ax=axes[0, 0])
    axes[0, 0].set_title('Age Distribution by Education and Income')

    sns.boxplot(x='workclass', y='age', hue='workclass', data=df_adult_new, palette='Set3', ax=axes[1, 1])
    axes[1, 1].set_title('Age Distribution by Workclass')

    edu_income = df_adult_new.groupby('education')['income'].value_counts(normalize=True).unstack()
    edu_income.plot(kind='bar', stacked=True, color=['grey', 'purple'], ax=axes[1, 0])
    axes[1, 0].set_title('Income Levels by Education')
    axes[1, 0].set_xlabel('Education')
    axes[1, 0].legend(title='Income')

    workclass_income = df_adult_new.groupby('workclass')['income'].value_counts(normalize=True).unstack()
    workclass_income.plot(kind='bar', color=['grey', 'blue'], ax=axes[0, 1])
    axes[0, 1].set_title('Income Levels by Workclass')
    axes[0, 1].set_xlabel('Workclass')
    axes[0, 1].legend(title='Income')
    st.write("A quick analysis of the 4 most common influencers of income showed that this data set did not contain any of the usual markers of income. Education seems to have a mild positive influence on income, but in all other cases, there are no clear trends.")
    plt.tight_layout()  
    st.pyplot(plt.gcf())

with tab2:
    st.write("We then used fnlwgt which is the number of people a variable represents to find per capita income and project density of incomes over 50K and under 50K by using a world map. What we saw was that almost anyone with income over 50K was in the US or Canada, even after adjusting for population size. This is therefore our key feature")
    df_adult_new['fnl_pct']=(df_adult_new['fnlwgt'] / df_adult_new['fnlwgt'].sum())*100    
    df_country_wt=df_adult_new.groupby(['nativecountry','income'])[['fnl_pct','fnlwgt']].sum().reset_index()
    df_country_wt['income'] = df_country_wt['income'].replace({'<=50K': 50000, '>50K': 10000})
    df_country_wt['nativecountry'] = df_country_wt['nativecountry'].replace({'United-States': 'USA'})
    df_country_wt['nativecountry'] = df_country_wt['nativecountry'].replace({'Dominican-Republic': 'Dominican Republic'})
    df_country_wt['nativecountry'] = df_country_wt['nativecountry'].replace({'Holand-Netherlands': 'Netherlands'})
    df_country_wt['nativecountry'] = df_country_wt['nativecountry'].replace({'Outlying-US(Guam-USVI-etc)': 'Puerto Rico'})
    df_country_wt['nativecountry'] = df_country_wt['nativecountry'].replace({'El-Salvador': 'El Salvador'})
    df_country_wt['nativecountry'] = df_country_wt['nativecountry'].replace({'Trinadad&Tobago': 'Trinidad and Tobago'})
    df_country_wt['nativecountry'] = df_country_wt['nativecountry'].replace({'Puerto-Rico': 'Puerto Rico'})
    
    df_country_wt['Wtd_income']=df_country_wt['income']*df_country_wt['fnl_pct']
    
    bins = df_country_wt['fnlwgt'].quantile([0, 0.5, 0.7, 0.9, 1]).tolist()
    
    world_geo = r'world_countries.json'

    world_map = folium.Map(location=[0, 0], zoom_start=2, width='100%', height='800px')  

    folium.Choropleth(
        geo_data=world_geo,
        data=df_country_wt,
        columns=['nativecountry', 'fnlwgt'],
        key_on='feature.properties.name',
        fill_color='RdYlGn',
        nan_fill_color="beige",
        fill_opacity=0.9,
        line_opacity=0.2,
        bins=bins).add_to(world_map)

    map_html = 'world_map.html'
    world_map.save(map_html)

    with open(map_html, 'r') as f:
        html_code = f.read()

    components_html(html_code, height=600)