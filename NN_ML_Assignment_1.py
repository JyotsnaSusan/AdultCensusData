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
from prediction import predict
import category_encoders as ce
import os

current_directory = os.path.dirname(__file__)


df_adult_new = pd.read_excel('Cleaned_Census_Data.xlsx')
df_Undersample = pd.read_excel('Undersampled_Model.xlsx')


st.set_page_config(layout="wide")

banner = "<div style='background-color: #90be6d; padding: 10px; text-align: center;'><h2><font color=white>Income Prediction By Demographics: A Machine Learning Intitiative</h2></div>"
st.write(banner, unsafe_allow_html=True)

with st.sidebar:
    sidebarletters="<div><Font size=3><font color=black>The income of a person is influenced by various social, political and geographic factors. In this assignment we have attempted to eliminate all other noise and only focus on demographic factors that influence income. The model used a combination of Undersampling and Random Forest Classification to determine the income of an individual with a 79.5% accuracy score"
    st.write(sidebarletters,unsafe_allow_html=True)


tab1, tab2, tab3, tab4, tab5,tab6, tab7  = st.tabs(["Feature Importance","Feature Selection", "Key Feature","Imputation","Outlier Detection","Correlation Matrix", "Prediction Model"])


with tab1:
    tab1write="<div><Font size=4><font color=black>A quick analysis of the 4 most common influencers of income showed that this data set did not contain any of the usual markers of income. <br>Education seems to have a mild positive influence on income, but in all other cases, there are no clear trends.<br></div>"
    st.write(tab1write,unsafe_allow_html=True)
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
    plt.tight_layout()  
    st.pyplot(plt.gcf())

with tab2:
    tab2write="<div><Font size=4><font color=black>Although we expected income to increase with hours per week. This feature did not seem to have a direct influence on income either..<br></div>"
    st.write(tab2write,unsafe_allow_html=True)
    sns.set_palette(sns.color_palette("twilight", 8))
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
    sns.scatterplot(x='age', y='hours_per_week', hue='income', palette ='Purples',data=df_adult_new,ax=axes[0, 0])
    axes[0, 0].set_title('Age Distribution by Hours per Week and Income')

    sns.barplot(x='education', y='hours_per_week', hue='income', data=df_adult_new,palette ='Purples', ax=axes[1, 1])
    axes[1, 1].set_title('Hours per Week by Education and Income')

    sns.boxplot(x='income', y='age', data=df_adult_new,hue = 'sex',palette ='Purples', ax=axes[0, 1])
    axes[0, 1].set_title('Age vs Income')

    race_count= df_adult_new['race'].value_counts()
    axes[1,0].pie(race_count, labels=race_count.index, autopct ='%.2f%%',wedgeprops={'edgecolor': 'black'})
    axes[1, 0].set_title('Race Distribution')
    plt.tight_layout()  
    st.pyplot(fig)

with tab6:
    tab6write="<div><Font size=4><font color=black>Using all these insights as our base, a correlation matrix was then drawn up to help us determine which features our model could use to best predict the income of an individual<br></div>"
    st.write(tab6write,unsafe_allow_html=True)
    cat_col = df_adult_new.select_dtypes(include=['object']).columns
    num_col = df_adult_new.select_dtypes(exclude=['object']).columns
    encoder = ce.OrdinalEncoder()
    encoded_cat= encoder.fit_transform(df_adult_new[cat_col])
    combined_df = pd.concat([df_adult_new[num_col], encoded_cat], axis=1)
    correlation_matrix = combined_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='twilight',fmt=".2f", annot_kws={"size": 8})
    st.pyplot(plt.gcf())



with tab3:
    tab3write="<div><Font size=4><font color=black>We then used fnlwgt which is the number of people a variable represents to find per capita income and project density of incomes over 50K and under 50K by using a world map.<br> What we saw was that almost anyone with income over 50K was in the US or Canada, even after adjusting for population size. Therefore, this became our key feature<br></div>"
    st.write(tab3write,unsafe_allow_html=True)
    
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

    world_map = folium.Map(location=[0, 0], zoom_start=2, width='100%', height=800)  

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
    
with tab4:
    col1, col2 = st.columns((0.65,0.35))  
    with col1:
        Edu_path = os.path.join(current_directory, "Education.PNG")
        Edu= open(Edu_path, "rb").read()
        st.image(Edu, width=700)
        tab4_1write="<div><Font size=4><font color=black>Education was translated into number as set forward by worldbanks' education department</div>"
        st.write(tab4_1write,unsafe_allow_html=True)

        Occ_path = os.path.join(current_directory, "Occupation.PNG")      
        Occ= open(Occ_path, "rb").read()
        st.image(Occ, width=700)
        tab4_2write="<div><Font size=4><font color=black>Occupation, work class and native country values were filled using on a manually determined cluster, based on our previous feature analysis</div>"
        st.write(tab4_2write,unsafe_allow_html=True)

        Acc_path = os.path.join(current_directory, "Accuracy.PNG")          
        Acc= open(Acc_path, "rb").read()
        st.image(Acc, width=700)
        tab4_3write="<div><Font size=4><font color=black>All these steps lead to an accuracy score of 79.5% using both logistic regression and random forest algorithms</div>"
        st.write(tab4_3write,unsafe_allow_html=True)

    with col2:
        tab4_4write="<div><Font size=4><font color=black>We ran our code through a second method of imputation using the KNN algorithm, after determining that 79% accuracy could be achieved using 12 neighbours. However, as shown in the previous graphs, 79% accuracy was already achieved through our original cluster.</div>"
        st.write(tab4_4write,unsafe_allow_html=True)

        df_knn = pd.read_csv("MeanAccKNN.csv")

        values = df_knn['Value']
        serial_numbers = df_knn['Serial Number']
    
        fig1 = plt.figure()  
        plt.plot(serial_numbers, values, 'g', marker='o')
        plt.xlabel('Number Of neighbours')
        plt.ylabel('Accuracy')
        plt.title('KKN Number of Neighbours Vs Accuracy')
        plt.tight_layout()
        st.pyplot(fig1)
    
with tab7:
    col5, col6 = st.columns((0.8,0.2))  
    
    with col5:

        age1 = st.selectbox("Age", df_Undersample['age'].unique()) 
        workclass1 = st.selectbox("Workclass", df_Undersample['workclass'].unique()) 
        occupation1 = st.selectbox("Occupation", df_Undersample['occupation'].unique()) 
        hours_per_week1 = st.selectbox("Hours per Week", df_Undersample['hours_per_week'].unique())
        education1 = st.selectbox("Education", df_Undersample['education'].unique()) 
        nativecountry1 = st.selectbox("Native Country", df_Undersample['nativecountry'].unique()) 
        maritalstatus1 = st.selectbox("Marital Status", df_Undersample['marital_status'].unique()) 
        gender = st.selectbox("Gender", df_Undersample['sex'].unique()) 
        CG = st.selectbox("Capital Gains: Yes Or No", df_Undersample['CG_Category'].unique()) 


    
        if st.button("Predict Income"):
            encoder = ce.OrdinalEncoder(cols=['workclass', 'education', 'nativecountry', 'occupation','marital_status','sex'])
            data = {
                'workclass': [workclass1],
                'education': [education1],
                'nativecountry': [nativecountry1],
                'occupation': [occupation1],
                'marital_status':[maritalstatus1],
                'sex':[gender]

                   }
            encoded_data = encoder.fit_transform(pd.DataFrame(data))
            input_data = np.concatenate((encoded_data.values[0], [hours_per_week1],[age1],[CG]))
            result = predict(np.array([input_data]))
            st.text(result[0])
        
        with col6:
            tab7write="<div><Font size=4><font color=black>Although hyper-parameter tuning and hypothesis testing of various models led us to believe that XGBoost would be the perfect algorithm for data with this much skew, we were not able to self-learn this model. <br>Instead we used Undersampling methods to remove the primary noise in this database and allow the machine to pick up on the secondary trend of demographics-related predictors of income.</div>"
            st.write(tab7write,unsafe_allow_html=True)

        
with tab5:
    col3, col4 = st.columns((0.65,0.35))  
    
    with col3:
        
        cat_col = df_adult_new.select_dtypes(include=['object']).columns
        num_col = df_adult_new.select_dtypes(exclude=['object']).columns

        num_rows = (len(num_col) + 1) // 2 

        fig2, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(20, 30))

        axes_flat = axes.flatten()

        for i, column in enumerate(num_col):
            ax = axes_flat[i]
            ax.boxplot(df_adult_new[column], patch_artist=True)
            ax.set_title(f'Box plot of {column}')

        if len(num_col) % 2 != 0:
            axes_flat[-1].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig2)
        
    with col4:
        tab5write="<div><Font size=4><font color=black>We also tried removing outliers like those in age brackets over 90. But the population size of the outliers was too small to reduce skew or improve accuracy. Our model continued to show a 79.5% accuracy despite these changes.</div>"
        st.write(tab5write,unsafe_allow_html=True)

        age_path = os.path.join(current_directory, "Age.PNG")
        
        age= open(age_path, "rb").read()
        st.image(age, width=500)
        


