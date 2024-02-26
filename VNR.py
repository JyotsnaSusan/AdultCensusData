import os
import base64
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
quality = pd.read_excel('VNR Quality Index dashboard database.xlsx')
gov=pd.read_excel('SDG National Governance dashboard Database.xlsx',sheet_name="Dataset")
quality.rename(columns={
    'Country (ENG)': 'countryeng',
    'Dimension (ENG)': 'dim',
    'Value (rating)': 'value',
    'Value category (ENG)': 'category',
    'Country (SPA)': 'countryspa'
}, inplace=True)
gov.rename(columns={
   'Country (ESP)': 'country_S',
    'Country (ENG)': 'country_E',
    'Dimensión (ESP)': 'Dim_S',
    'Dimension (ENG)': 'dim_E',
    'Value': 'val',
    'Índice (columna de validación)': 'idx'    
},inplace=True)
df_quality = quality[['countryeng', 'dim', 'value', 'category', 'Year']]
df_gov=gov[gov.columns]

# Streamlit UI
st.set_page_config(layout="wide")
logo_image = open("Logo_azul_final.png", "rb").read()

st.image(logo_image, width=200)

banner = "<div style='background-color: #24263f; padding: 10px; text-align: center;'><h2><font color=white>VNR Index-2030 Sustainable Development Goals</h2></div>"
st.write(banner, unsafe_allow_html=True)




tab1, tab2 = st.tabs(["VNR", "GovernanceIndex"])
with tab1:
    selected_tab = st.selectbox("Select Language", ["VNR(English)", "VNR(Spanish)"])

    if selected_tab == "VNR(English)":

        col1, col2,col3 = st.columns((0.4,0.4,0.2))  

        with col1:

            st.markdown("#### Select Country to view VNR Dashboard")
            countrylist = df_quality["countryeng"].unique().tolist()
            selected_country = st.selectbox("Country 1", countrylist, index=0)  # Use a unique widget ID

            yearlist = df_quality[df_quality["countryeng"] == selected_country]["Year"].unique().tolist()
            selected_year = st.selectbox("Year 1", yearlist, index=0)  # Use a unique widget ID


            st.markdown(f"<h4>Value Ratings for: {selected_year} {selected_country}</h4>", unsafe_allow_html=True)


            df_country_year = df_quality[(df_quality["countryeng"] == selected_country) & (df_quality["Year"] == selected_year)]

            df_country = df_quality[(df_quality["countryeng"] == selected_country)]
            grouped_df = df_country.groupby('Year')['value'].mean().reset_index()
            grouped_df.set_index('Year', inplace=True)
            grouped_df = grouped_df.sort_index(ascending=False)
            grouped_df['% Difference'] = grouped_df['value'].pct_change(periods=-1) * 100
            grouped_df['% Difference'] = grouped_df['% Difference'].fillna(0)  
            grouped_df['% Difference'] = grouped_df['% Difference'].round(1)
            grouped_df['% Difference'] = grouped_df['% Difference'].apply(lambda x: f'{x:.1f}% ↑' if x > 0 else f'{x:.1f}% ↓' if x < 0 else f'{x:.1f}%')


            htmlVNR = grouped_df.to_html(index=True, header=False, justify='center', index_names=False)
            styled_htmlVNR = f"<div style='font-family: Roboto, sans-serif; font-size: 16px; text-align: center;'>{htmlVNR}</div><br><br>"
            st.write(styled_htmlVNR, unsafe_allow_html=True)




            filteredcountries = df_country_year.groupby(['dim', 'category'])['value'].mean().reset_index()
            filteredcountries = filteredcountries.sort_values(by='value', ascending=True)


            unique_categories = filteredcountries['category'].unique()

            custom_palette = {
                'Low': '#fc4811',
                'Medium-Low': '#e77d25',
                'Medium-High': '#5978a2',
                'High': '#404d86'
            }

            category_to_color = {category: custom_palette.get(category, 'gray') for category in unique_categories}
            category_to_color[None] = 'gray'

            plt.figure(figsize=(6, 4))
            sns.set_style('darkgrid')

            # Set max value
            max_val = max(filteredcountries['value']) * 1.3
            ax = plt.subplot(projection='polar')

            # Set the subplot 
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(1)
            ax.set_rlabel_position(0)
            ax.set_thetagrids([], labels=[])
            ax.set_rgrids(range(len(filteredcountries)), labels=filteredcountries['dim'])

            ax = plt.subplot(projection='polar')

            colors = [category_to_color[cat] for cat in filteredcountries['category']]

            for i in range(len(filteredcountries)):
                ax.barh(i, list(filteredcountries['value'])[i] * 2 * np.pi / max_val, color=colors[i])

            st.pyplot(plt.gcf())
            
            filteredcountries_subset = filteredcountries[['dim', 'value']]
            filteredcountries_subset.set_index('dim', inplace=True)
            htmlVNR7 = filteredcountries_subset.to_html(index=True, header=False, justify='left', index_names=False)
            styled_htmlVNR7 = f"<div style='font-family: Roboto; font-size: 14px; text-align: left;'>{htmlVNR7}</div><br><br>"
            st.write(styled_htmlVNR7, unsafe_allow_html=True)


        with col2:
            st.markdown("#### Make Selections for Comparison dashboard")
            countrylist2 = df_quality["countryeng"].unique().tolist()
            selected_country2 = st.selectbox("Country 2", countrylist2, index=0)  # Use a unique widget ID

            yearlist2 = df_quality[df_quality["countryeng"] == selected_country2]["Year"].unique().tolist()
            selected_year2 = st.selectbox("Year 2", yearlist2, index=0)

            df_country_year2 = df_quality[(df_quality["countryeng"] == selected_country2) & (df_quality["Year"] == selected_year2)]

            st.markdown(f"<h4>Value Ratings for: {selected_year2} {selected_country2}</h4>", unsafe_allow_html=True)
            filteredcountries2 = df_country_year2.groupby(['dim', 'category'])['value'].mean().reset_index()
            filteredcountries2 = filteredcountries2.sort_values(by='value', ascending=True)

            df_country2 = df_quality[(df_quality["countryeng"] == selected_country2)]

            grouped_df2 = df_country2.groupby('Year')['value'].mean().reset_index()
            grouped_df2.set_index('Year', inplace=True)
            grouped_df2 = grouped_df2.sort_index(ascending=False)  # Use grouped_df2 here
            grouped_df2['% Difference'] = grouped_df2['value'].pct_change(periods=-1) * 100
            grouped_df2['% Difference'] = grouped_df2['% Difference'].fillna(0)
            grouped_df2['% Difference'] = grouped_df2['% Difference'].round(1)
            grouped_df2['% Difference'] = grouped_df2['% Difference'].apply(lambda x: f'{x:.1f}% ↑' if x > 0 else f'{x:.1f}% ↓' if x < 0 else f'{x:.1f}%')

            htmlVNR2 = grouped_df2.to_html(index=True, header=False, justify='center', index_names=False)
            styled_htmlVNR2 = f"<div style='font-family: Roboto, sans-serif; font-size: 16px; text-align: center;'>{htmlVNR2}</div><br><br>"
            st.write(styled_htmlVNR2, unsafe_allow_html=True)



            unique_categories2 = filteredcountries2['category'].unique()

            custom_palette2 = {
                'Low': '#fc4811',
                'Medium-Low': '#e77d25',
                'Medium-High': '#5978a2',
                'High': '#404d86'
            }

            category_to_color2 = {category2: custom_palette2.get(category2, 'gray') for category2 in unique_categories2}
            category_to_color2[None] = 'gray'


            plt.figure(figsize=(6, 4))
            sns.set_style('darkgrid')

            # Set max value
            max_val2 = max(filteredcountries2['value']) * 1.3
            ax2 = plt.subplot(projection='polar')

            # Set the subplot 
            ax2.set_theta_zero_location('N')
            ax2.set_theta_direction(1)
            ax2.set_rlabel_position(0)
            ax2.set_thetagrids([], labels=[])
            ax2.set_rgrids(range(len(filteredcountries2)), labels=filteredcountries2['dim'])


            colors2 = [category_to_color2[cat2] for cat2 in filteredcountries2['category']]


            for i in range(len(filteredcountries2)):
                ax2.barh(i, list(filteredcountries2['value'])[i] * 2 * np.pi / max_val2, color=colors2[i])

            st.pyplot(plt.gcf())
            
            filteredcountries2_subset = filteredcountries2[['dim', 'value']]
            filteredcountries2_subset.set_index('dim', inplace=True)
            htmlVNR8 = filteredcountries2_subset.to_html(index=True, header=False, justify='left', index_names=False)
            styled_htmlVNR8 = f"<div style='font-family: Roboto; font-size: 14px; text-align: left;'>{htmlVNR8}</div><br><br>"
            st.write(styled_htmlVNR8, unsafe_allow_html=True)



        with col3:
            grouped_by_country = df_quality[df_quality['Year'] == selected_year].groupby('countryeng')['value'].sum().reset_index()

            grouped_by_country = grouped_by_country.sort_values(by='value', ascending=True)

            st.markdown(f"<h4>Country Ranking: {selected_year}</h4>", unsafe_allow_html=True)

            fig3 = px.bar(
                grouped_by_country,
                x='value',
                y='countryeng',
                orientation='h',  # Horizontal orientation
                text='value',
                title='',
                color_discrete_sequence=['#5978a2']  # Set the color for the bars
                )

            fig3.update_layout(
                xaxis_title='',
                yaxis_title='',
                font=dict(size=14),
                legend=dict(font=dict(size=14)),
                yaxis=dict(tickfont=dict(size=14)),
                xaxis=dict(tickfont=dict(size=14), showticklabels=False),
                bargap=0.1,
                margin=dict(l=0, r=0, t=5, b=40),
                height=400,
                width=230,
                showlegend=False  # Hide the legend
            )

            for bar in fig3.data:
                bar.textposition = 'auto'
                bar.texttemplate = '%{text:.2f}'

            st.plotly_chart(fig3)


        # START OF SPANISH TAB

    qualityS = pd.read_excel('VNR Quality Index dashboard database.xlsx')
    govS=pd.read_excel('SDG National Governance dashboard Database.xlsx',sheet_name="Dataset")

    qualityS.rename(columns={
        'Country (SPA)': 'countryspa',
        'Dimension (SPA)': 'dimspa',
        'Value (rating)': 'value',
        'Value category (SPA)': 'categoryspa',
        'Country (SPA)': 'countryspa'
    }, inplace=True)
    govS.rename(columns={
       'Country (ESP)': 'country_S',
        'Country (ENG)': 'country_E',
        'Dimensión (ESP)': 'Dim_S',
        'Dimension (ENG)': 'dim_E',
        'Value': 'val',
        'Índice (columna de validación)': 'idx'    
    },inplace=True)

    df_qualitys = qualityS[['countryspa', 'dimspa', 'value', 'categoryspa', 'Year']]
    df_govs=gov[gov.columns]


    if selected_tab == "VNR(Spanish)":

        col4, col5,col6 = st.columns((0.4,0.4,0.2)) 

        with col4:

            st.markdown("#### Seleccione el país para ver el panel VNR")
            countrylists = df_qualitys["countryspa"].unique().tolist()
            selected_countrys = st.selectbox("País", countrylists, index=0)

            yearlists = df_qualitys[df_qualitys["countryspa"] == selected_countrys]["Year"].unique().tolist()
            selected_years = st.selectbox("Año", yearlists, index=0)

            st.markdown(f"<h4>Calificaciones de valor para: {selected_years} {selected_countrys}</h4>", unsafe_allow_html=True)


            df_country_years = df_qualitys[(df_qualitys["countryspa"] == selected_countrys) & (df_qualitys["Year"] == selected_years)]

            df_countrys = df_qualitys[(df_qualitys["countryspa"] == selected_countrys)]
            grouped_dfs = df_countrys.groupby('Year')['value'].mean().reset_index()
            grouped_dfs.set_index('Year', inplace=True)
            grouped_dfs = grouped_dfs.sort_index(ascending=False)
            grouped_dfs['% Diferencia'] = grouped_dfs['value'].pct_change(periods=-1) * 100
            grouped_dfs['% Diferencia'] = grouped_dfs['% Diferencia'].fillna(0)  
            grouped_dfs['% Diferencia'] = grouped_dfs['% Diferencia'].round(1)
            grouped_dfs['% Diferencia'] = grouped_dfs['% Diferencia'].apply(lambda x: f'{x:.1f}% ↑' if x > 0 else f'{x:.1f}% ↓' if x < 0 else f'{x:.1f}%')



            htmlVNRs = grouped_dfs.to_html(index=True, header=False, justify='center', index_names=False)
            styled_htmlVNRs = f"<div style='font-family: Roboto, sans-serif; font-size: 16px; text-align: center;'>{htmlVNRs}</div><br>"
            st.write(styled_htmlVNRs, unsafe_allow_html=True)




            sfilteredcountries = df_country_years.groupby(['dimspa', 'categoryspa'])['value'].mean().reset_index()
            sfilteredcountries = sfilteredcountries.sort_values(by='value', ascending=True)


            Sunique_categories = sfilteredcountries['categoryspa'].unique()

            custom_paletteS = {
                'Bajo': '#fc4811',
                'Mediano Bajo': '#e77d25',
                'Mediano Alto': '#5978a2',
                'Alto': '#404d86'
            }

            Scategory_to_color = {categorys: custom_paletteS.get(categorys, 'gray') for categorys in Sunique_categories}
            Scategory_to_color[None] = 'gray'



            plt.figure(figsize=(6, 4))
            sns.set_style('darkgrid')

            # Set max value
            max_val3 = max(sfilteredcountries['value']) * 1.3
            axS = plt.subplot(projection='polar')

            # Set the subplot 
            axS.set_theta_zero_location('N')
            axS.set_theta_direction(1)
            axS.set_rlabel_position(0)
            axS.set_thetagrids([], labels=[])
            axS.set_rgrids(range(len(sfilteredcountries)), labels=sfilteredcountries['dimspa'])

            axS = plt.subplot(projection='polar')

            Scolors = [Scategory_to_color[Scat] for Scat in sfilteredcountries['categoryspa']]

            for i in range(len(sfilteredcountries)):
                axS.barh(i, list(sfilteredcountries['value'])[i] * 2 * np.pi / max_val3, color=Scolors[i])

            st.pyplot(plt.gcf())
            
            Sfilteredcountries_subset = sfilteredcountries[['dimspa', 'value']]
            Sfilteredcountries_subset.set_index('dimspa', inplace=True)
            htmlVNR9 = Sfilteredcountries_subset.to_html(index=True, header=False, justify='left', index_names=False)
            styled_htmlVNR9 = f"<div style='font-family: Roboto; font-size: 14px; text-align: left;'>{htmlVNR9}</div><br><br>"
            st.write(styled_htmlVNR9, unsafe_allow_html=True)

            
        with col5:

            st.markdown("#### Elija el panel de comparación")
            countrylists2 = df_qualitys["countryspa"].unique().tolist()
            selected_countrys2 = st.selectbox("País 2", countrylists2, index=0)  # Change the widget ID

            yearlists2 = df_qualitys[df_qualitys["countryspa"] == selected_countrys2]["Year"].unique().tolist()
            selected_years2 = st.selectbox("Año 2", yearlists2, index=0)  # Change the widget ID

            st.markdown(f"<h4>Calificaciones de valor para: {selected_years2} {selected_countrys2}</h4>", unsafe_allow_html=True)

            df_country_years2 = df_qualitys[(df_qualitys["countryspa"] == selected_countrys2) & (df_qualitys["Year"] == selected_years2)]

            df_countrys2 = df_qualitys[(df_qualitys["countryspa"] == selected_countrys2)]
            grouped_dfs2 = df_countrys2.groupby('Year')['value'].mean().reset_index()  # Change the variable name
            grouped_dfs2.set_index('Year', inplace=True)
            grouped_dfs2 = grouped_dfs2.sort_index(ascending=False)
            grouped_dfs2['% Diferencia'] = grouped_dfs2['value'].pct_change(periods=-1) * 100
            grouped_dfs2['% Diferencia'] = grouped_dfs2['% Diferencia'].fillna(0)  
            grouped_dfs2['% Diferencia'] = grouped_dfs2['% Diferencia'].round(1)
            grouped_dfs2['% Diferencia'] = grouped_dfs2['% Diferencia'].apply(lambda x: f'{x:.1f}% ↑' if x > 0 else f'{x:.1f}% ↓' if x < 0 else f'{x:.1f}%')

            htmlVNRs2 = grouped_dfs2.to_html(index=True, header=False, justify='center', index_names=False)
            styled_htmlVNRs2 = f"<div style='font-family: Roboto, sans-serif; font-size: 16px; text-align: center;'>{htmlVNRs2}</div><br>"
            st.write(styled_htmlVNRs2, unsafe_allow_html=True)

            sfilteredcountries2 = df_country_years2.groupby(['dimspa', 'categoryspa'])['value'].mean().reset_index()
            sfilteredcountries2 = sfilteredcountries2.sort_values(by='value', ascending=True)

            Sunique_categories2 = sfilteredcountries2['categoryspa'].unique()

            custom_paletteS2 = {  # Change the variable name to custom_paletteS2
                'Bajo': '#fc4811',
                'Mediano Bajo': '#e77d25',
                'Mediano Alto': '#5978a2',
                'Alto': '#404d86'
            }

            Scategory_to_color2 = {categorys: custom_paletteS2.get(categorys, 'gray') for categorys in Sunique_categories2}  # Change the variable name to 
            Scategory_to_color2[None] = 'gray'

            plt.figure(figsize=(6, 4))
            sns.set_style('darkgrid')

        # Set max value
            max_val4 = max(sfilteredcountries2['value']) * 1.3
            axS2 = plt.subplot(projection='polar')

        # Set the subplot 
            axS2.set_theta_zero_location('N')
            axS2.set_theta_direction(1)
            axS2.set_rlabel_position(0)
            axS2.set_thetagrids([], labels=[])
            axS2.set_rgrids(range(len(sfilteredcountries2)), labels=sfilteredcountries2['dimspa'])

            Scolors2 = [Scategory_to_color2[Scat2] for Scat2 in sfilteredcountries2['categoryspa']]  # Change the variable name to Scategory_to_color2


            for i in range(len(sfilteredcountries2)):
                axS2.barh(i, list(sfilteredcountries2['value'])[i] * 2 * np.pi / max_val4, color=Scolors2[i])


            st.pyplot(plt.gcf())

            Sfilteredcountries_subset2 = sfilteredcountries2[['dimspa', 'value']]
            Sfilteredcountries_subset2.set_index('dimspa', inplace=True)
            htmlVNR10 = Sfilteredcountries_subset2.to_html(index=True, header=False, justify='left', index_names=False)
            styled_htmlVNR10 = f"<div style='font-family: Roboto; font-size: 14px; text-align: left;'>{htmlVNR10}</div><br><br>"
            st.write(styled_htmlVNR10, unsafe_allow_html=True)


        with col6:
            grouped_by_country2 = df_qualitys[df_qualitys['Year'] == selected_years].groupby('countryspa')['value'].sum().reset_index()

            grouped_by_country2 = grouped_by_country2.sort_values(by='value', ascending=True)

            st.markdown(f"<h4>Clasificación de países para: {selected_years}</h4>", unsafe_allow_html=True)

            fig4 = px.bar(
                grouped_by_country2,
                x='value',
                y='countryspa',
                orientation='h',  # Horizontal orientation
                text='value',
                title='',
                color_discrete_sequence=['#5978a2']  # Set the color for the bars
                )

            fig4.update_layout(
                xaxis_title='',
                yaxis_title='',
                font=dict(size=14),
                legend=dict(font=dict(size=14)),
                yaxis=dict(tickfont=dict(size=14)),
                xaxis=dict(tickfont=dict(size=14), showticklabels=False),
                bargap=0.1,
                margin=dict(l=0, r=0, t=5, b=40),
                height=400,
                width=230,
                showlegend=False  # Hide the legend
            )

            for bar in fig4.data:
                bar.textposition = 'auto'
                bar.texttemplate = '%{text:.2f}'

            st.plotly_chart(fig4)
with tab2:
    gov=pd.read_excel('SDG National Governance dashboard Database.xlsx',sheet_name="Dataset")
    gov.rename(columns={
   'Country (ESP)': 'country_S',
    'Country (ENG)': 'country_E',
    'Dimensión (ESP)': 'Dim_S',
    'Dimension (ENG)': 'dim_E',
    'Value': 'val',
    'Índice (columna de validación)': 'idx'    
    },inplace=True)
    df_gov=gov[gov.columns]

    selected_tab = st.selectbox("Select Language", ["Governance Index","Índice de Gobernanza"])

    if selected_tab == "Governance Index":

        col1, col2,col3 = st.columns((0.35,0.35,0.3))  

        with col1:
            st.markdown("#### Select Country to view Governance Index")
            countrylist_gov_col1 = df_gov["country_E"].unique().tolist()
            #selected country stores in variable
            selected_gov_country_col1 = st.selectbox("Country", countrylist_gov_col1, index=0)
            # finding avg for selected governance 
            gov_counsel_col1= df_gov[(df_gov["country_E"] == selected_gov_country_col1)]
            govgrouped_df_col1=gov_counsel_col1.groupby(['country_E','dim_E'])['val'].sum()
            govht=govgrouped_df_col1.to_frame()
            htmlgov = govht.to_html(index=True, header=False, justify='center', index_names=False)
            styled_htmlgov = f"<div style='font-family: Roboto, sans-serif; font-size: 16px; text-align: center;'>{htmlgov}</div><br><br>"
            st.write(styled_htmlgov, unsafe_allow_html=True)
            #st.write(govgrouped_df_col1)
            gauge_val=gov_counsel_col1.groupby('country_E')['idx'].max().reset_index()
            gau=gauge_val.iloc[:,lambda gauge_val: [1]]
            gauge_filtered=float(gau._get_value(0,"idx"))
            fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = gauge_filtered,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Governance Index", 'font':{'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 100],'visible': False, 'tickwidth': 1, 'tickcolor': "#fc4811"},
                        'bar': {'color': "#fc4811"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "black",
                        'steps': [
                            {'range': [0, 50], 'color': '#404d86'},
                            {'range': [50, 70], 'color': 'grey'}],
                        }))

            fig.update_layout(width=350,font = {'color': "#24263f", 'family': "Arial"})
            st.write(fig)
        with col2:
            st.markdown("#### Select Country to view Governance Index")
            countrylist_gov_col2 = df_gov["country_E"].unique().tolist()
            #selected country stores in variable
            selected_gov_country_col2= st.selectbox("Country", countrylist_gov_col2, index=2)
            # finding avg for selected governance 
            gov_counsel_col2= df_gov[(df_gov["country_E"] == selected_gov_country_col2)]
            govgrouped_df_col2=gov_counsel_col2.groupby(['country_E','dim_E'])['val'].sum()
            govht=govgrouped_df_col2.to_frame()
            htmlgov = govht.to_html(index=True, header=False, justify='center', index_names=False)
            styled_htmlgov = f"<div style='font-family: Roboto, sans-serif; font-size: 16px; text-align: center;'>{htmlgov}</div><br><br>"
            st.write(styled_htmlgov, unsafe_allow_html=True)
            #st.write(govgrouped_df_col2)
            gauge_val=gov_counsel_col2.groupby('country_E')['idx'].max().reset_index()
            gau=gauge_val.iloc[:,lambda gauge_val: [1]]
            gauge_filtered_col2=float(gau._get_value(0,"idx"))
            fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = gauge_filtered_col2,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Governance Index", 'font':{'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 100],'visible': False, 'tickwidth': 1, 'tickcolor': "#fc4811"},
                        'bar': {'color': "#fc4811"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "black",
                        'steps': [
                            {'range': [0, 50], 'color': '#404d86'},
                            {'range': [50, 70], 'color': 'grey'}], }))

            fig.update_layout(width=350, font = {'color': "#24263f", 'family': "Roboto"})
            st.write(fig)
            
        
        with col3:
            CountryByGI = df_gov.groupby('country_E')['idx'].max().reset_index()
            CountryByGI=CountryByGI.sort_values(by='idx', ascending=True)

            st.markdown(f"<h4>Country Ranking:</h4>", unsafe_allow_html=True)

            fig9 = px.bar(
                CountryByGI,
                x='idx',
                y='country_E',
                orientation='h',  # Horizontal orientation
                text='idx',
                title='',
                color_discrete_sequence=['#5978a2']  # Set the color for the bars
                )

            fig9.update_layout(
                xaxis_title='',
                yaxis_title='',
                font=dict(size=14),
                legend=dict(font=dict(size=14)),
                yaxis=dict(tickfont=dict(size=14)),
                xaxis=dict(tickfont=dict(size=14), showticklabels=False),
                bargap=0.1,
                margin=dict(l=0, r=0, t=5, b=40),
                height=700,
                width=350,
                showlegend=False  # Hide the legend
            )

            for bar in fig9.data:
                bar.textposition = 'auto'
                bar.texttemplate = '%{text:.2f}'

            st.plotly_chart(fig9)


    if selected_tab == "Índice de Gobernanza":
        col4, col5,col6 = st.columns((0.35,0.35,0.3))  

        with col4:
            st.markdown("#### Seleccione el país para ver la calidad de la gobernanza")
            countrylist_gov_col3 = df_gov["country_S"].unique().tolist()
            #selected country stores in variable
            selected_gov_country_col3 = st.selectbox("Country", countrylist_gov_col3, index=0)
            # finding avg for selected governance 
            gov_counsel_col3= df_gov[(df_gov["country_S"] == selected_gov_country_col3)]
            govgrouped_df_col3=gov_counsel_col3.groupby(['country_S','Dim_S'])['val'].sum()
            govht=govgrouped_df_col3.to_frame()
            htmlgov = govht.to_html(index=True, header=False, justify='center', index_names=False)
            styled_htmlgov = f"<div style='font-family: Roboto, sans-serif; font-size: 16px; text-align: center;'>{htmlgov}</div><br><br>"
            st.write(styled_htmlgov, unsafe_allow_html=True)
            #st.write(govgrouped_df_col3)
            gauge_val=gov_counsel_col3.groupby('country_S')['idx'].max().reset_index()
            gau=gauge_val.iloc[:,lambda gauge_val: [1]]
            gauge_filtered_col3=float(gau._get_value(0,"idx"))
            fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = gauge_filtered_col3,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Índice de gobernanza", 'font':{'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 100],'visible': False, 'tickwidth': 1, 'tickcolor': "#fc4811"},
                        'bar': {'color': "#fc4811"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "black",
                        'steps': [
                            {'range': [0, 50], 'color': '#404d86'},
                            {'range': [50, 70], 'color': 'grey'}],
                        }))

            fig.update_layout(width=350,font = {'color': "#24263f", 'family': "Arial"})
            st.write(fig)
        with col5:
            st.markdown("#### Seleccione el país para ver la calidad de la gobernanza")
            countrylist_gov_col4 = df_gov["country_S"].unique().tolist()
            #selected country stores in variable
            selected_gov_country_col4 = st.selectbox("Country", countrylist_gov_col4, index=2)
            # finding avg for selected governance 
            gov_counsel_col4= df_gov[(df_gov["country_S"] == selected_gov_country_col4)]
            govgrouped_df_col4=gov_counsel_col4.groupby(['country_S','Dim_S'])['val'].sum()
            govht=govgrouped_df_col4.to_frame()
            htmlgov = govht.to_html(index=True, header=False, justify='center', index_names=False)
            styled_htmlgov = f"<div style='font-family: Roboto, sans-serif; font-size: 16px; text-align: center;'>{htmlgov}</div><br><br>"
            st.write(styled_htmlgov, unsafe_allow_html=True)
            #st.write(govgrouped_df_col4)
            gauge_val=gov_counsel_col4.groupby('country_S')['idx'].max().reset_index()
            gau=gauge_val.iloc[:,lambda gauge_val: [1]]
            gauge_filtered=float(gau._get_value(0,"idx"))
            fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = gauge_filtered,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Índice de gobernanza", 'font':{'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 100],'visible': False, 'tickwidth': 1, 'tickcolor': "#fc4811"},
                        'bar': {'color': "#fc4811"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "black",
                        'steps': [
                            {'range': [0, 50], 'color': '#404d86'},
                            {'range': [50, 70], 'color': 'grey'}],
                        }))

            fig.update_layout(width=350,font = {'color': "#24263f", 'family': "Arial"})
            st.write(fig)
        
        with col6:
            CountryByGI_S = df_gov.groupby('country_S')['idx'].max().reset_index()
            CountryByGI_S=CountryByGI_S.sort_values(by='idx', ascending=True)

            st.markdown(f"<h4>Clasificación de países:</h4>", unsafe_allow_html=True)

            fig8 = px.bar(
                CountryByGI_S,
                x='idx',
                y='country_S',
                orientation='h',  # Horizontal orientation
                text='idx',
                title='',
                color_discrete_sequence=['#5978a2']  # Set the color for the bars
                )

            fig8.update_layout(
                xaxis_title='',
                yaxis_title='',
                font=dict(size=14),
                legend=dict(font=dict(size=14)),
                yaxis=dict(tickfont=dict(size=14)),
                xaxis=dict(tickfont=dict(size=14), showticklabels=False),
                bargap=0.1,
                margin=dict(l=0, r=0, t=5, b=40),
                height=700,
                width=350,
                showlegend=False  # Hide the legend
            )

            for bar in fig8.data:
                bar.textposition = 'auto'
                bar.texttemplate = '%{text:.2f}'

            st.plotly_chart(fig8)

    
            
            
    
 