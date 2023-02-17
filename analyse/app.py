import streamlit as st
# Data wrangling
import pandas as pd
import numpy as np
import re

# Visualization 
import missingno as msno
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Machine learning 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.title('Case for PFA - findings')

# Add some text to the app
st.write('Welcome to my app!')

# Create a simple plot using Matplotlib
x = np.linspace(0, 10, 100)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)

# Filepaths, tilpas disse til lokal-sti
brancher_fp = r'/Users/albertcortbanke/Case/data/arbejdsmarkedsanalyse_brancher.csv'
koen_alder_fp = r'/Users/albertcortbanke/Case/data/arbejdsmarkedsanalyse_koen_alder.csv'

# Load the data
branche_data = pd.read_csv(brancher_fp)
koen_alder_data = pd.read_csv(koen_alder_fp)

# Show the first few rows of the branche_data
st.write("Branche data:")
st.write(branche_data.head())

# List of columns to drop
columns_to_drop = ['Spm Formulering', 'Field Values Index', 'Farveskala', 'Field Values Index (Fixed)', 'Navigation - Arbejdsmiljøprofiler', 'Gennemsnit', 
                   'Score (Total)', 'Main Group', 'Kategori1', 'Kategori2', 'Kategori3', 'Kategori4', 'Kategori5', 'Kategori6', 'Kategori7', 'Kategori8', 'Kategori9',
                   'Kategori10', 'Kategori11', 'Kategori12', 'Kategori13', 'Kategori14', 'Kategori15', 'Score (Total) (Fixed)', 'Field Values', 'Mean', 'Sluttekst', 
                   'Score (Indekseret score) (gennemsnit)', 'Score (Indekseret score)']

# Drop the specified columns
branche_data = branche_data.drop(columns=columns_to_drop)

# Show the first few rows of the cleaned branche_data
st.write("Cleaned branche data:")
st.write(branche_data.head())

# List of columns to drop for the koen_alder_data
columns_to_drop = ['Score (Indekseret score) (gennemsnit)', 'Score (Indekseret score) (gennemsnit) (label)', 'Farveskala', 'Gennemsnit']

# Drop the specified columns
koen_alder_data = koen_alder_data.drop(columns=columns_to_drop)

# Show the first few rows of the cleaned koen_alder_data
st.write("Cleaned koen_alder data:")
st.write(koen_alder_data.head())