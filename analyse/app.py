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
branche_data = pd.read_csv(brancher_fp, delimiter = ";", encoding='latin-1')
koen_alder_data = pd.read_csv(koen_alder_fp, delimiter = ";", encoding='latin-1')

# Show the first few rows of the branche_data
st.write("Branche data:")
st.write(branche_data.head())

# List of columns to drop
columns_to_drop = ['Spm Formulering', 'Field Values Index', 'Farveskala', 'Field Values Index (Fixed)', 'Navigation - Arbejdsmiljøprofiler', 'Gennemsnit', 
                   'Score (Total)', 'Main Group', 'Kategori1', 'Kategori2', 'Kategori3', 'Kategori4', 'Kategori5', 'Kategori6', 'Kategori7', 'Kategori8', 'Kategori9',
                   'Kategori10', 'Kategori11', 'Kategori12', 'Kategori13', 'Kategori14', 'Kategori15', 'Score (Total) (Fixed)', 'Field Values', 'Mean', 'Sluttekst', 
                   'Score (Indekseret score) (gennemsnit)', 'Score (Indekseret score)', 'Antpct']


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
st.write("Cleaned koen_alder data HELLO:")
st.write(koen_alder_data.head())

def convert_columns_to_numeric(df):
    for column in df.columns:
        if df[column].dtype == 'object' and re.match(r'^[0-9,]+$', df[column].str.strip().str.cat()):
            try:
                df[column] = df[column].str.replace(',', '.')
                df[column] = pd.to_numeric(df[column], errors='coerce')
            except ValueError:
                pass
    return df

branche_data = convert_columns_to_numeric(branche_data)
koen_alder_data = convert_columns_to_numeric(koen_alder_data)

st.write("value counts for branche_data after transformation:")
st.write(branche_data.dtypes.value_counts())
st.write("\nvalue counts for koen_alder_data after transformation:")
st.write(koen_alder_data.dtypes.value_counts())

branche_data = branche_data.dropna(subset=['Type'])

koen_alder_data = koen_alder_data.dropna(subset=['Ordforklaring'])

# Definer et regex mønster, der matcher 'Spørgsmål:' og efterfølgende kun bevarer single mellemrum og ikke-special-karakterer
regex_pattern = r'Spørgsmål:\s+|\s{2,}|\W+'

# Anvend regex mønsteret på 'Ordforklaring' kolonnen i begge dataframes
branche_data['Ordforklaring'] = branche_data['Ordforklaring'].apply(lambda x: re.sub(regex_pattern, ' ', str(x)).strip())
koen_alder_data['Ordforklaring'] = koen_alder_data['Ordforklaring'].apply(lambda x: re.sub(regex_pattern, ' ', str(x)).strip())

st.write("Checking one of the rows:")
st.write(branche_data.Ordforklaring[1])

# Bruger igen regex mønster til at fange informationen vi ønsker 
koen_alder_data[['køn', 'alder']] = koen_alder_data['Group'].str.extract(r'(\w+),?\s*(\d+\s*-\s*\d+\s*år)?')

# Fillna på alder, der hvor det er totaler
koen_alder_data['alder'] = koen_alder_data['alder'].fillna('18-100 år')

# Filtrer dataet
branche_data_filtered = branche_data[branche_data['Ordforklaring'].str.contains('I resultaterne præsenteres den gennemsnitlige score 1 5')]
koen_alder_data_filtered = koen_alder_data[koen_alder_data['Ordforklaring'].str.contains('I resultaterne præsenteres den gennemsnitlige score 1 5')]

# Skaber kolonenn 'Hoej Score Godt' i data framen koen_alder_data_filtreret ud fra tilsvarende værdier for 'Ordforklaring' i branche_data_filtered

# Merge data frames
merged_df = koen_alder_data_filtered.merge(branche_data_filtered, on='Ordforklaring', how='left')

# Drop duplikater og reset index
merged_df.drop_duplicates(subset='Ordforklaring', inplace=True)
merged_df.set_index('Ordforklaring', inplace=True)

# Skab kolonnen i den originale data frame ud fra merged_df's værdier (inhereted fra )
koen_alder_data_filtered['Hoej Score Godt'] = koen_alder_data_filtered['Ordforklaring'].map(merged_df['Hoej Score Godt'])


st.markdown("## 3. Eksplorativ Data Analyse (EDA)")
st.markdown("I denne blok analyseres det processerede data eksplorativt med statistik og visualiseringer. Dette afgiver to interresante findings, som præsenteres og analyseres sammen.")
st.markdown("Et godt udgangspunkt for at lave stærke dataanalyser er at stille skarpe spørgsmål. Her er to problemformuleringer til de to datasæt - begge med et samfundsmæssigt makro-perspektiv.")
st.markdown("- Hvordan scorer arbejdsmiljøet i forskellige brancher i 2018?")
st.markdown("- Hvordan scorer arbejdsmiljøet i forskellige grupper af køn og alder i 2018?")
st.markdown("De to findings er:")
st.markdown("1) Arbejdsmiljøet scorer bedst i brancher med mere selvstændighed og ansvar, som typisk er i det private")
st.markdown("2) Arbejdsmiljøet scorer bedst i grupper med mænd og grupper som er yngre")
st.markdown("Gå gennem koden nedenfor, for at se findings og deres tilsvarende analyse")

st.markdown("Vi starter med at udforske dataets statistikker, efter vores transformationer i DW.")
st.markdown("Summary statistics, hvor 'Hoej Score Godt' == 1, for branche_data_filtered")

st.write(branche_data_filtered[branche_data_filtered['Hoej Score Godt'] == 1].describe())


