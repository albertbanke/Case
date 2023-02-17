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



# Filepaths, tilpas disse til lokal-sti
brancher_fp = r'/Users/albertcortbanke/Case/data/arbejdsmarkedsanalyse_brancher.csv'
koen_alder_fp = r'/Users/albertcortbanke/Case/data/arbejdsmarkedsanalyse_koen_alder.csv'

# Load the data
branche_data = pd.read_csv(brancher_fp, delimiter = ";", encoding='latin-1')
koen_alder_data = pd.read_csv(koen_alder_fp, delimiter = ";", encoding='latin-1')



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



branche_data = branche_data.dropna(subset=['Type'])

koen_alder_data = koen_alder_data.dropna(subset=['Ordforklaring'])

# Definer et regex mønster, der matcher 'Spørgsmål:' og efterfølgende kun bevarer single mellemrum og ikke-special-karakterer
regex_pattern = r'Spørgsmål:\s+|\s{2,}|\W+'

# Anvend regex mønsteret på 'Ordforklaring' kolonnen i begge dataframes
branche_data['Ordforklaring'] = branche_data['Ordforklaring'].apply(lambda x: re.sub(regex_pattern, ' ', str(x)).strip())
koen_alder_data['Ordforklaring'] = koen_alder_data['Ordforklaring'].apply(lambda x: re.sub(regex_pattern, ' ', str(x)).strip())



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


st.markdown("## 3. Eksplorativ Data Analyse (EDA)\n\nI denne blok analyseres det processerede data eksplorativt med statistik og visualiseringer. Dette afgiver to interresante findings, som præsenteres og analyseres sammen.\n\nEt godt udgangspunkt for at lave stærke dataanalyser er at stille skarpe spørgsmål. Her er to problemformuleringer til de to datasæt - begge med et samfundsmæssigt makro-perspektiv.\n\n- Hvordan scorer arbejdsmiljøet i forskellige brancher i 2018?\n- Hvordan scorer arbejdsmiljøet i forskellige grupper af køn og alder i 2018?\n\nDe to findings er:\n1) Arbejdsmiljøet scorer bedst i brancher med mere selvstændighed og ansvar, som typisk er i det private\n2) Arbejdsmiljøet scorer bedst i grupper med mænd og grupper som er yngre\n\nGå gennem koden nedenfor, for at se findings og deres tilsvarende analyse")


st.subheader("1) Statistik")

st.markdown("#### Branche")

st.markdown("Vi starter med at udforske dataets statistikker, efter vores transformationer i DW.")

st.markdown("Summary statistics, hvor 'Hoej Score Godt' == 1, for branche_data_filtered")
st.write(branche_data_filtered[branche_data_filtered['Hoej Score Godt'] == 1].describe())

st.markdown("Summary statistics, hvor 'Hoej Score Godt' == 0, for branche_data_filtered")
st.write(branche_data_filtered[branche_data_filtered['Hoej Score Godt'] == 0].describe())

st.markdown("Den statistiske analyse af branche_data_filtered viser den gennemsnitlige indekserede score for spørgsmål, hvor høj score er godt, til 3.72. For data, hvor høj score er dårligt, er dette tal 2.83. Dette viser, at respondenterne i undersøgelsen generelt har svaret højere ved spørgsmål, hvor 5 er godt, end hvor 5 er dårligt.")
st.markdown("Gennemsnittet af antal personer, der har svaret på hvert spørgsmål, er 968, hvor høj score er godt, mens det er 967, hvor høj score er dårligt. Dette tal skal dog holdes op i lyset af at der er total rækker, der kan skabe statistisk inflation i dette. **Fokus skal derfor rettes på medianen (50%), hvor tallet falder til henholdsvis 277 og 275 respektivt.**")
st.markdown("Der er en lille diskrepans på gennemsnittet for 'Score' og Score Indekseret Gennemsnit. Dette er dog ned til 1. decimal og ændrer derfor ikke i det store billede for dataet og vores fremtidige analyse.")

st.markdown("#### Køn og alder")

st.markdown("Summary statistics, hvor 'Hoej Score Godt' == 1, for koen_alder_data_filtered")
st.write(koen_alder_data_filtered[koen_alder_data_filtered['Hoej Score Godt'] == 1].describe())

st.markdown("Summary statistics, hvor 'Hoej Score Godt' == 0, for koen_alder_data_filtered")
st.write(koen_alder_data_filtered[koen_alder_data_filtered['Hoej Score Godt'] == 0].describe())

st.markdown("Den statistiske analyse af koen_alder_data_filtered viser den gennemsnitlige indekserede score for spørgsmål, hvor høj score er godt, til 3.72. For data, hvor høj score er dårligt, er dette tal 2.815. Dette peger på stort overlap blandt spørgsmålene, samtidig også på den samplede population.")
st.markdown("En stor forskel i forhold til branche-dataet er gennemsnittet af antal personer, der har svaret på hvert spørgsmål. **Sættes fokus på medianen (50%) som beskrevet i analysen for branche-dataet, er tallene 4716 og 4811 respektivt.** Altså markant højere end før. Dette giver god mening, da sampling her ikke er på branche, men derimod på køn og grupperet alder.")
st.markdown("Der er flere spørgsmål, hvor 'Hoej Score Godt' er == 1. Tallene er 348 for godt og 192 for ikke godt.")

st.subheader("2) Visualiseringer")


def plot_top_and_bottom_groups(dataframe, hsg_value, kilde):
    # Filtrer dataframen til kun at inkludere rækker, hvor 'Hoej Score Godt' er lig med hsg_værdi
    filtreret_hoej_score_data = dataframe[dataframe['Hoej Score Godt'] == hsg_value]

    # Gruppér data efter "Group" kolonnen og beregn hver gruppes gennemsnitsscore for alle spørgsmål
    grupperet_data = filtreret_hoej_score_data.groupby(['Group'], as_index=False).mean()

    # Sortér data efter gennemsnitsscore i faldende orden
    grupperet_data.sort_values(by='Score', ascending=True, inplace=True)

    # Filtrer de øverste 5 og de nederste 5 grupper
    top_5_grupper = grupperet_data.head(5)
    bottom_5_grupper = grupperet_data.tail(5)

    # Tilføj en kolonne for at adskille de øverste og nederste 5 grupper
    top_5_grupper.loc[:, 'Group Type'] = 'Nederste 5 Grupper'
    bottom_5_grupper.loc[:, 'Group Type'] = 'Øverste 5 Grupper'

    # Kobl de to dataframes
    kombineret_data = pd.concat([top_5_grupper, bottom_5_grupper])

    # Opret et søjlediagram med Plotlys px
    fig = px.bar(kombineret_data, x='Score', y='Group', color='Group Type', color_discrete_sequence=['lightsalmon', '#1f77b4'], orientation='h')
    fig.update_layout(title=f'Øverste og nederste 5 grupper efter samlet score for {kilde}. Høj score godt = {hsg_value}', xaxis_title='Score', yaxis_title='Group')
    st.plotly_chart(fig)

st.markdown("#### Branche data")

plot_top_and_bottom_groups(branche_data_filtered, 1, 'branche data')

plot_top_and_bottom_groups(branche_data_filtered, 0, 'branche data')

st.markdown("#### Problemstilling: Hvordan scorer arbejdsmiljøet i forskellige brancher i 2018?\n\n#### Finding 1: Arbejdsmiljøet scorer bedst i brancher med mere selvstændighed og ansvar, som typisk er i det private.\n\nNår man sammenligner de øverste 5 grupper med de nederste 5 grupper i den første graf, er to ting tydelige. Ledere scorer højest i HSG = 1 med en gennemsnitlig score på 3.93, mens passagerservicemedarbejdere scorer lavest med en gennemsnitlig score på 3.41.\n\nDe øverste 5 grupper i denne graf med HSG = 1 består typisk af brancher og arbejdstyper, der har mere selvstændighed og ansvar, såsom frisører, kosmetologer og ledere. Disse brancher er mere autonome end f.eks. bud og kurer, der findes i de nederste 5 grupper. For eksempel har frisører mere frihed og ejer ofte deres egen butik, mens bud typisk er ansat i en virksomhed som DHL.\n\nNår man sammenligner de øverste 5 grupper med de nederste 5 grupper i den anden graf med HSG = 0, ser man et klart mønster: Folk ansat i offentlige stillinger scorer i gennemsnit højere. Socialrådgivere (3.10), gymnasielærere (3.07) og SOSU-assistenter (2.99) er typisk ansat i offentlige stillinger (med undtagelse af særlige stillinger som vikariater). De nederste 5 grupper indeholder typisk brancher, der er mere privat end offentlig i ansættelsesnatur, såsom frisører og kosmetologer (2.65) og f.eks. også tømrere og snedkere (2.66).\n\nDet er vigtigt at bemærke, at dataet er et snapshot fra 2018, og det er derfor nødvendigt at sammenligne med andre tidspunkter for at se udviklingen i de forskellige branchers arbejdsmiljø.")

st.markdown("#### Køn og alder data")

plot_top_and_bottom_groups(koen_alder_data_filtered, 0, 'køn og alder data')

df = koen_alder_data_filtered[koen_alder_data_filtered['Hoej Score Godt'] == 0]
fig = px.box(df, x='alder', y='Score', color='køn')
st.plotly_chart(fig)

st.markdown("""
#### Problemstilling: Hvordan scorer arbejdsmiljøet i forskellige grupper af køn og alder i 2018?

#### **Finding 2: Arbejdsmiljøet scorer bedst i grupper med mænd og grupper som er yngre**

Når man sammenligner de øverste 5 grupper med de nederste 5 grupper i den første graf med HSG = 1, er det tydeligt, at mænd i alderen 18-24 scorer højest med en gennemsnitlig score på 3.77, mens kvinder i alderen 25-34 scorer lavest med en gennemsnitlig score på 3.69. De øverste 5 grupper i denne graf består af et flertal af grupper med mænd (3 ud af 5) og de to øverste grupper er de yngste i sættet (18-24 år og 25-34 år).

Når man sammenligner de øverste 5 grupper med de nederste 5 grupper i den anden graf med HSG = 0, ser man et klart mønster: Grupper med kvinder scorer i gennemsnit højere. Kvinder i alderen 25-34 har den højeste score (2.95) og alle grupper i de øverste 5 værdier i denne graf er kvinder. Samtidig er kvinders samlede gruppe i de øverste rækker med en score på 2.88, mens mænds samlede gruppe er at finde i de nederste 5 rækker med en score på 2.77. Selvom forskellen mellem kvinders og mænds score for arbejdsmiljø ikke er stor i decimalpunkter, er det vigtigt at fokusere på det overordnede billede. Kvinder er overrepræsenterede, når scoren for arbejdsmiljø er høj, hvilket indikerer et dårligt arbejdsmiljø for kvinder.

Det er vigtigt at bemærke, at dataet er et snapshot fra 2018, og det er derfor nødvendigt at sammenligne med andre tidspunkter for at se udviklingen i arbejdsmiljøet i grupper med forskellige køn og alder. Desuden er det værd at bemærke, at dataet ikke tager højde for personer, der ikke identificerer sig som mand eller kvinde, hvilket begrænser generaliserbarheden til den danske befolkning.
""")

