import streamlit as st

# Data wrangling
import pandas as pd
import numpy as np
import re
import requests
import io

# Visualization 

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Machine learning 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.title('Case for PFA - findings')
st.subheader('Albert Cort Banke')

# Add some text to the app
st.markdown("""Velkommen til min HTML-visning. Denne visning fokuserer på findings. Hvis du også vil se data wrangling-delen samt flere steps så gå til min [notebook](https://github.com/albertbanke/Case/blob/main/analyse/analyse.ipynb).
            """)

st.markdown("""
            Visningen består af to blokke, hver med to findings: Eksplorativ dataanalyse og dataanalyse af time-series samt cluster-modellering.
            """)


st.markdown("""
_________
""")

url_branche = "https://raw.githubusercontent.com/albertbanke/Case/main/data/arbejdsmarkedsanalyse_brancher.csv" # Make sure the url is the raw version of the file on GitHub
download_branche = requests.get(url_branche).content

url_koen = "https://raw.githubusercontent.com/albertbanke/Case/main/data/arbejdsmarkedsanalyse_koen_alder.csv" # Make sure the url is the raw version of the file on GitHub
download_koen = requests.get(url_koen).content

# Filepaths, tilpas disse til lokal-sti
brancher_fp = r'/Users/albertcortbanke/Case/data/arbejdsmarkedsanalyse_brancher.csv'
koen_alder_fp = r'/Users/albertcortbanke/Case/data/arbejdsmarkedsanalyse_koen_alder.csv'

file_like_branche = io.StringIO(download_branche.decode('latin-1'))
file_like_koen = io.StringIO(download_koen.decode('latin-1'))


# Load the data
branche_data = pd.read_csv(file_like_branche, delimiter = ";", encoding='latin-1')
koen_alder_data = pd.read_csv(file_like_koen, delimiter = ";", encoding='latin-1')



# List of columns to drop
columns_to_drop = ['Spm Formulering', 'Field Values Index', 'Farveskala', 'Field Values Index (Fixed)', 'Navigation - Arbejdsmiljøprofiler', 'Gennemsnit', 
                   'Score (Total)', 'Main Group', 'Kategori1', 'Kategori2', 'Kategori3', 'Kategori4', 'Kategori5', 'Kategori6', 'Kategori7', 'Kategori8', 'Kategori9',
                   'Kategori10', 'Kategori11', 'Kategori12', 'Kategori13', 'Kategori14', 'Kategori15', 'Score (Total) (Fixed)', 'Field Values', 'Mean', 'Sluttekst', 
                   'Score (Indekseret score) (gennemsnit)', 'Score (Indekseret score)', 'Antpct']


# Drop the specified columns
branche_data = branche_data.drop(columns=columns_to_drop)



# List of columns to drop for the koen_alder_data
columns_to_drop = ['Score (Indekseret score) (gennemsnit)', 'Score (Indekseret score) (gennemsnit) (label)', 'Farveskala', 'Gennemsnit']

# Drop the specified columns
koen_alder_data = koen_alder_data.drop(columns=columns_to_drop)

# Show the first few rows of the cleaned koen_alder_data


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
koen_alder_data['alder'] = koen_alder_data['alder'].fillna('18-64 år')

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

st.markdown('Her er de wranglede datasæt som bruges til at vise findings. Dataet stammer fra det Nationale Center For Arbejdsmiljøs undersøgelse om fysiske og psykiske forhold fra 50.000 adspurgte danskere i 2018')

st.text('(Begge vist med .head())')

st.text("Branche data efter data wrangling")
st.dataframe(branche_data_filtered.head())

st.text("Køn og alder data efter data wrangling")
st.dataframe(koen_alder_data_filtered.head())


st.markdown("""
_________
""")

st.markdown("## Eksplorativ Data Analyse (EDA)\n\nI denne blok analyseres det processerede data eksplorativt med statistik og visualiseringer. Dette afgiver to interresante findings, som præsenteres og analyseres sammen.\n\nEt godt udgangspunkt for at lave stærke dataanalyser er at stille skarpe spørgsmål. Her er to problemformuleringer til de to datasæt - begge med et samfundsmæssigt makro-perspektiv.\n\n- Hvordan scorer arbejdsmiljøet i forskellige brancher i 2018?\n- Hvordan scorer arbejdsmiljøet i forskellige grupper af køn og alder i 2018?\n\nDe to findings er:\n1) Arbejdsmiljøet scorer bedst i brancher med mere selvstændighed og ansvar, som typisk er i det private\n2) Arbejdsmiljøet scorer bedst i grupper med mænd og grupper som er yngre\n\nGå gennem koden nedenfor, for at se findings og deres tilsvarende analyse")


st.subheader("1) Statistik")

st.markdown("#### Branche")

st.markdown("Vi starter med at udforske dataets statistikker, efter vores transformationer i DW.")

st.markdown("Summary statistics, hvor 'Hoej Score Godt' == 1, for branche_data_filtered")
st.write(branche_data_filtered[branche_data_filtered['Hoej Score Godt'] == 1].describe())

st.markdown("Summary statistics, hvor 'Hoej Score Godt' == 0, for branche_data_filtered")
st.write(branche_data_filtered[branche_data_filtered['Hoej Score Godt'] == 0].describe())


st.markdown("""
##### Statistisk analyse af branche_data_filtered

* Den statistiske analyse af `branche_data_filtered` viser et gennemsnitligt indekseret score på 3.72 for spørgsmål, hvor en høj score er god. For spørgsmål, hvor en høj score er dårlig, er tallet 2.83. Dette indikerer, at respondenterne i undersøgelsen generelt har givet højere svar på spørgsmål, hvor 5 er godt, end hvor 5 er dårligt.

* Gennemsnittet af antallet af personer, der har svaret på hvert spørgsmål, er 968 for spørgsmål, hvor en høj score er god, og 967 for spørgsmål, hvor en høj score er dårlig. Dette tal skal dog tages med et gran salt, da der kan være totalrækker, der skaber statistisk inflation. Derfor bør fokus i stedet rettes mod medianen (50%), hvor tallet falder til 277 og 275, henholdsvis.

* Der er en lille diskrepans mellem gennemsnittet for 'Score' og Score Indekseret Gennemsnit, men dette er kun ned til 1. decimal og påvirker ikke det overordnede billede af dataene og vores fremtidige analyse.
""")


st.markdown("#### Køn og alder")

st.markdown("Summary statistics, hvor 'Hoej Score Godt' == 1, for koen_alder_data_filtered")
st.write(koen_alder_data_filtered[koen_alder_data_filtered['Hoej Score Godt'] == 1].describe())

st.markdown("Summary statistics, hvor 'Hoej Score Godt' == 0, for koen_alder_data_filtered")
st.write(koen_alder_data_filtered[koen_alder_data_filtered['Hoej Score Godt'] == 0].describe())

st.markdown("""
##### Statistisk analyse af koen_alder_data_filtered

* Gennemsnitlig indekseret score for spørgsmål med høj score som godt er 3.72, mens det er 2.815 for spørgsmål med høj score som dårligt. Dette antyder et stort overlap blandt spørgsmålene og den samplede befolkning.

* Sammenlignet med branche-dataet er medianen (50%) af antal personer, der har svaret på hvert spørgsmål, højere i koen_alder_data_filtered. For spørgsmål med høj score som godt er medianen 4716, mens den for spørgsmål med høj score som dårligt er 4811. Dette giver god mening, da samplingen i dette datasæt er baseret på køn og aldersgrupper i stedet for branche, hvorfor grupperne er større end eksempelvis niche brancher. 

* Der er flere spørgsmål, hvor høj score betragtes som godt (1), med 348 tilfælde, sammenlignet med 192 tilfælde for spørgsmål, hvor høj score betragtes som dårligt.
""")


st.markdown("""
_________
""")

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
fig.update_layout(title='Boxplot for aldersgrupper og køn ud fra score. Høj score godt = 0')
st.plotly_chart(fig)

st.markdown("""
#### Problemstilling: Hvordan scorer arbejdsmiljøet i forskellige grupper af køn og alder i 2018?

#### **Finding 2: Arbejdsmiljøet scorer bedst i grupper med mænd og grupper som er yngre**

Når man sammenligner de øverste 5 grupper med de nederste 5 grupper i den anden graf med HSG = 0 (desto højere desto dårligere), ser man et klart mønster: Grupper med kvinder scorer i gennemsnit højere. Kvinder i alderen 25-34 har den højeste score (2.95) og alle grupper i de øverste 5 værdier i denne graf er kvinder. Samtidig er kvinders samlede gruppe i de øverste rækker med en score på 2.88, mens mænds samlede gruppe er at finde i de nederste 5 rækker med en score på 2.77. Selvom forskellen mellem kvinders og mænds score for arbejdsmiljø ikke er stor i decimalpunkter, er det vigtigt at fokusere på det overordnede billede. Kvinder er overrepræsenterede, når scoren for arbejdsmiljø er høj, hvilket indikerer et dårligt arbejdsmiljø for kvinder.

Dette bekræftes i boxplottet. Her er alle medianerne for kvinders aldersgrupper, og den samlede (18-64 år) konsekvent over mændenes, hvilket repræsenterer at kvinders arbejdsmiljø (psykisk og fysisk) er værre end mændenes for vores data. Samtidig kan man i boksplottet se at medianen er lavest for begge køn i aldersgruppen 18-24. Dette viser altså at disse aldersgrupper har det bedste arbejdsmiljø, når de rater, hvor HSG = 0.

Det er vigtigt at bemærke, at dataet er et snapshot fra 2018, og det er derfor nødvendigt at sammenligne med andre tidspunkter for at se udviklingen i arbejdsmiljøet i grupper med forskellige køn og alder. Desuden er det værd at bemærke, at dataet ikke tager højde for personer, der ikke identificerer sig som mand eller kvinde, hvilket begrænser generaliserbarheden til den danske befolkning.
""")

st.markdown("""
_________
""")

st.markdown("""
##  Data analysis: Time-Series Analyse og Cluster-Modelling

I denne blok bruger vi indsigterne fra ovenstående EDA til at lave mere dybdegående analyser af vores data. Dette inkluderer en (kort) time-series analyse af informationen omkring offentlige brancher. Bagefter laves der clustering modelling (unsupervised learning). Tilsammen afgiver dette afgiver to interresante findings, som præsenteres og analyseres sammen. 

Igen stilles der to problemformuleringer til de to datasæt. Denne gange zoomes der ind på specifikke brancher og aldersgrupper. 

* Hvordan har arbejdsmiljøet udviklet sig for branchen undervisning (eksempelvis gymnasielærere) ift. tid til arbejdsopgaver fra 2012 til 2018?
* Spørgsmål til clustering

De to findings er

1) Arbejdsmiljøet scorer bedst i brancher med mere selvstændighed og ansvar, som typisk er i det private
2) Arbejdsmiljøet scorer bedst i grupper med mænd og grupper som er yngre
""")

st.markdown("""
### 1) Time-series analyse

Data indlæses fra https://at.dk/arbejdsmiljoe-i-tal/national-overvaagning-af-arbejdsmiljoeet-blandt-loenmodtagere-2021/arbejdsmiljoe-og-helbred-2012-2018/. Bagefter plottes det sammen med data fra branche_data_filtered som er repræsentativt for undervisningsbranchen
""")

my_url = r'https://raw.githubusercontent.com/albertbanke/Case/main/data/Udvikling%20i%20branchen%20undervisning%20(2012%20til%202018).csv'
download_undervisning = requests.get(my_url).content

file_like_undervisning = io.StringIO(download_undervisning.decode('utf-8'))

# Load the data
undervisning_2012_2018 = pd.read_csv(file_like_undervisning, sep = ",", encoding='utf-8')

# Checker at dataet er loaded korrekt
undervisning_2012_2018 # Ser helt fint ud, ikke behov for så tunge transformationer som tidligere data 

# Laver en skal af branche der indeholder undervisere og 'Question Label' er 'Ikke nok tid til arbejdsopgaver'

branche_data_filtered_shell = branche_data_filtered[((branche_data_filtered['Group'] == 'Gymnasielærere') | 
                                                      (branche_data_filtered['Group'] == 'Undervisere og forskere ved universiteter') | 
                                                      (branche_data_filtered['Group'] == 'Undervisere ved erhvervsskoler')) & 
                                                     (branche_data_filtered['Question Label'].str.contains('Ikke nok tid til arbejdsopgaver'))]

branche_data_filtered_shell

# Opret et line plot med data fra undervisning_2012_2018
# x-akse er "År" og y-akse er "Score eller andel", som er navngivet som "Score"
fig = px.line(undervisning_2012_2018, x='År', y='Score eller andel', title='Udvikling i tid til arbejdsopgaver for branchen "Undervisning" fra 2012-2018 (HSG = 0)', labels={'Score eller andel': 'Score'})

# Tilføj et scatter plot for hver unik gruppe i branche_data_filtered_shell
for group in branche_data_filtered_shell['Group']:
    # Filtrer branche_data_filtered_shell for en given gruppe
    df = branche_data_filtered_shell[branche_data_filtered_shell['Group'] == group]
    # Tilføj scatter plot til fig med data fra df
    fig.add_scatter(x=df['Year'], y=df['Score'], mode='markers', name=group)

# Tilføj en lodret linje til fig på år 2014
fig.add_shape(
    type='line',
    x0=2014,
    x1=2014,
    y0=0,
    y1=1,
    yref='paper',
    line=dict(color='darkgray', width=2)
)

# Tilføj en annotation til fig, der angiver "Folkeskolereform" på linjen
fig.add_annotation(
    x=2014,
    y=1,
    yref='paper',
    yanchor='bottom',
    text='Folkeskolereform',
    font=dict(color='darkgray', size=14),
    showarrow=False
)

# Vis fig
st.plotly_chart(fig)

st.markdown("""
#### Problemstilling: Hvordan har arbejdsmiljøet udviklet sig for branchen undervisning (eksempelvis gymnasielærere) ift. tid til arbejdsopgaver fra 2012 til 2018?

#### **Finding 3: Tid til arbejdsopgaver for branchen undervisning er blevet mindre fra 2012 til 2018.**

Ved at sammenligne punkterne på grafen og iagttage udviklingen for branchen undervisning, er det tydeligt, at tiden til arbejdsopgaver er blevet mindre. I 2012 ratede branche scoren til 2.6 (mindre er bedre), mens tallet steg til 2.9 i 2018. Selvom den nominelle stigning kan synes lille, repræsenterer dette en stigning på 11,5%. Det betyder altså, at der er mindre tid til de opgaver, som undervisere, såsom gymnasielærere, skal løse. Dette understøtter Finding 1 godt, som viste, at gymnasielærere (og andre brancher inden for det offentlige) lå lavest i forhold til rating af deres arbejdsmiljø.

Her er der visualiseret data for et specifikt emne i Arbejdstilsynets rapport - 'Kvantitative krav og grænseløshed'. Ved at gå fra et makrobillede til et mikrobillede af dataet er vi i stand til at lave nogle induktive overvejelser. Bl.a. kan man undersøge reformer som 'Folkeskolereformen' fra 2014, og om disse har haft implikationer på branchens tilstand. Plotter man denne reform på dataet og ser ændringer efter reformen blev lanceret kan det ikke afkræftes at det kan have haft en effekt. 
""")

st.markdown("""
_________
""")

st.markdown("""
### 2) Clustering-analyse

Til den sidste af vores fund, vil vi udføre en K-means clustering-analyse. K-means clustering-analysen anvender unsupervised learning, hvorfor vi ikke vil give nogen labels. I stedet beder vi K-means om at identificere et antal clusters i vores data baseret på tre kolonner: alder, køn og score. For at gøre de to første kolonner numeriske og anvendelige i analysen, vil vi benytte hot-encoding og map-encoding. Samtidig standard-scaler vi dataet (så det er nemmere for K-means algoritmen at mappe det grundet mindre dimensionalitet).
""")

# Først laver vi one-hot encoding af køn og alder variablerne. Step 1 er at fjerne 'total' rækkerne for kvinder og mænd
koen_alder_no_total = koen_alder_data_filtered[~koen_alder_data_filtered['Group'].isin(['Kvinder', 'Mænd'])]

# Step 2 er at one-hot encode køn-kolonnen vi lavede tidligere
koen_alder_no_total['køn_n'] = np.where(koen_alder_no_total['køn'] == 'Mænd', 0, 1)

# Step 3 er at map encode alder-kolonnen. Først en alders range

age_range_mapping = {
    '18 - 24 år': 0,
    '25 - 34 år': 1,
    '35 - 44 år': 2,
    '45 - 54 år': 3,
    '55 - 64 år': 4
}

# Brug mappingen til at give værdier til kolonnen
koen_alder_no_total['alder_n'] = koen_alder_no_total['alder'].map(age_range_mapping)

# Fjern Score Indekseret, da vi fokusere på rangen 1-5 som tidligere
koen_alder_no_total.drop(columns='Score (Indekseret score)')

# Tjek resultatet
koen_alder_no_total # Ser super ud

# Udtræk de funktioner fra koen_alder_no_total som skal bruges til clustering
features = koen_alder_no_total[['køn_n', 'alder_n', 'Score']]

# Skaler funktionerne
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Definer en range for antallet af clusters
cluster_range = range(1, 15)

# Initialiser en liste til at gemme Within-Cluster-Sum-of-Squared-Errors (WCSS) for hvert antal klynger
wcss = []

# Loop over antallet af clusters
for num_clusters in cluster_range:
    # Fit KMeans-modellen
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(scaled_features)
    
    # Beregn WCSS
    wcss.append(kmeans.inertia_)

# Plot WCSS
fig = px.line(x=cluster_range, y=wcss)
fig.update_layout(
    title="Albue-metode til optimal antal clusters",
    xaxis_title="Antal clusters",
    yaxis_title="Within-Cluster-Sum-of-Squared-Errors (WCSS)",
)
st.plotly_chart(fig)


st.markdown("""
Grunden til at WCSS foretrækkes over SSD (Sum of squared distances) til at finde det optimale antal clusters i K-Means clustering, er at det er direkte relateret til målet med clusteralgoritmen, nemlig at minimere summen af kvadrerede afstande mellem hver datapunkt og samtidig undgå at overfitte clusters i datasættet, som SSD godt kan være tilbøjelig til.
""")

# Sætter et random seed for reproducibility
np.random.seed(12)

# Lav K-means clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(scaled_features)

# Tilføj cluster labels til data framen
koen_alder_no_total['Cluster'] = kmeans.labels_

# Udtræk cluster centers
cluster_centers = kmeans.cluster_centers_

# Udtræk koordinaterne af de forskellige clusters
x = scaled_features[:, 0]
y = scaled_features[:, 1]
z = scaled_features[:, 2]

# Create the 3D scatter plot with the KMeans labels
fig = px.scatter_3d(x=x, y=y, z=z, color=kmeans.labels_, color_discrete_sequence=px.colors.sequential.Viridis)

# Plot cluster centers
fig.add_scatter3d(x=cluster_centers[:, 0], y=cluster_centers[:, 1], z=cluster_centers[:, 2], mode='markers', marker=dict(size=10, color='red', symbol='circle'))

# Update the plot layout
fig.update_layout(scene=dict(xaxis_title='Køn (skaleret)', yaxis_title='Alder (skaleret)', zaxis_title='Score (skaleret)'))

# Show the plot in Streamlit
st.plotly_chart(fig)

st.markdown("""
#### Problemstilling: Hvor mange clusters er der i køn og alder dataet, og hvordan kan machine learning anvendes ved sensitive variabler som disse?

#### **Finding 4: K-means clustering og elbow-metoden viser at det optimale antal clusters er 4 for køn og alder dataet. Machine learning skal anvendes med stor forsigtighed ved sensitive variabler som disse**

Grafen viser, at det optimale antal clusters er 4, som afsløres af de fire røde centroider, der er markeret i den 3D-graf. Dette er interessant, da man kunne have argumenteret for, at der skulle være 10 clusters (2 x køn og 5 x aldersgrupper = 10). En visuel analyse af k-means clusteringen viser noget meget fascinerende. En af clusterne, den markeret med den gule farve, indeholder data fra begge køn. Alle de andre clusters indeholder kun data fra ét køn. Derudover omfatter den gule og lilla cluster data fra alle aldersgrupper. En dybere analyse af dette kunne undersøge, om den gule cluster måske repræsenterer en branche eller en anden variabel, der ikke er medtaget i denne clustering.

Denne analyse er en unsupervised learning-øvelse, hvilket betyder, at vi ikke beder modellen om at fitte noget data ud fra et label. Derudover er denne clustering ikke en regression, hvilket betyder, at den ikke scorer noget - den udforsker blot dataet og grupperingerne i det. Dette er vigtigt at have i mente, da de variabler, vi anvender (køn, alder), er ekstremt sensitive. Hvis øvelsen f.eks. var at modellere en model til at godkende eller afvise banklån, ville det være uetisk at anvende disse variabler som predictors for tilbagebetalingsevne. 

Clustering modellen her er forholdsvis 'white-box', hvilket betyder, at vi (relativt) nemt kan forklare, hvordan den beregner disse clusters. Hvis det derimod var en 'black-box'-model, ville man igen skulle være forsigtig med at anvende sensitive variabler som disse, da vi ikke ville have en god mulighed for at forklare, hvordan modellen opnår de resultater og predictions, den gør.
""")


st.markdown("""
_________
""")


st.markdown("""
# Konklusion

Denne notebook har fokuseret på rapporten fra Arbejdstilsynet om danskernes arbejdsmiljø i 2018. Ved hjælp af data wrangling blev datasættene gjort klar til analyse, og EDA viste nogle interessante resultater. 

1. Arbejdsmiljøet scorer bedst i brancher med større selvstændighed og ansvar, typisk i det private.
2. Arbejdsmiljøet scorer bedre for grupper med mænd og yngre personer.

Time-series analyse og clustering modellering gav yderligere indsigt i dataet:

3. Tiden til arbejdsopgaver for branchen undervisning er blevet mindre fra 2012 til 2018.
4. K-means clustering og elbow-metoden viste, at det optimale antal klynger er 4 for køn og alder dataet. Machine learning skal anvendes med stor forsigtighed ved sensitive variabler som disse.

Dataet er historisk og reflektering af udviklingen i arbejdsmiljøet på baggrund af findings er på sin plads. For eksempel kan strejkerne i 2021 blandt sygeplejersker relateres til den lavere vurdering af deres arbejdsmiljø, som blev vist i resultat 1. Kigger man på time-series analysen er en implikation de samfundsændringer som kan påvirke elementer som tid til arbejdsopgaver. Lovgivninger og reformer (såsom folkeskole- og gymnasiereformerne) skal holdes op sammen med dataudviklingen for at hjælpe med at forklare, hvorfor ændringerne ser ud som de gør. Til sidst skal clustering og machine learning anvendes med forsigtighed for sensitive variabler som køn og alder, da det kan føre til indsat bias i modeller, der påvirker menneskers liv.

""")


st.markdown("""
_________
""")


