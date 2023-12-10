import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
from scipy.stats import zscore, norm

# Function to scrape Wikipedia table and calculate statistical measures
def scrape_wikipedia_table(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        tables = soup.find_all("table", {"class": "wikitable"})
        dataframes_list = []

        for i, table in enumerate(tables):
            df = pd.read_html(str(table), header=0)[0]

            # Flatten the DataFrame if it contains nested columns
            df.columns = df.columns.map(' '.join)

            # Convert any columns containing strings to NaN
            df = df.apply(pd.to_numeric, errors='coerce')

            # Drop rows where all values are NaN
            df = df.dropna(how='all')

            dataframes_list.append(df)

        return dataframes_list

    else:
        st.error(f"Failed to retrieve data from the Wikipedia link. Status code: {response.status_code}")
        return []

# # Function to calculate statistical measures for each column
# def calculate_statistics(df):
#     st.subheader("Mesure Statistique pour Chaque Colonne")

#     for column in df.columns:
#         st.write(f"### Colonne: {column}")

#         # Print the content of the column for debugging
#         print(f"DataFrame content for column {column}:\n", df[column])

#         st.write(df[column].describe().round(2))
        


def perform_z_test(data, column1, column2):
    try:
        z_stat, p_value = zscore(data[column1]), zscore(data[column2])
        z_test_result = norm.sf(abs(z_stat)) * 2  # two-tailed test
        return z_stat, p_value, z_test_result
    except Exception as e:
        st.error(f"Error performing Z test: {str(e)}")
        return None, None, None

def display_z_test_results(z_stat, p_value, z_test_result, significance_level=0.05):
    st.subheader("Z Test Results")

    st.write(f"P-Value Summary: Min = {p_value.min():.10f}, Max = {p_value.max():.10f}")
    st.write(f"Z Statistic Summary: Min = {z_stat.min():.10f}, Max = {z_stat.max():.10f}")

    if p_value.min() < significance_level:
        st.success("Result: There is a significant difference between the means of the selected columns.")
    else:
        st.warning("Result: There is no significant difference between the means of the selected columns.")

    st.write("Individual Results:")
    for i, (p_val, z_stat_val) in enumerate(zip(p_value, z_stat)):
        st.write(f"Pair {i + 1}: P-Value = {p_val:.10f}, Z Statistic = {z_stat_val:.10f}")

# Interface Streamlit
def main():

    st.markdown('<link rel="stylesheet" type="text/css" href="styles.css">', unsafe_allow_html=True)


    st.title('Projet d\'analyse de données')

    # Choix de la source de données (Téléchargement ou Lien)
    data_source = st.radio("Source de données :", ('Téléchargement', 'Lien'))

    if data_source == 'Téléchargement':
        # Sélection du type de fichier (CSV, XLSX, ou TXT)
        file_type = st.selectbox("Sélectionnez le type de fichier :", ['CSV', 'XLSX', 'TXT'])

        # Téléchargement du fichier
        uploaded_file = st.file_uploader(f"Téléchargez un fichier {file_type}", type=[file_type.lower()])

        # Charger les données du fichier si disponible
        if uploaded_file is not None and uploaded_file:
            try:
                if file_type == 'CSV':
                    data = pd.read_csv(uploaded_file)
                elif file_type == 'XLSX':
                    data = pd.read_excel(uploaded_file, engine='openpyxl')
                elif file_type == 'TXT':
                    data = pd.read_csv(uploaded_file, sep='\t')  # Exemple avec séparateur tabulation, ajustez en fonction de votre format

                process_data(data)

            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")


    elif data_source == 'Lien':
        # Saisie du lien vers les données
        data_url = st.text_input("Entrez le lien vers les données (CSV) :", "")

        if data_url:
            try:
                # Use the Wikipedia scraping function
                wikipedia_tables = scrape_wikipedia_table(data_url)

                # Process the scraped tables if available
                if wikipedia_tables:
                    for i, wiki_df in enumerate(wikipedia_tables):
                        st.subheader(f"Table Wikipedia {i + 1}")
                        st.write(wiki_df)
                    
                     # Display statistical measures under each table
                        st.subheader("Mesure Statistique")
                        selected_statistic = st.radio(f"Sélectionnez la mesure statistique (Table {i + 1}):", ('mean', 'median', 'std', 'min', 'max'), key=f"statistic_{i}")

                        for col in wiki_df.columns:
                            st.write(f"### Column: {col}")
                            if selected_statistic == 'mean':
                                st.write("Mean:", wiki_df[col].mean())
                            elif selected_statistic == 'median':
                                st.write("Median:", wiki_df[col].median())
                            elif selected_statistic == 'std':
                                st.write("Standard Deviation:", wiki_df[col].std())
                            elif selected_statistic == 'min':
                                st.write("Minimum:", wiki_df[col].min())
                            elif selected_statistic == 'max':
                                st.write("Maximum:", wiki_df[col].max())
                            st.write("\n" + "=" * 80 + "\n")

                        # Display visualization options
                        st.subheader("Visualisation")
                        selected_chart_type = st.radio(f"Sélectionnez le type de graphique (Table {i + 1}):", ['bar', 'hist', 'heatmap', 'line', 'scatter'])

                        if selected_chart_type in ['bar', 'hist', 'line']:
                            selected_x_column = st.selectbox(f"Saisissez la colonne de l'axe X (Table {i + 1}):", wiki_df.columns)
                            selected_y_column = st.selectbox(f"Saisissez la colonne de l'axe Y (Table {i + 1}):", wiki_df.columns)

                        elif selected_chart_type == 'heatmap':
                            selected_x_columns = st.multiselect(f"Sélectionnez les colonnes pour l'axe X (Table {i + 1}):", wiki_df.columns, key=f"selected_x_columns_{i}")

                        # Display the graph based on user's choice
                        button_key = f"button_{i}"
                        if st.button("Afficher le graphique", key=button_key):
                            try:
                                if selected_chart_type == 'bar':
                                    fig, ax = plt.subplots()
                                    wiki_df.plot.bar(x=selected_x_column, y=selected_y_column, ax=ax)
                                    ax.set_xlabel(selected_x_column)
                                    ax.set_ylabel(selected_y_column)
                                    st.pyplot(fig)
                                elif selected_chart_type == 'hist':
                                    fig, ax = plt.subplots()
                                    wiki_df[selected_x_column].plot.hist(ax=ax)
                                    ax.set_xlabel(selected_x_column)
                                    ax.set_ylabel('Count')
                                    st.pyplot(fig)
                                elif selected_chart_type == 'line':
                                    fig, ax = plt.subplots()
                                    wiki_df.plot.line(x=selected_x_column, y=selected_y_column, ax=ax)
                                    ax.set_xlabel(selected_x_column)
                                    ax.set_ylabel(selected_y_column)
                                    st.pyplot(fig)
                                elif selected_chart_type == 'heatmap':
                                    fig, ax = plt.subplots()
                                    correlation_matrix = wiki_df[selected_x_columns].corr()
                                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
                                    st.pyplot(fig)
                                elif selected_chart_type == 'scatter':
                                    fig, ax = plt.subplots()
                                    wiki_df.plot.scatter(x=selected_x_column, y=selected_y_column, ax=ax)
                                    ax.set_xlabel(selected_x_column)
                                    ax.set_ylabel(selected_y_column)
                                    st.pyplot(fig)

                            except Exception as e:
                                st.error(f"Erreur lors de l'affichage du graphique : {str(e)}")

                                if "categorical data" in str(e):
                                    st.write("Erreur : Vous ne pouvez pas créer un graphique de ligne avec des données catégoriques.")
                    # Process the Wikipedia tables further or integrate with existing code

                else:
                    st.warning("Aucune table Wikipedia trouvée dans le lien spécifié.")

            except Exception as e:
                st.error(f"Erreur lors de la lecture des données depuis le lien : {e}")

def process_data(data):
    # Sélection des colonnes et lignes
    selected_columns = st.multiselect("Sélectionnez les colonnes :", data.columns)
    selected_rows = st.multiselect("Sélectionnez les lignes :", data.index)

    # Vérifier si les listes de colonnes et de lignes sont vides
    if not selected_columns:
        selected_columns = data.columns  # Afficher toutes les colonnes si aucune colonne n'est sélectionnée
    if not selected_rows:
        selected_rows = data.index  # Afficher toutes les lignes si aucune ligne n'est sélectionnée

    # Affichage du tableau résultant
    result_data = data.loc[selected_rows, selected_columns]
    st.write("Résultat du tableau :")
    st.write(result_data)

    # Ajouter la fonction describe()
    st.subheader("Mesure Statistique")
    selected_statistic = st.radio("Sélectionnez la mesure statistique :", ('All', 'mean', 'median', 'std', 'min', 'max'))

    # Sélecteur de colonnes pour les mesures statistiques
    selected_statistic_column = st.selectbox("Sélectionnez la colonne :", data.columns)

    if st.button("Afficher la mesure statistique"):
        if selected_statistic == 'All':
            result = data[selected_statistic_column].describe()
            result = result.round(2)  # Arrondir à deux chiffres après la virgule
            st.write(result)
        else:
            result = getattr(data[selected_statistic_column], selected_statistic)()
            result = round(result, 2)  # Arrondir à deux chiffres après la virgule
            st.write(result)

    # Sélection du type de graphique
    selected_chart_type = st.selectbox("Sélectionnez le type de graphique :", ['nuage des points', 'droite de reg', 'bar', 'histogram', 'heatmap', 'courbe'])

    # Saisie des colonnes pour le graphique choisi
    if selected_chart_type == 'heatmap':
         selected_x_columns = st.multiselect("Sélectionnez les colonnes pour l'axe X :", data.columns, key="selected_x_columns")
    elif selected_chart_type not in ['histogram', 'bar', 'courbe']:
        selected_x_column = st.selectbox("Saisissez la colonne X :", data.columns)
        selected_y_column = st.selectbox("Saisissez la colonne Y :", data.columns)
    else:
        selected_x_column = selected_y_column = st.selectbox("Saisissez la colonne de l'axe X :", data.columns)

    # Bouton pour afficher le graphique
    if st.button("Afficher le graphique"):
        try:
            if selected_chart_type == 'nuage des points':
                fig, ax = plt.subplots()
                ax.scatter(data.loc[selected_rows, selected_x_column], data.loc[selected_rows, selected_y_column])
                ax.set_xlabel(selected_x_column)
                ax.set_ylabel(selected_y_column)
                st.pyplot(fig)
            elif selected_chart_type == 'droite de reg':
                fig, ax = plt.subplots()
                sns.regplot(x=data.loc[selected_rows, selected_x_column], y=data.loc[selected_rows, selected_y_column], ax=ax)
                st.pyplot(fig)
            elif selected_chart_type == 'bar':
                fig, ax = plt.subplots()
                data.loc[selected_rows, selected_x_column].value_counts().plot(kind='bar', ax=ax)
                ax.set_xlabel(selected_x_column)
                ax.set_ylabel('Count')
                st.pyplot(fig)
            elif selected_chart_type == 'histogram':
                fig, ax = plt.subplots()
                ax.hist(data.loc[selected_rows, selected_x_column], bins=20)
                ax.set_xlabel(selected_x_column)
                ax.set_ylabel('Count')
                st.pyplot(fig)
            elif selected_chart_type == 'heatmap':
                fig, ax = plt.subplots()
                correlation_matrix = data[selected_x_columns].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)        
            elif selected_chart_type == 'courbe':
                fig, ax = plt.subplots()
                ax.plot(data.loc[selected_rows, selected_x_column].value_counts())
                ax.set_xlabel(selected_x_column)
                ax.set_ylabel('Count')
                st.pyplot(fig)
        except Exception as e:
             st.error(f"Erreur lors de l'affichage du graphique : {str(e)}")
             if "zero-size array to reduction operation fmin which has no identity" in str(e):
                 st.write("Erreur : La matrice de corrélation est de taille zéro. Veuillez vérifier vos données.")
    # Interface for Z test
    st.subheader("Z Test (Statistical Hypothesis Testing)")

    selected_column1 = st.selectbox("Select the first column for Z test:", data.columns)
    selected_column2 = st.selectbox("Select the second column for Z test:", data.columns)

    if st.button("Perform Z Test"):
        z_stat, p_value, z_test_result = perform_z_test(data, selected_column1, selected_column2)
        display_z_test_results(z_stat, p_value, z_test_result)

if __name__ == '__main__':
    main()