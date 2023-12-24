import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from textblob import TextBlob 
import requests
from scipy.stats import zscore, norm
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
import numpy as np
from scipy.stats import gaussian_kde
import streamlit.components.v1 as com
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
# Dictionary mapping visualization types to numerical identifiers
visualization_types = {
    "Line Plot": 1,
    "Scatter Plot": 2,
    "Box Plot": 3,
    "Histogram": 4,
    "KDE Plot": 5,
    "Bar Plot": 6,
    "Heatmap1": 7,
    "Heatmap2": 8,
    "Pie Chart": 9,
    "Violin Plot": 10,
    "Pair Plot": 11,
}



def detect_distribution_type(column):
    # Check for normal distribution based on skewness and kurtosis
    skewness = column.skew()
    kurtosis = column.kurtosis()

    if abs(skewness) < 0.5 and abs(kurtosis) < 2:
        return "normal distribution"
    elif abs(skewness) >= 0.5 and abs(kurtosis) < 2:
        return "skewed distribution"
    elif abs(skewness) < 0.5 and abs(kurtosis) >= 2:
        return "heavy-tailed distribution"
    else:
        return "unknown distribution"
    
    
def visualize_data(df, visualization_type, selected_columns, distribution_type=None):
    try:
        # Display information about the chosen visualization type and selected columns
        st.write(f"Visualization Type: {visualization_type}")
        st.write(f"Selected Columns: {selected_columns}")
        
        if visualization_type == "Line Plot":
            # Display a subheader for the Line Plot section
            st.subheader("Line Plot")
            # Check the number of selected columns for Line Plot
            if len(selected_columns) == 1:
                y_variable = selected_columns[0]
                # Check if the selected column is present in the DataFrame
                if y_variable in df.columns:
                    # Create and display a Line Plot for the selected column
                    fig = px.line(df, y=y_variable, title=f"Line Plot for {y_variable}")
                    st.plotly_chart(fig)
                else:
                    st.warning("Please select a valid column for Line Plot.")
            elif len(selected_columns) == 2:
                x_variable, y_variable = selected_columns
                # Check if both selected columns are present in the DataFrame
                if x_variable in df.columns and y_variable in df.columns:
                    # Create and display a Line Plot for the selected columns
                    fig = px.line(df, x=x_variable, y=y_variable, title=f"Line Plot for {x_variable} and {y_variable}")
                    st.plotly_chart(fig)
                else:
                    st.warning("Please select valid columns for Line Plot.")
            else:
                st.warning("Please select one or two columns for Line Plot.")
        elif visualization_type == "Scatter Plot":
            # Check the number of selected columns for Scatter Plot
            if len(selected_columns) == 2:
                x_variable = selected_columns[0]
                y_variable = selected_columns[1]
                
                # Check if both selected columns are numeric
                if not pd.api.types.is_numeric_dtype(df[x_variable]) or not pd.api.types.is_numeric_dtype(df[y_variable]):
                    st.warning("Please select numeric columns for Scatter Plot.")
                else:
                    # Create and display a Scatter Plot for the selected columns
                    fig = px.scatter(df, x=x_variable, y=y_variable, title="Scatter Plot")
                    st.plotly_chart(fig)
            else:
                st.warning("Please select two columns for Scatter Plot.")

        elif visualization_type == "Box Plot":
            # Check if at least one column is selected for Box Plot
            if selected_columns:
                # Check if all selected columns are numeric
                if all(pd.api.types.is_numeric_dtype(df[column]) for column in selected_columns):
                    # Melt the DataFrame to reshape it for a single box plot
                    df_melted = df[selected_columns].melt(var_name='Variable', value_name='Value')
                    # Create a box plot with the melted DataFrame
                    fig = px.box(df_melted, x='Variable', y='Value', title="Box Plot")
                    st.plotly_chart(fig)
                else:
                    st.warning("Please select numeric columns for Box Plot.")
            else:
                st.warning("Please select at least one column for Box Plot.")

        elif visualization_type == "Histogram":
            # Check if exactly one column is selected for Histogram
            if len(selected_columns) == 1:
                x_variable = selected_columns[0]
                # Check if the selected column is present in the DataFrame
                if x_variable in df.columns:
                    # Check if the selected column is numeric
                    if pd.api.types.is_numeric_dtype(df[x_variable]):
                        fig = px.histogram(df, x=x_variable, title="Histogram")
                        st.plotly_chart(fig)
                    else:
                        st.warning("Please select a numeric column for Histogram.")
                else:
                    st.warning("Please select a valid column for Histogram.")
            else:
                st.warning("Please select only one column for Histogram.")

        elif visualization_type == "KDE Plot":
            # Check if exactly one column is selected for KDE Plot
            if len(selected_columns) == 1:
                selected_column = selected_columns[0]
                # Check if the selected column is numeric
                if not pd.api.types.is_numeric_dtype(df[selected_column]):
                    st.warning(f"The selected column '{selected_column}' is not numeric. Please choose a numeric column for KDE Plot.")
                else:
                    # Display a subheader for the KDE Plot section
                    st.subheader("Kernel Density Estimation (KDE) Plot:")

                    # Calculate KDE using scipy
                    kde = gaussian_kde(df[selected_column])
                    x_vals = np.linspace(df[selected_column].min(), df[selected_column].max(), 1000)
                    y_vals = kde(x_vals)

                    # Create KDE plot using Plotly Graph Objects
                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(width=2, color='blue'), fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.4)', name='KDE')
                    )

                    # Customize layout
                    fig.update_layout(
                        xaxis_title=selected_column,
                        yaxis_title="Density",
                        title="Kernel Density Estimation (KDE) Plot",
                    )

                    # Show the plot
                    st.plotly_chart(fig)

                    # Identify the distribution after displaying the KDE Plot
                    distribution_type = detect_distribution_type(df[selected_column])
                    st.write(f"This KDE plot may suggest a {distribution_type} distribution.")
            else:
                st.warning("Please select exactly one column for KDE Plot.")
        elif visualization_type == "Bar Plot":
            # Check if exactly two columns are selected for Bar Plot
            if len(selected_columns) == 2:
                x_variable, y_variable = selected_columns

                # Check if the selected columns are valid
                if x_variable in df.columns and y_variable in df.columns:
                    # Display data types and the selected data for exploration
                    st.write(df.dtypes[[x_variable, y_variable]])            
                    st.write(df[[x_variable, y_variable]])

                    # Check if the Y-axis variable is categorical
                    if pd.api.types.is_categorical_dtype(df[y_variable]):
                        category_orders = {
                            y_variable: df[y_variable].cat.categories.tolist()
                        }
                        fig = px.bar(df, x=x_variable, y=y_variable, title="Bar Plot", color=x_variable, category_orders=category_orders)
                    else:
                        # Check if the X-axis variable is categorical
                        if pd.api.types.is_categorical_dtype(df[x_variable]):
                            category_orders = {
                                x_variable: df[x_variable].cat.categories.tolist()
                            }
                            fig = px.bar(df, x=x_variable, y=y_variable, title="Bar Plot", color=x_variable, category_orders=category_orders)
                        else:
                            fig = px.bar(df, x=x_variable, y=y_variable, title="Bar Plot", color=x_variable)

                    st.plotly_chart(fig)
                else:
                    st.warning("Please select valid columns for Bar Plot.")
            else:
                st.warning("Please select two columns for Bar Plot.")

        elif visualization_type == "Heatmap1":
            # Check if at least two columns are selected for Heatmap
            if len(selected_columns) >= 2:
                # Check if the selected columns are present in the DataFrame
                if all(column in df.columns for column in selected_columns):
                    selected_data = df[selected_columns]

                    # Encode categorical columns to numerical values
                    for column in selected_data.select_dtypes(include='object').columns:
                        label_encoder = LabelEncoder()
                        selected_data[column] = label_encoder.fit_transform(selected_data[column])

                    corr_matrix = selected_data.corr()
                    # Create a heatmap with correlation values using Plotly
                    fig = px.imshow(
                        corr_matrix,
                        color_continuous_scale='Viridis',
                        labels=dict(x="X-axis", y="Y-axis", color="Correlation"),
                        height=500,
                        width=800,
                    )
                    # Add text annotations
                    for i, row in enumerate(corr_matrix.index):
                        for j, col in enumerate(corr_matrix.columns):
                            fig.add_annotation(
                                x=col,
                                y=row,
                                text=str(corr_matrix.iloc[i, j].round(2)),
                                showarrow=False,
                                font=dict(color='black'),
                            )

                    # Customize layout
                    fig.update_layout(
                        title="Heatmap with Correlation Values",
                        xaxis_title="X-axis",
                        yaxis_title="Y-axis",
                    )

                    st.plotly_chart(fig)
                else:
                    st.warning("Please select valid columns for Heatmap.")
            else:
                st.warning("Please select at least two columns for Heatmap.")
        elif visualization_type == "Heatmap2":
            # Select only numeric columns
            numeric_columns = df.select_dtypes(include=['number'])

            # Encode categorical columns to numerical values
            for column in df.select_dtypes(include='object').columns:
                label_encoder = LabelEncoder()
                df[column] = label_encoder.fit_transform(df[column])

            corr_matrix = numeric_columns.corr()

            # Convert NaN values to a string representation
            corr_matrix_string = corr_matrix.applymap(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")

            fig, ax = plt.subplots(figsize=(10, 8))
            # Create a heatmap with correlation values using Seaborn
            sns.heatmap(
                corr_matrix,
                annot=corr_matrix_string,
                cmap="viridis",
                fmt="",
                linewidths=.5,
                ax=ax,
                cbar_kws={"shrink": 0.8},
            )

            ax.set_title("Heatmap with Correlation Values")
            st.pyplot(fig)

        elif visualization_type == "Pie Chart":
            # Check if exactly one column is selected for Pie Chart
            if len(selected_columns) == 1:
                selected_column = selected_columns[0]

                # Check if the selected column is categorical
                if pd.api.types.is_categorical_dtype(df[selected_column]):
                    unique_values = len(df[selected_column].cat.categories)
                    if unique_values > 20:
                        st.warning("Too many unique values for Pie Chart. Select a column with fewer categories.")
                        return
                # Create a Pie Chart using Plotly
                fig = px.pie(df, names=selected_column, title=f"Pie Chart - {selected_column}")
                st.plotly_chart(fig)
            else:
                st.warning("Please select one column for Pie Chart.")

        elif visualization_type == "Violin Plot":
            
            # Check if at least one column is selected for Violin Plot
            if len(selected_columns) >= 1:
                # If one column is selected, create Violin Plot with y as the selected column
                if len(selected_columns) == 1:
                    y_variable = selected_columns[0]
                    if y_variable in df.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Violin(y=df[y_variable], box_visible=True, line_color='blue', fillcolor='lightblue', name=y_variable))
                        fig.update_layout(title=f"Violin Plot for {y_variable}", showlegend=True)
                        st.plotly_chart(fig)
                        st.markdown(f"Axe Y : {y_variable}")
                    else: 
                        st.warning("Please select a valid column for Violin Plot.")
                # If two columns are selected, create Violin Plot with x and y as the selected columns
                elif len(selected_columns) == 2:
                    x_variable =selected_columns[0] 
                    y_variable =selected_columns[1] 
                    if x_variable in df.columns and y_variable in df.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Violin(x=df[x_variable], y=df[y_variable], box_visible=True, line_color='blue', fillcolor='lightblue', name=y_variable))
                        fig.update_layout(title=f"Violin Plot for {x_variable} and {y_variable}", showlegend=True)
                        st.plotly_chart(fig)
                        st.markdown(f"Axe X : {x_variable}")
                        st.markdown(f"Axe Y : {y_variable}")
                    else:
                        st.warning("Please select valid columns for Violin Plot.")
                else:
                    st.warning("Please select only one or two columns for Violin Plot.")
            else:
                st.warning("Please select at least one column for Violin Plot.")
        elif visualization_type == "Pair Plot":
            try:
                # Check if at least two columns are selected for Pair Plot
                if len(selected_columns) >= 2:
                    # Filter numeric columns
                    numeric_columns = df[selected_columns].select_dtypes(include=['number'])
                    if not numeric_columns.empty:
                        # Create a Pair Plot using Plotly
                        fig = px.scatter_matrix(numeric_columns, title="Pair Plot")
                        st.plotly_chart(fig)
                    else:
                        st.warning("Please select at least two numeric columns for Pair Plot.")
                else:
                    st.warning("Please select at least two columns for Pair Plot.")
            except Exception as e:
                st.error(f"Error visualizing Pair Plot: {str(e)}")
                return None
            
        elif not visualization_type:
            st.warning("Please choose a valid visualization type.")
    except Exception as e:
        st.error(f"Error during visualization: {str(e)}")
        return None
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

# Function to perform sentiment analysis and add a "Feedback" column
def perform_fine_grained_sentiment_analysis(data, text_column):
    try:
        # Apply sentiment analysis using TextBlob with fine-grained categories
        data['Sentiment_Polarity'] = data[text_column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

        # Define thresholds for sentiment categories
        threshold_very_negative = -np.inf
        threshold_negative = -0.3
        threshold_mildly_negative = -0.1
        threshold_mildly_positive = 0.1
        threshold_positive = 0.3
        threshold_very_positive = np.inf

        # Categorize sentiments based on thresholds
        bins = [threshold_very_negative, threshold_negative, threshold_mildly_negative, threshold_mildly_positive, threshold_positive, threshold_very_positive]
        labels = ['Very Negative', 'Negative', 'Mildly Negative', 'Mildly Positive', 'Positive']

        # Calculate sentiment category
        data['Feedback'] = pd.cut(data['Sentiment_Polarity'], bins=bins, labels=labels, include_lowest=True)

        # Drop the temporary column used for calculation
        data = data.drop('Sentiment_Polarity', axis=1)

        return data
    except Exception as e:
        st.error(f"Error performing sentiment analysis: {str(e)}")
        return None



def perform_linear_regression(data, x_column, y_column):
    try:
        # Extract the selected columns
        x = data[[x_column]]
        y = data[y_column]

        # Create and fit the linear regression model
        model = LinearRegression()
        model.fit(x, y)

        # Get the coefficients and intercept
        slope = model.coef_[0]
        intercept = model.intercept_

        return slope, intercept
    except Exception as e:
        st.error(f"Error performing Linear Regression: {str(e)}")
        return None, None

def display_linear_regression_results(data, slope, intercept, x_column, y_column):
    st.subheader("Linear Regression Results")

    st.write(f"Slope (Coefficient): {slope:.6f}")
    st.write(f"Intercept: {intercept:.6f}")

    # Display the linear regression equation
    st.write("Linear Regression Equation:")
    st.latex(f"Y = {slope:.6f}X + {intercept:.6f}")

    # Create a scatter plot with the regression line
    fig = go.Figure()

    # Scatter plot for the original data points
    fig.add_trace(go.Scatter(x=data[x_column], y=data[y_column], mode='markers', name='Data Points'))

    # Regression line
    x_range = np.linspace(data[x_column].min(), data[x_column].max(), 100)
    y_range = slope * x_range + intercept
    fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', line=dict(color='red'), name='Regression Line'))

    # Customize layout
    fig.update_layout(
        title="Linear Regression",
        xaxis_title=x_column,
        yaxis_title=y_column,
    )

    # Show the plot
    st.plotly_chart(fig)



def perform_anova_test(data, column_group, column_values):
    try:
        groups = [data[data[column_group] == group][column_values] for group in data[column_group].unique()]
        f_stat, p_value = f_oneway(*groups)
        return f_stat, p_value
    except Exception as e:
        st.error(f"Error performing ANOVA test: {str(e)}")
        return None, None

def display_anova_test_results(f_stat, p_value, significance_level=0.05):
    st.subheader("ANOVA Test Results")

    st.write(f"P-Value: {p_value:.10f}")
    st.write(f"F Statistic: {f_stat:.10f}")

    if p_value < significance_level:
        st.success("Result: There is a significant difference between the means of the groups.")
    else:
        st.warning("Result: There is no significant difference between the means of the groups.")


def perform_chi2_test(data, column1, column2):
    try:
        # Perform Chi-square test
        contingency_table = pd.crosstab(data[column1], data[column2])
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
        return chi2_stat, p_value
    except Exception as e:
        st.error(f"Error performing Chi-square test: {str(e)}")
        return None, None

def display_chi2_test_results(chi2_stat, p_value, significance_level=0.05):
    st.subheader("Chi-square Test Results")

    st.write(f"P-Value: {p_value:.10f}")
    st.write(f"Chi-square Statistic: {chi2_stat:.10f}")

    if p_value < significance_level:
        st.success("Result: There is a significant association between the selected columns.")
    else:
        st.warning("Result: There is no significant association between the selected columns.")


def perform_t_test(data, column1, column2):
    try:
        t_stat, p_value = ttest_ind(data[column1], data[column2])
        return t_stat, p_value
    except Exception as e:
        st.error(f"Error performing t-test: {str(e)}")
        return None, None


def display_t_test_results(t_stat, p_value, significance_level=0.05):
    st.subheader("T-Test Results")

    st.write(f"P-Value: {p_value:.10f}")
    st.write(f"T Statistic: {t_stat:.10f}")

    if p_value < significance_level:
        st.success("Result: There is a significant difference between the means of the selected columns.")
    else:
        st.warning("Result: There is no significant difference between the means of the selected columns.")


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
#################################################################
    with open("style.css") as source:
        design = source.read()

    # Appliquer le CSS à tous les composants
    st.markdown(f"<style>{design}</style>", unsafe_allow_html=True)
    st.markdown("<h2 class='heading'>Projet Analyse des Données</h2>", unsafe_allow_html=True)
    with st.sidebar:
        selected = option_menu(
            menu_title = "MENU",
            options = ["Accueil","Fichier","Lien"]
        )
    if selected=="Accueil":
        st.markdown(f"<style>{design}</style>", unsafe_allow_html=True)
        st.markdown("<p class='subheader'>Souhaitez vous analyser vos données? Veuillez choisir une source de donnée svp!.</p>", unsafe_allow_html=True)
    elif selected=="Fichier":
        st.title(f"Veuillez ajouter le  {selected}")
        file_type = st.sidebar.selectbox("Sélectionnez le type de fichier :", ['CSV', 'XLSX', 'TXT'])

        # Fichier du fichier
        uploaded_file = st.file_uploader(f"Téléchargez un fichier {file_type}", type=[file_type.lower()])

        # Charger les données du fichier si disponible
        if uploaded_file is not None and uploaded_file:
            try:
                if file_type == 'CSV':
                    data = pd.read_csv(uploaded_file)
                elif file_type == 'XLSX':
                    # Sélection de la feuille
                    sheet_selector = st.sidebar.selectbox("Choisir la feuille", pd.ExcelFile(uploaded_file).sheet_names)  
                    print(sheet_selector)
                    data = pd.read_excel(uploaded_file, sheet_name=sheet_selector)
                elif file_type == 'TXT':
                    data = pd.read_csv(uploaded_file, sep='\t')  # Exemple avec séparateur tabulation, ajustez en fonction de votre format

                process_data(data)

            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")
    elif selected=="Lien":
        st.markdown(f"Veuillez ajouter le  {selected}")
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
                        selected_statistic = st.sidebar.radio(f"Sélectionnez la mesure statistique (Table {i + 1}):", ('mean', 'median', 'std', 'min', 'max'), key=f"statistic_{i}")

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

                        # Updated visualization code
                        selected_visualization_type = st.sidebar.selectbox(f"Select the visualization type (Table {i + 1}):", list(visualization_types.keys()))
                        selected_columns = st.sidebar.multiselect(f"Select columns for visualization (Table {i + 1}):", wiki_df.columns)

                        # Check if the visualization type is not empty
                        if selected_visualization_type:
                            visualize_data(wiki_df, selected_visualization_type, selected_columns)

                        # End of updated visualization code

            except Exception as e:
                st.error(f"Erreur lors de la lecture des données depuis le lien : {e}")

def process_data(data):

    st.sidebar.title("Mesures Statistiques")
    # Sélection des colonnes et lignes
    selected_columns = st.sidebar.multiselect("Sélectionnez les colonnes :", data.columns)
    selected_rows = st.sidebar.multiselect("Sélectionnez les lignes :", data.index)

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
    st.markdown("<p class='subheader'>Mesure Statistique</p>", unsafe_allow_html=True)

    selected_statistic = st.sidebar.radio("Sélectionnez la mesure statistique :", ('All', 'mean', 'median', 'std', 'min', 'max'))

    # Sélecteur de colonnes pour les mesures statistiques
    selected_statistic_column = st.sidebar.selectbox("Sélectionnez la colonne :", data.columns)

    if st.sidebar.button("Afficher la mesure statistique"):
        if selected_statistic == 'All':
            result = data[selected_statistic_column].describe()
            result = result.round(2)  # Arrondir à deux chiffres après la virgule
            st.write(result)
        else:
            result = getattr(data[selected_statistic_column], selected_statistic)()
            result = round(result, 2)  # Arrondir à deux chiffres après la virgule
            st.write(result)

    # Sélection du type de graphique
    st.sidebar.title("Visualisation")
    selected_chart_type = st.sidebar.selectbox("Sélectionnez le type de graphique :", ['nuage des points', 'droite de reg', 'bar', 'histogram', 'heatmap', 'courbe'])

    # Saisie des colonnes pour le graphique choisi
    if selected_chart_type == 'heatmap':
         selected_x_columns = st.sidebar.multiselect("Sélectionnez les colonnes pour l'axe X :", data.columns, key="selected_x_columns")
    elif selected_chart_type not in ['histogram', 'bar', 'courbe']:
        selected_x_column = st.sidebar.selectbox("Saisissez la colonne X :", data.columns)
        selected_y_column = st.sidebar.selectbox("Saisissez la colonne Y :", data.columns)
    else:
        selected_x_column = selected_y_column = st.sidebar.selectbox("Saisissez la colonne de l'axe X :", data.columns)

    # Bouton pour afficher le graphique
    if st.sidebar.button("Afficher le graphique"):
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
    st.markdown("<p class='subheader'>Z Test (Statistical Hypothesis Testing).</p>", unsafe_allow_html=True)
 
    st.sidebar.title("Z Test")
    selected_column1 = st.sidebar.selectbox("Select the first column for Z test:", data.columns)
    selected_column2 = st.sidebar.selectbox("Select the second column for Z test:", data.columns)

    if st.sidebar.button("Perform Z Test"):
        z_stat, p_value, z_test_result = perform_z_test(data, selected_column1, selected_column2)
        display_z_test_results(z_stat, p_value, z_test_result)
    # Interface for T test
    st.markdown("<p class='subheader'>T Test (Statistical Hypothesis Testing)</p>", unsafe_allow_html=True)

    st.sidebar.title("T Test")
    selected_column1_t = st.sidebar.selectbox("Select the first column for T test:", data.columns)
    selected_column2_t = st.sidebar.selectbox("Select the second column for T test:", data.columns)

    if st.sidebar.button("Perform T Test"):
        t_stat, p_value = perform_t_test(data, selected_column1_t, selected_column2_t)
        display_t_test_results(t_stat, p_value)
    # Interface for Chi-square test
    st.markdown("<p class='subheader'>Chi-square Test (Statistical Hypothesis Testing)</p>", unsafe_allow_html=True)

    st.sidebar.title("Chi-2 Test")
    selected_column1_chi2 = st.sidebar.selectbox("Select the first column for Chi-square test:", data.columns)
    selected_column2_chi2 = st.sidebar.selectbox("Select the second column for Chi-square test:", data.columns)

    if st.sidebar.button("Perform Chi-square Test"):
        chi2_stat, p_value_chi2 = perform_chi2_test(data, selected_column1_chi2, selected_column2_chi2)
        display_chi2_test_results(chi2_stat, p_value_chi2)
    # Interface for Linear Regression
    st.markdown("<p class='subheader'>Linear Regression</p>", unsafe_allow_html=True)

    st.sidebar.title("Linear Regression")
    selected_x_column_lr = st.sidebar.selectbox("Select the independent variable (X) for Linear Regression:", data.columns)
    selected_y_column_lr = st.sidebar.selectbox("Select the dependent variable (Y) for Linear Regression:", data.columns)

    if st.sidebar.button("Perform Linear Regression"):
        slope_lr, intercept_lr = perform_linear_regression(data, selected_x_column_lr, selected_y_column_lr)
        display_linear_regression_results(data, slope_lr, intercept_lr, selected_x_column_lr, selected_y_column_lr)
    # Interface for sentiment analysis
    st.markdown("<p class='subheader'>Sentiment Analysis</p>", unsafe_allow_html=True)

    st.sidebar.title("Sentiment Analysis")
    # Select the column containing text data for sentiment analysis
    text_column_for_sentiment = st.sidebar.selectbox("Select the column for sentiment analysis:", data.select_dtypes(include='object').columns)

    # Button to perform fine-grained sentiment analysis and add a new "Feedback" column
    if st.sidebar.button("Perform Fine-Grained Sentiment Analysis"):
        data = perform_fine_grained_sentiment_analysis(data, text_column_for_sentiment)
        st.write("Data with Fine-Grained Sentiment Analysis:")
        st.write(data)
    
    
    # Interface for ANOVA test
    st.markdown("<p class='subheader'>ANOVA Test (Analysis of Variance)</p>", unsafe_allow_html=True)

    st.sidebar.title("ANOVA Test")
    selected_group_column_anova = st.sidebar.selectbox("Select the column for grouping:", data.columns)
    selected_values_column_anova = st.sidebar.selectbox("Select the column for ANOVA test:", data.columns)

    if st.sidebar.button("Perform ANOVA Test"):
        f_stat_anova, p_value_anova = perform_anova_test(data, selected_group_column_anova, selected_values_column_anova)
        display_anova_test_results(f_stat_anova, p_value_anova)

if __name__ == '__main__':

    main()
