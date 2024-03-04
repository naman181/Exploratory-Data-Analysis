from flask import Flask, render_template
from io import BytesIO
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

from ucimlrepo import fetch_ucirepo

app = Flask(__name__)

# Fetch dataset using fetch_ucirepo
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
X_no_outliers = None

# Extract features and targets
X = pd.DataFrame(breast_cancer_wisconsin_diagnostic.data.features, columns=breast_cancer_wisconsin_diagnostic.feature_names)
y = pd.Series(breast_cancer_wisconsin_diagnostic.data.target, name='target', dtype='category')

numerical_cols = X.select_dtypes(include=[np.number]).columns

X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].mean())
X = X.drop_duplicates()

# Function to convert matplotlib figure to base64 for HTML rendering
def fig_to_base64(fig):
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def remove_outliers():
    global X_no_outliers
    numeric_X = X.select_dtypes(include=[np.number])
    z_scores = np.abs((numeric_X - numeric_X.mean()) / numeric_X.std())

    outliers=np.where(z_scores>3)
    X_no_outliers = numeric_X[(z_scores<3).all(axis=1)]

# Routes for each step in the EDA process
@app.route('/')
def home():
    return """
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>EDA Web App</title>
    </head>
    <body>
        <h1>Exploratory Data Analysis (EDA)</h1>
        <ul>
            <li><a href="/show_all_data">Show All Data</a></li>
            <li><a href="/missing_values">Check Missing Values</a></li>
            <li><a href="/handle_missing_values">Handle Missing Values</a></li>
            <li><a href="/remove_duplicates">Remove Duplicates</a></li>
            <li><a href="/univariate_histogram">Univariate Histogram</a></li>
            <li><a href="/univariate_boxplot">Univariate Box Plot</a></li>
            <li><a href="/bivariate_scatterplot">Bivariate Scatter Plot</a></li>
            <li><a href="/correlation_matrix">Correlation Matrix</a></li>
            <li><a href="/outlier_removal">Outlier Removal</a></li>
            <li><a href="/univariate_histogram_no_outliers">Univariate Histogram (Outliers Removed)</a></li>
            <li><a href="/univariate_boxplot_no_outliers">Univariate Box Plot (Outliers Removed)</a></li>
            <li><a href="/bivariate_scatterplot_no_outliers">Bivariate Scatter Plot (Outliers Removed)</a></li>
            <li><a href="/correlation_matrix_no_outliers">Correlation Matrix (Outliers Removed)</a></li>
        </ul>
    </body>
    </html>
    """

# Route for checking missing values
@app.before_request
def before_request():
    remove_outliers()

# Route to show all data
@app.route('/show_all_data')
def show_all_data():
    global X
    return f"<html><body>{X.to_html()}</body></html>"

@app.route('/missing_values')
def missing_values():
    missing_values_counts = X.isnull().sum()
    return f"<html><body><p>{missing_values_counts}</p></body></html>"

# Route for handling missing values
@app.route('/handle_missing_values')
def handle_missing_values():
    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].mean())
    return "<html><body><p>Missing values handled</p></body></html>"

# Route for removing duplicates
@app.route('/remove_duplicates')
def remove_duplicates():
    global X
    original_duplicates_count = X.duplicated().sum()
    X_no_duplicates = X.drop_duplicates()
    html_table = X_no_duplicates.to_html()
    X = X.drop_duplicates()
    return "<html><body><p>Duplicates removed</p></body></html>"

# Route for univariate analysis - Histograms
@app.route('/univariate_histogram')
def univariate_histogram():
    fig, ax = plt.subplots(figsize=(12, 8))
    X.hist(ax=ax)
    plt.suptitle('Histograms of Numerical Variables')
    img = fig_to_base64(fig)
    return render_template('figure.html', image=img)

# Route for univariate analysis - Box Plots
@app.route('/univariate_boxplot')
def univariate_boxplot():
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=X, ax=ax)
    plt.title('Box Plots of Numerical Variables')
    img = fig_to_base64(fig)
    return render_template('figure.html', image=img)

# Route for bivariate analysis - Scatter Plots
@app.route('/bivariate_scatterplot')
def bivariate_scatterplot():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create scatter plot for each pair of numerical variables
    for i, col1 in enumerate(numerical_cols):
        for j, col2 in enumerate(numerical_cols):
            if i < j:
                ax.scatter(X[col1], X[col2], label=f'{col1} vs {col2}')

    ax.set_title('Bivariate Scatter Plots')
    ax.legend()
    img = fig_to_base64(fig)
    return render_template('figure.html', image=img)

# Route for correlation matrix
@app.route('/correlation_matrix')
def correlation_matrix():
    numeric_X = X.select_dtypes(include=[np.number])
    correlation_matrix = numeric_X.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    img = fig_to_base64(plt)
    return render_template('figure.html', image=img)

# Route for outlier detection and removal
@app.route('/outlier_removal')
def outlier_removal():
    global X_no_outliers
    numeric_X = X.select_dtypes(include=[np.number])
    z_scores = np.abs((numeric_X - numeric_X.mean()) / numeric_X.std())
    outliers = np.where(z_scores > 3)
    X_no_outliers = numeric_X[(z_scores < 3).all(axis=1)]

    # Display the removed outliers
    removed_outliers_count = numeric_X.shape[0] - X_no_outliers.shape[0]
    return f"<html><body><p>Number of outliers removed: {removed_outliers_count}</p></body></html>"

# Route for univariate analysis after handling outliers - Histograms
@app.route('/univariate_histogram_no_outliers')
def univariate_histogram_no_outliers():
    fig, ax = plt.subplots(figsize=(12, 8))
    X_no_outliers.hist(ax=ax)
    plt.suptitle('Histograms of Numerical Variables (Outliers Removed)')
    img = fig_to_base64(fig)
    return render_template('figure.html', image=img)

# Route for univariate analysis after handling outliers - Box Plots
@app.route('/univariate_boxplot_no_outliers')
def univariate_boxplot_no_outliers():
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=X_no_outliers, ax=ax)
    plt.title('Box Plots of Numerical Variables (Outliers Removed)')
    img = fig_to_base64(fig)
    return render_template('figure.html', image=img)

# Route for bivariate analysis after handling outliers - Scatter Plots
@app.route('/bivariate_scatterplot_no_outliers')
def bivariate_scatterplot_no_outliers():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create scatter plot for each pair of numerical variables after outlier removal
    for i, col1 in enumerate(numerical_cols):
        for j, col2 in enumerate(numerical_cols):
            if i < j:
                ax.scatter(X_no_outliers[col1], X_no_outliers[col2], label=f'{col1} vs {col2}')

    ax.set_title('Bivariate Scatter Plots (Outliers Removed)')
    ax.legend()
    img = fig_to_base64(fig)
    return render_template('figure.html', image=img)

# Route for correlation matrix after handling outliers
@app.route('/correlation_matrix_no_outliers')
def correlation_matrix_no_outliers():
    correlation_matrix_no_outliers = X_no_outliers.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix_no_outliers, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix (Outliers Removed)')
    img = fig_to_base64(plt)
    return render_template('figure.html', image=img)

if __name__ == '__main__':
    app.run(debug=True)