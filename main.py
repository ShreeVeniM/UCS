import warnings
import logging
from src.data_loader import load_data
from src.vizz import pairplot, scatter_plot, elbow_plot, silhouette_plot
from src.clustering import kmeans_clustering, calculate_wss, calculate_silhouette_scores

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info('Starting main function')
    try:
        # Load data
        filepath = 'src/dataset/mall_customers.csv'
        df = load_data(filepath)
        if df is None:
            logging.error('Failed to load data.')
            return
        logging.info('Data loaded successfully.')

        # Pairplot
        logging.info('Creating pairplot.')
        pairplot(df)

        # KMeans clustering with 5 clusters
        logging.info('Performing KMeans clustering with 5 clusters.')
        kmodel, df = kmeans_clustering(df, ['Annual_Income', 'Spending_Score'], 5)
        if kmodel is None or df is None:
            logging.error('KMeans clustering failed.')
            return
        logging.info('KMeans clustering completed.')

        # Scatter plot
        logging.info('Creating scatter plot.')
        scatter_plot(df, 'Annual_Income', 'Spending_Score', 'Cluster')

        # Calculate WSS for Elbow plot
        logging.info('Calculating WSS for Elbow plot.')
        wss = calculate_wss(df, ['Annual_Income', 'Spending_Score'], range(3, 9))
        if wss is None:
            logging.error('Failed to calculate WSS.')
            return
        logging.info('WSS calculation completed.')
        elbow_plot(wss['cluster'], wss['WSS_Score'])

        # Calculate silhouette scores
        logging.info('Calculating silhouette scores.')
        silhouette_scores = calculate_silhouette_scores(df, ['Annual_Income', 'Spending_Score'], range(3, 9))
        if silhouette_scores is None:
            logging.error('Failed to calculate silhouette scores.')
            return
        logging.info('Silhouette score calculation completed.')
        silhouette_plot(silhouette_scores['cluster'], silhouette_scores['Silhouette_Score'])

        # Train model on 'Age', 'Annual_Income', 'Spending_Score'
        logging.info('Calculating silhouette scores for three features.')
        silhouette_scores_3 = calculate_silhouette_scores(df, ['Age', 'Annual_Income', 'Spending_Score'], range(3, 9))
        if silhouette_scores_3 is None:
            logging.error('Failed to calculate silhouette scores for three features.')
            return
        silhouette_plot(silhouette_scores_3['cluster'], silhouette_scores_3['Silhouette_Score'])

        # Calculate WSS for Elbow plot on 3 features
        logging.info('Calculating WSS for Elbow plot with three features.')
        wss_3 = calculate_wss(df, ['Age', 'Annual_Income', 'Spending_Score'], range(3, 9))
        if wss_3 is None:
            logging.error('Failed to calculate WSS for three features.')
            return
        logging.info('WSS calculation for three features completed.')
        elbow_plot(wss_3['cluster'], wss_3['WSS_Score'])

    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
