from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

def kmeans_clustering(df, features, n_clusters):
    try:
        kmodel = KMeans(n_clusters=n_clusters).fit(df[features])
        df['Cluster'] = kmodel.labels_
        return kmodel, df
    except Exception as e:
        print(f"Error in KMeans clustering: {e}")
        return None, None

def calculate_wss(df, features, k_range):
    try:
        K = []
        WCSS = []
        for i in k_range:
            kmodel = KMeans(n_clusters=i).fit(df[features])
            wcss_score = kmodel.inertia_
            WCSS.append(wcss_score)
            K.append(i)
        return pd.DataFrame({'cluster': K, 'WSS_Score': WCSS})
    except Exception as e:
        print(f"Error calculating WSS: {e}")
        return None

def calculate_silhouette_scores(df, features, k_range):
    try:
        K = []
        ss = []
        for i in k_range:
            kmodel = KMeans(n_clusters=i).fit(df[features])
            ypred = kmodel.labels_
            sil_score = silhouette_score(df[features], ypred)
            K.append(i)
            ss.append(sil_score)
        return pd.DataFrame({'cluster': K, 'Silhouette_Score': ss})
    except Exception as e:
        print(f"Error calculating silhouette scores: {e}")
        return None
