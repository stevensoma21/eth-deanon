"""
Clustering analysis functions for Ethereum transaction analysis
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan
from ..config.settings import DBSCAN_PARAMS, HDBSCAN_PARAMS

logger = logging.getLogger(__name__)

def evaluate(features, labels):
    """
    Compute clustering validity metrics, ignoring noise (-1).
    Returns a dict or None if too few clusters.
    """
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return None
    feats = features[mask]
    lbls = labels[mask]
    return {
        'silhouette': silhouette_score(feats, lbls),
        'calinski_harabasz': calinski_harabasz_score(feats, lbls),
        'davies_bouldin': davies_bouldin_score(feats, lbls),
        'n_clusters': len(set(lbls)),
        'n_noise': int((labels == -1).sum())
    }

def apply_clustering(df, methods=None, dbscan_params=None, hdbscan_params=None, optimize=True):
    """
    Apply clustering with optional hyperparameter optimization.
    If optimize=True, runs grid search with default ranges to find best params.
    Otherwise uses provided param dicts or defaults.
    """
    # Prepare features
    X = df[['degree','entropy']].copy()
    X['degree'] = np.log1p(X['degree'])
    features = StandardScaler().fit_transform(X)

    if methods is None:
        methods = ['dbscan','hdbscan','kmeans']

    # Optionally optimize
    if optimize:
        if 'dbscan' in methods and dbscan_params is None:
            eps_vals = np.linspace(0.5, 2.0, 10)
            ms_vals = [5,10,20]
            best_db, db_records = optimize_dbscan(features, eps_vals, ms_vals)
            logger.info("Best DBSCAN params:\n%s", best_db)
            dbscan_params = {'eps': best_db['eps'], 'min_samples': best_db['min_samples']}
        if 'hdbscan' in methods and hdbscan_params is None:
            mcs_list = [20,50,100]
            ms_list  = [5,10,20]
            best_hdb, hdb_records = optimize_hdbscan(features, mcs_list, ms_list)
            logger.info("Best HDBSCAN params:\n%s", best_hdb)
            hdbscan_params = {'min_cluster_size': best_hdb['min_cluster_size'], 'min_samples': best_hdb['min_samples']}

    # Apply clustering
    if 'dbscan' in methods:
        params = dbscan_params or DBSCAN_PARAMS
        db = DBSCAN(**params)
        lbl = db.fit_predict(features)
        df['dbscan_cluster'] = lbl
        logger.info("DBSCAN evaluation: %s", evaluate(features, lbl))

    if 'hdbscan' in methods:
        params = hdbscan_params or HDBSCAN_PARAMS
        hdb = hdbscan.HDBSCAN(**{k:v for k,v in params.items() if v is not None})
        lbl = hdb.fit_predict(features)
        df['hdbscan_cluster'] = lbl
        logger.info("HDBSCAN evaluation: %s", evaluate(features, lbl))

    if 'kmeans' in methods:
        km = KMeans(n_clusters=5, random_state=42)
        lbl = km.fit_predict(features)
        df['kmeans_cluster'] = lbl
        logger.info("KMeans evaluation: %s", evaluate(features, lbl))

    return df

def optimize_dbscan(features, eps_values, min_samples_values):
    """
    Grid-search DBSCAN over eps and min_samples, return best metrics and all records.
    """
    best = {'silhouette': -1}
    records = []
    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms)
            lbl = db.fit_predict(features)
            metrics = evaluate(features, lbl)
            if metrics:
                metrics.update({'eps': eps, 'min_samples': ms})
                records.append(metrics)
                if metrics['silhouette'] > best['silhouette']:
                    best = metrics
    return best, pd.DataFrame(records)

def optimize_hdbscan(features, min_cluster_sizes, min_samples_list):
    """
    Grid-search HDBSCAN over min_cluster_size and min_samples.
    """
    best = {'silhouette': -1}
    records = []
    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
            lbl = hdb.fit_predict(features)
            metrics = evaluate(features, lbl)
            if metrics:
                metrics.update({'min_cluster_size': mcs, 'min_samples': ms})
                records.append(metrics)
                if metrics['silhouette'] > best['silhouette']:
                    best = metrics
    return best, pd.DataFrame(records)
