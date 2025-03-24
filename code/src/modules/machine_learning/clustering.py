import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from kneed import KneeLocator

def load_data():
    """Uploads dataset for clustering."""
    uploaded_file = st.file_uploader("📂 Upload dataset for clustering", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

def find_optimal_k(df):
    """Finds the optimal number of clusters using the Elbow Method."""
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numeric_features])

    sse = []
    K = range(2, 11)  # Checking K from 2 to 10
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        sse.append(kmeans.inertia_)

    optimal_k = KneeLocator(K, sse, curve="convex", direction="decreasing").elbow
    return optimal_k if optimal_k else 3  # Default to 3 if no clear elbow is found

def perform_kmeans_clustering(df, n_clusters):
    """Applies KMeans clustering with PCA for visualization."""
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Scale the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numeric_features])

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(df_scaled)

    # Apply PCA for visualization
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_scaled)
    df["PCA1"] = pca_components[:, 0]
    df["PCA2"] = pca_components[:, 1]

    return df, kmeans

def clustering_main():
    st.title("🔍 K-Means Clustering with PCA Visualization")
    df = load_data()

    if df is not None:
        st.write("### Preview of Dataset")
        st.dataframe(df.head())

        # Find optimal K
        optimal_k = find_optimal_k(df)
        st.write(f"📌 Recommended number of clusters (K): **{optimal_k}**")

        k = st.slider("Select Number of Clusters (K)", 2, 10, optimal_k, 1)

        if st.button("Run Clustering"):
            df, kmeans = perform_kmeans_clustering(df, k)

            st.write("### Clustering Results")
            st.dataframe(df)

            # Visualization using PCA
            fig = px.scatter(df, x="PCA1", y="PCA2", color=df["cluster"].astype(str),
                             title="K-Means Clustering (PCA Visualization)", labels={"color": "Cluster"},
                             hover_data=df.columns)
            st.plotly_chart(fig)

            st.download_button("📥 Download Clustered Data", df.to_csv(index=False), "clustering_results.csv", "text/csv")
