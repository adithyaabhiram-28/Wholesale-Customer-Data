import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🟢 Customer Segmentation Dashboard")
st.markdown(
    "This system uses **K-Means Clustering** to group customers based on their purchasing behavior and similarities."
)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Wholesale customers data.csv")

df = load_data()

numerical_features = df.select_dtypes(include=np.number).columns.tolist()

# -----------------------------
# Sidebar - Input Controls
# -----------------------------
st.sidebar.header("🔧 Clustering Controls")

selected_features = st.sidebar.multiselect(
    "Select Features (minimum 2)",
    numerical_features,
    default=numerical_features[:2]
)

k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3)

random_state = st.sidebar.number_input(
    "Random State (optional)",
    min_value=0,
    value=42,
    step=1
)

run_clustering = st.sidebar.button("🟦 Run Clustering")

# -----------------------------
# Validation
# -----------------------------
if len(selected_features) < 2:
    st.warning("Please select **at least two numerical features**.")
    st.stop()

# -----------------------------
# Run K-Means
# -----------------------------
if run_clustering:
    X = df[selected_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init="auto"
    )

    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters

    # -----------------------------
    # Visualization Section
    # -----------------------------
    st.subheader("📊 Customer Clusters Visualization")

    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster_id in range(k):
        cluster_data = df[df["Cluster"] == cluster_id]
        ax.scatter(
            cluster_data[selected_features[0]],
            cluster_data[selected_features[1]],
            label=f"Cluster {cluster_id}",
            alpha=0.7
        )

    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)

    ax.scatter(
        centers_original[:, 0],
        centers_original[:, 1],
        s=200,
        c="black",
        marker="X",
        label="Cluster Centers"
    )

    ax.set_xlabel(selected_features[0])
    ax.set_ylabel(selected_features[1])
    ax.set_title("Customer Segments Based on Selected Features")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # -----------------------------
    # Cluster Summary Table
    # -----------------------------
    st.subheader("📋 Cluster Summary")

    summary = (
        df.groupby("Cluster")[selected_features]
        .agg(["count", "mean"])
        .round(2)
    )

    summary.columns = ["_".join(col) for col in summary.columns]
    st.dataframe(summary, use_container_width=True)

    # -----------------------------
    # Business Interpretation
    # -----------------------------
    st.subheader("💡 Business Interpretation")

    for cluster_id in range(k):
        cluster_size = (df["Cluster"] == cluster_id).sum()
        avg_values = df[df["Cluster"] == cluster_id][selected_features].mean()

        dominant_feature = avg_values.idxmax()

        st.markdown(
            f"""
            🟢 **Cluster {cluster_id}**  
            Customers in this group tend to spend more on **{dominant_feature}**  
            and show similar purchasing behavior across selected categories.
            """
        )

    # -----------------------------
    # User Guidance Box
    # -----------------------------
    st.info(
        "Customers in the same cluster exhibit similar purchasing behaviour and "
        "can be targeted with similar business strategies."
    )

else:
    st.info("👈 Select features and click **Run Clustering** to begin.")
