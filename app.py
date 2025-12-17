import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances

# --- Page Configuration ---
st.set_page_config(page_title="Wholesale Customer Segmentation", layout="wide")

# --- Custom CSS for Background and Fonts ---
# This adds the grocery background image and enforces dark fonts
page_bg_img = """
<style>
/* App background image */
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1542838132-92c53300491e?q=80&w=2574&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Make the sidebar slightly transparent white so text is readable */
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.9);
}

/* container styling to make text readable over the image */
.block-container {
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

/* Force Dark Fonts */
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
    color: #1a1a1a !important;
    font-family: 'Helvetica', sans-serif;
}

/* Button styling */
.stButton>button {
    color: white;
    background-color: #2e7d32; /* Green for groceries */
    border: none;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Title ---
st.title("🛒 Wholesale Customer Clustering App")
st.markdown("### Hierarchical Clustering Implementation")
st.write("Enter annual spending amounts (m.u.) to classify a customer into a segment.")

# --- 1. Load and Process Data ---
@st.cache_data
def load_and_train_model():
    # Using the direct URL from UCI (Source of the dataset used in the PDF)
    # This ensures the app runs without needing the local PDF/CSV file.
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
    try:
        df = pd.read_csv(url)
    except:
        st.error("Could not load data. Please check your internet connection.")
        return None, None, None, None

    # Preprocessing (As per PDF Page 2)
    # Dropping Channel and Region as done in your notebook
    X = df.drop(['Channel', 'Region'], axis=1)
    
    # Scaling (As per PDF Page 3)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Clustering (As per PDF Page 4 - Ward Linkage, 3 Clusters)
    hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
    y_clusters = hc.fit_predict(X_scaled)
    
    # Add clusters back to data for visualization
    df['Cluster'] = y_clusters
    
    # Calculate Centroids to allow "prediction" for new user inputs
    # (Finding the average of each cluster to see where new data fits best)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled['Cluster'] = y_clusters
    centroids = df_scaled.groupby('Cluster').mean()
    
    return df, scaler, centroids, X.columns

df, scaler, centroids, feature_names = load_and_train_model()

if df is not None:
    # --- 2. Sidebar: User Input ---
    st.sidebar.header("📝 Customer Spending Input")
    
    def user_input_features():
        # Default values set near the mean of the dataset
        fresh = st.sidebar.number_input("Fresh Products", min_value=0, value=12000)
        milk = st.sidebar.number_input("Milk Products", min_value=0, value=5000)
        grocery = st.sidebar.number_input("Grocery", min_value=0, value=7000)
        frozen = st.sidebar.number_input("Frozen Products", min_value=0, value=3000)
        det_paper = st.sidebar.number_input("Detergents & Paper", min_value=0, value=3000)
        delica = st.sidebar.number_input("Delicassen", min_value=0, value=1500)
        
        data = {
            'Fresh': fresh,
            'Milk': milk,
            'Grocery': grocery,
            'Frozen': frozen,
            'Detergents_Paper': det_paper,
            'Delicassen': delica
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    # --- 3. Main Area: Prediction Logic ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Your Input")
        st.dataframe(input_df.T, height=250)

    with col2:
        if st.button("Predict Segment"):
            # Scale the user input using the SAME scaler from training
            input_scaled = scaler.transform(input_df)
            
            # Calculate distance to the 3 centroids
            dists = euclidean_distances(input_scaled, centroids.values)
            predicted_cluster = np.argmin(dists)
            
            st.success(f"### Result: Cluster {predicted_cluster}")
            
            # Interpretation based on general dataset knowledge
            if predicted_cluster == 0:
                st.info("**Interpretation:** This segment typically represents high-volume buyers.")
            elif predicted_cluster == 1:
                st.info("**Interpretation:** This segment typically represents standard/moderate buyers.")
            else:
                st.info("**Interpretation:** This segment typically represents small retailers/restaurants.")

            # --- 4. Visualization (Replicating PDF Page 8/9 Scatterplot) ---
            st.markdown("### Visualization: Milk vs Grocery")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot all existing points
            sns.scatterplot(x=df['Milk'], y=df['Grocery'], hue=df['Cluster'], palette='viridis', alpha=0.6, ax=ax)
            
            # Plot user Input
            ax.scatter(input_df['Milk'], input_df['Grocery'], color='red', s=200, marker='*', label='Your Input')
            
            plt.title("Hierarchical Clustering Result with Your Input")
            plt.legend()
            st.pyplot(fig)
