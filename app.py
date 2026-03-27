import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score, silhouette_score)
from scipy.cluster.hierarchy import dendrogram, linkage
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="Vastraa Analytics Dashboard",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #1B2A4A; margin-bottom: 0.2rem;}
    .sub-header {font-size: 1.1rem; color: #888; margin-bottom: 1.5rem;}
    .metric-card {background: linear-gradient(135deg, #f5f0eb 0%, #fff 100%);
                  border-radius: 12px; padding: 1.2rem; border-left: 4px solid #D4A574;
                  box-shadow: 0 2px 8px rgba(0,0,0,0.06);}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {padding: 8px 16px; font-weight: 600;}
    div[data-testid="stMetric"] {background: #f8f6f3; border-radius: 10px; padding: 12px;
                                  border-left: 3px solid #D4A574;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("vastraa_cleaned_dataset.csv")
    return df

df = load_data()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🧵 VASTRAA")
    st.markdown("*Data-Driven Textile Intelligence*")
    st.markdown("---")

    page = st.radio(
        "📊 Navigation",
        [
            "1. Executive Summary",
            "2. EDA Deep Dive",
            "3. Customer Segmentation",
            "4. Purchase Prediction",
            "5. Spend Prediction",
            "6. Market Basket Analysis",
            "7. Prescriptive Strategy",
            "8. New Customer Prediction"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("**Global Filters**")
    gender_filter = st.multiselect("Gender", df["Q1_Gender"].unique(), default=df["Q1_Gender"].unique())
    age_filter = st.multiselect("Age Group", df["Q2_Age_Group"].unique(), default=df["Q2_Age_Group"].unique())
    city_filter = st.multiselect("City Tier", df["Q5_City_Tier"].unique(), default=df["Q5_City_Tier"].unique())
    income_filter = st.multiselect("Income", df["Q3_Annual_Income"].unique(), default=df["Q3_Annual_Income"].unique())

filtered_df = df[
    (df["Q1_Gender"].isin(gender_filter)) &
    (df["Q2_Age_Group"].isin(age_filter)) &
    (df["Q5_City_Tier"].isin(city_filter)) &
    (df["Q3_Annual_Income"].isin(income_filter))
]

# ============================================================
# HELPER: get feature columns for ML
# ============================================================
def get_ml_features(dataframe):
    exclude = ["Respondent_ID", "_Latent_Persona", "Q30_Purchase_Intent",
               "Q30_Purchase_Intent_Binary", "Q30_Intent_Code"]
    cat_text_cols = [c for c in dataframe.columns if dataframe[c].dtype == 'object']
    drop_cols = list(set(exclude + cat_text_cols))
    feature_cols = [c for c in dataframe.columns if c not in drop_cols]
    return feature_cols

VASTRAA_COLORS = ["#1B2A4A", "#C0392B", "#D4A574", "#2E86AB", "#A23B72",
                  "#F18F01", "#3C6E71", "#E8D5B7", "#284B63", "#FF6B6B"]

# ============================================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================================
if page == "1. Executive Summary":
    st.markdown('<p class="main-header">📊 Executive Summary & KPI Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Descriptive Analytics — What does our market look like?</p>', unsafe_allow_html=True)

    fdf = filtered_df

    # KPI Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Respondents", f"{len(fdf):,}")
    c2.metric("Avg Annual Spend", f"₹{fdf['Q15_Annual_Spend_Midpoint_INR'].mean():,.0f}")
    c3.metric("Purchase Intent Rate", f"{fdf['Q30_Purchase_Intent_Binary'].mean()*100:.1f}%")
    c4.metric("Avg Digital Readiness", f"{fdf['Derived_Digital_Readiness_Score'].mean():.1f}/5")
    c5.metric("Gifting Rate", f"{fdf['Derived_Is_Gifter'].mean()*100:.1f}%")
    top_product = max(
        ["Handloom Sarees", "Block Print Kurtas & Dupatta Sets", "Handloom Stoles & Scarves",
         "Artisan Home Textiles", "Mens Handloom Kurtas & Nehru Jackets"],
        key=lambda x: fdf[[c for c in fdf.columns if c.startswith("Q6_")][
            ["Handloom Sarees", "Block Print Kurtas & Dupatta Sets", "Handloom Stoles & Scarves",
             "Artisan Home Textiles", "Mens Handloom Kurtas & Nehru Jackets"].index(x)]].sum()
    )
    c6.metric("Top Product", top_product.split(" ")[0])

    st.markdown("---")

    # Row 1: Demographics
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(fdf, names="Q1_Gender", title="Gender Distribution",
                     color_discrete_sequence=VASTRAA_COLORS, hole=0.4)
        fig.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        age_order = ["18-24", "25-34", "35-44", "45-54", "55+"]
        age_counts = fdf["Q2_Age_Group"].value_counts().reindex(age_order).fillna(0)
        fig = px.bar(x=age_counts.index, y=age_counts.values, title="Age Group Distribution",
                     color=age_counts.index, color_discrete_sequence=VASTRAA_COLORS,
                     labels={"x": "Age Group", "y": "Count"})
        fig.update_layout(height=350, margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: City & Income
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(fdf, names="Q5_City_Tier", title="City Tier Distribution",
                     color_discrete_sequence=VASTRAA_COLORS, hole=0.4)
        fig.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        inc_order = ["Below ₹3 Lakhs", "₹3 - ₹6 Lakhs", "₹6 - ₹10 Lakhs",
                     "₹10 - ₹15 Lakhs", "₹15 - ₹25 Lakhs", "Above ₹25 Lakhs"]
        inc_counts = fdf["Q3_Annual_Income"].value_counts().reindex(inc_order).fillna(0)
        fig = px.bar(x=inc_counts.index, y=inc_counts.values, title="Income Distribution",
                     color_discrete_sequence=["#D4A574"],
                     labels={"x": "Income Bracket", "y": "Count"})
        fig.update_layout(height=350, margin=dict(t=40, b=20), xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Product Interest & Purchase Intent
    col1, col2 = st.columns(2)
    with col1:
        prod_cols = [c for c in fdf.columns if c.startswith("Q6_")]
        prod_labels = ["Sarees", "Kurtas & Dupattas", "Stoles & Scarves", "Home Textiles", "Men's Wear"]
        prod_vals = [fdf[c].sum() for c in prod_cols]
        fig = px.bar(x=prod_labels, y=prod_vals, title="Product Category Interest",
                     color=prod_labels, color_discrete_sequence=VASTRAA_COLORS,
                     labels={"x": "Category", "y": "Interested Respondents"})
        fig.update_layout(height=370, margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        intent_order = ["Highly Likely", "Likely", "Neutral", "Unlikely", "Highly Unlikely"]
        intent_counts = fdf["Q30_Purchase_Intent"].value_counts().reindex(intent_order).fillna(0)
        colors_intent = ["#2E86AB", "#3C6E71", "#D4A574", "#C0392B", "#1B2A4A"]
        fig = px.bar(x=intent_counts.index, y=intent_counts.values,
                     title="Purchase Intent Distribution (Target Variable)",
                     color=intent_counts.index, color_discrete_sequence=colors_intent,
                     labels={"x": "Intent", "y": "Count"})
        fig.update_layout(height=370, margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Row 4: Textile Styles
    style_cols = [c for c in fdf.columns if c.startswith("Q7_")]
    style_labels = [c.replace("Q7_", "").replace("_", " ") for c in style_cols]
    style_vals = [fdf[c].sum() for c in style_cols]
    style_df = pd.DataFrame({"Style": style_labels, "Interest": style_vals}).sort_values("Interest", ascending=True)
    fig = px.bar(style_df, x="Interest", y="Style", orientation="h",
                 title="Textile Style Preferences (Top Picks)", color_discrete_sequence=["#C0392B"])
    fig.update_layout(height=420, margin=dict(t=40, b=20, l=10))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 2: EDA DEEP DIVE
# ============================================================
elif page == "2. EDA Deep Dive":
    st.markdown('<p class="main-header">🔍 Exploratory Data Analysis — Deep Dive</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Descriptive + Diagnostic Analytics — Patterns, correlations, and the "why" behind the data</p>', unsafe_allow_html=True)

    fdf = filtered_df

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Correlations", "📊 Bivariate Analysis", "🎨 Textile & Fabric", "📅 Behavioral Patterns"])

    with tab1:
        st.subheader("Correlation Heatmap — Numerical Features")
        num_cols = ["Q2_Age_Numeric", "Q3_Income_Midpoint_Lakhs", "Q5_City_Tier_Code",
                    "Q9_Budget_Midpoint_INR", "Q13_Frequency_Code", "Q15_Annual_Spend_Midpoint_INR",
                    "Q16_Gift_Budget_Midpoint_INR", "Q21_Social_Media_Code", "Q22_Online_Trust_Code",
                    "Q23_Sustainability_Code", "Q26_Review_Code", "Derived_Digital_Readiness_Score",
                    "Derived_Total_Products_Interested", "Derived_Total_Addons",
                    "Q30_Intent_Code"]
        short_labels = ["Age", "Income", "City Tier", "Budget/Purchase", "Frequency",
                        "Annual Spend", "Gift Budget", "Social Media", "Online Trust",
                        "Sustainability", "Review Imp.", "Digital Readiness",
                        "Products Interested", "Add-ons", "Purchase Intent"]
        corr_matrix = fdf[num_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values, x=short_labels, y=short_labels,
            colorscale="RdBu_r", zmid=0, text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}", textfont={"size": 9},
            colorbar=dict(title="Correlation")
        ))
        fig.update_layout(height=620, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.info("**Key Diagnostic Insights:** Online Trust and Digital Readiness are the strongest predictors of Purchase Intent. Income correlates strongly with Budget and Annual Spend. Social Media usage drives Online Trust — younger, digitally active consumers are Vastraa's sweet spot.")

    with tab2:
        st.subheader("Bivariate Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(fdf, x="Q2_Age_Group", y="Q9_Budget_Midpoint_INR",
                         color="Q2_Age_Group", title="Budget per Purchase by Age Group",
                         color_discrete_sequence=VASTRAA_COLORS,
                         category_orders={"Q2_Age_Group": ["18-24","25-34","35-44","45-54","55+"]})
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(fdf, x="Q5_City_Tier", y="Q15_Annual_Spend_Midpoint_INR",
                         color="Q5_City_Tier", title="Annual Textile Spend by City Tier",
                         color_discrete_sequence=VASTRAA_COLORS)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            ct_intent = fdf.groupby(["Q5_City_Tier", "Q30_Purchase_Intent"]).size().reset_index(name="Count")
            fig = px.bar(ct_intent, x="Q5_City_Tier", y="Count", color="Q30_Purchase_Intent",
                         title="Purchase Intent by City Tier", barmode="group",
                         color_discrete_sequence=VASTRAA_COLORS,
                         category_orders={"Q30_Purchase_Intent": ["Highly Likely","Likely","Neutral","Unlikely","Highly Unlikely"]})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            age_trust = fdf.groupby(["Q2_Age_Group", "Q22_Online_Purchase_Comfort"]).size().reset_index(name="Count")
            fig = px.bar(age_trust, x="Q2_Age_Group", y="Count", color="Q22_Online_Purchase_Comfort",
                         title="Online Trust Level by Age Group", barmode="stack",
                         color_discrete_sequence=VASTRAA_COLORS,
                         category_orders={"Q2_Age_Group": ["18-24","25-34","35-44","45-54","55+"]})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Income vs Sustainability
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(fdf, x="Q23_Sustainability_Premium", color="Q3_Annual_Income",
                               title="Sustainability Willingness by Income",
                               barmode="group", color_discrete_sequence=VASTRAA_COLORS)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            occ_spend = fdf.groupby("Q4_Occupation")["Q15_Annual_Spend_Midpoint_INR"].mean().sort_values(ascending=True)
            fig = px.bar(x=occ_spend.values, y=occ_spend.index, orientation="h",
                         title="Avg Annual Spend by Occupation", color_discrete_sequence=["#D4A574"],
                         labels={"x": "Avg Annual Spend (₹)", "y": "Occupation"})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Textile Style & Fabric Preference Analysis")
        col1, col2 = st.columns(2)
        with col1:
            style_cols = [c for c in fdf.columns if c.startswith("Q7_")]
            style_labels_map = {c: c.replace("Q7_", "").replace("_", " ") for c in style_cols}
            age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
            style_age_data = []
            for ag in age_groups:
                subset = fdf[fdf["Q2_Age_Group"] == ag]
                for sc in style_cols:
                    style_age_data.append({"Age": ag, "Style": style_labels_map[sc],
                                           "Percentage": subset[sc].mean() * 100})
            style_age_df = pd.DataFrame(style_age_data)
            fig = px.bar(style_age_df, x="Style", y="Percentage", color="Age",
                         barmode="group", title="Textile Style Preference by Age Group",
                         color_discrete_sequence=VASTRAA_COLORS)
            fig.update_layout(height=450, xaxis_tickangle=-40)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fab_cols = [c for c in fdf.columns if c.startswith("Q8_")]
            fab_labels = [c.replace("Q8_", "").replace("_", "/") for c in fab_cols]
            fab_vals = [fdf[c].mean() * 100 for c in fab_cols]
            fab_df = pd.DataFrame({"Fabric": fab_labels, "Preference %": fab_vals}).sort_values("Preference %", ascending=True)
            fig = px.bar(fab_df, x="Preference %", y="Fabric", orientation="h",
                         title="Fabric Type Preferences (%)", color_discrete_sequence=["#C0392B"])
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

        # Style by Gender
        style_gender_data = []
        for g in fdf["Q1_Gender"].unique():
            subset = fdf[fdf["Q1_Gender"] == g]
            for sc in style_cols:
                style_gender_data.append({"Gender": g, "Style": style_labels_map[sc],
                                          "Percentage": subset[sc].mean() * 100})
        style_gender_df = pd.DataFrame(style_gender_data)
        fig = px.bar(style_gender_df, x="Style", y="Percentage", color="Gender",
                     barmode="group", title="Textile Style Preference by Gender",
                     color_discrete_sequence=VASTRAA_COLORS[:3])
        fig.update_layout(height=400, xaxis_tickangle=-40)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Shopping Behavior & Occasion Patterns")
        col1, col2 = st.columns(2)
        with col1:
            occ_cols = [c for c in fdf.columns if c.startswith("Q11_")]
            occ_labels = [c.replace("Q11_", "").replace("_", " ").replace("and", "&") for c in occ_cols]
            occ_vals = [fdf[c].sum() for c in occ_cols]
            occ_df = pd.DataFrame({"Occasion": occ_labels, "Count": occ_vals}).sort_values("Count", ascending=True)
            fig = px.bar(occ_df, x="Count", y="Occasion", orientation="h",
                         title="Purchase Occasion Triggers", color_discrete_sequence=["#2E86AB"])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(fdf, names="Q12_Peak_Shopping_Season", title="Peak Shopping Season",
                         color_discrete_sequence=VASTRAA_COLORS, hole=0.35)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(fdf, names="Q13_Purchase_Frequency", title="Purchase Frequency Distribution",
                         color_discrete_sequence=VASTRAA_COLORS, hole=0.35)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            ch_cols = [c for c in fdf.columns if c.startswith("Q17_")]
            ch_labels = [c.replace("Q17_", "").replace("_", " ") for c in ch_cols]
            ch_vals = [fdf[c].sum() for c in ch_cols]
            ch_df = pd.DataFrame({"Channel": ch_labels, "Count": ch_vals}).sort_values("Count", ascending=True)
            fig = px.bar(ch_df, x="Count", y="Channel", orientation="h",
                         title="Preferred Shopping Channels", color_discrete_sequence=["#A23B72"])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 3: CUSTOMER SEGMENTATION (CLUSTERING)
# ============================================================
elif page == "3. Customer Segmentation":
    st.markdown('<p class="main-header">👥 Customer Segmentation — Know Your Tribes</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Prescriptive Analytics — Who are my distinct customer segments and how should I treat each?</p>', unsafe_allow_html=True)

    cluster_features = ["Q2_Age_Numeric", "Q3_Income_Midpoint_Lakhs", "Q5_City_Tier_Code",
                        "Q9_Budget_Midpoint_INR", "Q13_Frequency_Code", "Q15_Annual_Spend_Midpoint_INR",
                        "Q16_Gift_Budget_Midpoint_INR", "Q21_Social_Media_Code", "Q22_Online_Trust_Code",
                        "Q23_Sustainability_Code", "Derived_Digital_Readiness_Score",
                        "Derived_Total_Products_Interested", "Derived_Total_Addons"]

    X_cluster = df[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    tab1, tab2, tab3 = st.tabs(["🔢 Optimal K Selection", "📊 Cluster Profiles", "🌳 Hierarchical Clustering"])

    with tab1:
        st.subheader("Finding Optimal Number of Clusters")
        col1, col2 = st.columns(2)
        with col1:
            inertias = []
            sil_scores = []
            K_range = range(2, 11)
            for k in K_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_scaled)
                inertias.append(km.inertia_)
                sil_scores.append(silhouette_score(X_scaled, km.labels_))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers',
                                     name='Inertia (WCSS)', line=dict(color='#C0392B', width=2),
                                     marker=dict(size=8)))
            fig.update_layout(title="Elbow Method (WCSS)", xaxis_title="Number of Clusters (K)",
                              yaxis_title="Inertia", height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(K_range), y=sil_scores, mode='lines+markers',
                                     name='Silhouette Score', line=dict(color='#2E86AB', width=2),
                                     marker=dict(size=8)))
            best_k = list(K_range)[np.argmax(sil_scores)]
            fig.update_layout(title=f"Silhouette Score (Best K = {best_k})",
                              xaxis_title="Number of Clusters (K)",
                              yaxis_title="Silhouette Score", height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.success(f"**Optimal K = {best_k}** based on highest Silhouette Score ({max(sil_scores):.3f})")

    with tab2:
        n_clusters = st.slider("Select number of clusters", 2, 8, best_k)
        km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["Cluster"] = km_final.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, df["Cluster"])
        st.info(f"**Silhouette Score for K={n_clusters}: {sil:.3f}**")

        # Cluster sizes
        cluster_sizes = df["Cluster"].value_counts().sort_index()
        fig = px.bar(x=[f"Cluster {i}" for i in cluster_sizes.index], y=cluster_sizes.values,
                     title="Cluster Sizes", color=[f"Cluster {i}" for i in cluster_sizes.index],
                     color_discrete_sequence=VASTRAA_COLORS, labels={"x": "Cluster", "y": "Count"})
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Cluster profiles — Radar chart
        st.subheader("Cluster Profiles — Radar Comparison")
        profile_cols = ["Q3_Income_Midpoint_Lakhs", "Q9_Budget_Midpoint_INR",
                        "Q15_Annual_Spend_Midpoint_INR", "Q21_Social_Media_Code",
                        "Q22_Online_Trust_Code", "Q23_Sustainability_Code",
                        "Derived_Digital_Readiness_Score", "Q13_Frequency_Code"]
        profile_labels = ["Income", "Budget/Purchase", "Annual Spend", "Social Media",
                          "Online Trust", "Sustainability", "Digital Readiness", "Frequency"]

        cluster_profiles = df.groupby("Cluster")[profile_cols].mean()
        # Normalize for radar
        profile_norm = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min() + 1e-9)

        fig = go.Figure()
        for idx in range(n_clusters):
            vals = profile_norm.loc[idx].tolist()
            vals.append(vals[0])
            fig.add_trace(go.Scatterpolar(r=vals, theta=profile_labels + [profile_labels[0]],
                                           fill='toself', name=f'Cluster {idx}',
                                           line=dict(color=VASTRAA_COLORS[idx % len(VASTRAA_COLORS)])))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                          title="Cluster Radar Profiles (Normalized)", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Detailed stats
        st.subheader("Cluster Statistics")
        display_cols = ["Q2_Age_Numeric", "Q3_Income_Midpoint_Lakhs", "Q9_Budget_Midpoint_INR",
                        "Q13_Frequency_Code", "Q15_Annual_Spend_Midpoint_INR",
                        "Q21_Social_Media_Code", "Q22_Online_Trust_Code", "Q23_Sustainability_Code",
                        "Q30_Purchase_Intent_Binary"]
        display_labels = ["Avg Age", "Avg Income (L)", "Avg Budget (₹)", "Avg Frequency",
                          "Avg Annual Spend (₹)", "Social Media", "Online Trust",
                          "Sustainability", "Purchase Intent %"]
        stats = df.groupby("Cluster")[display_cols].mean().round(2)
        stats.columns = display_labels
        stats["Purchase Intent %"] = (stats["Purchase Intent %"] * 100).round(1)
        stats["Avg Budget (₹)"] = stats["Avg Budget (₹)"].apply(lambda x: f"₹{x:,.0f}")
        stats["Avg Annual Spend (₹)"] = stats["Avg Annual Spend (₹)"].apply(lambda x: f"₹{x:,.0f}")
        stats.index = [f"Cluster {i}" for i in stats.index]
        st.dataframe(stats, use_container_width=True)

    with tab3:
        st.subheader("Hierarchical Clustering — Dendrogram")
        sample_size = min(300, len(X_scaled))
        np.random.seed(42)
        sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[sample_idx]

        linked = linkage(X_sample, method='ward')
        fig_dend, ax = plt.subplots(figsize=(14, 5))
        dendrogram(linked, truncate_mode='lastp', p=30, ax=ax,
                   color_threshold=linked[-best_k+1, 2],
                   above_threshold_color='#1B2A4A')
        ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=14)
        ax.set_xlabel("Data Points")
        ax.set_ylabel("Distance")
        st.pyplot(fig_dend)
        plt.close()


# ============================================================
# PAGE 4: PURCHASE PREDICTION (CLASSIFICATION)
# ============================================================
elif page == "4. Purchase Prediction":
    st.markdown('<p class="main-header">🎯 Purchase Prediction — Will They Buy?</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predictive Analytics — Classification with 7 algorithms, full performance comparison</p>', unsafe_allow_html=True)

    feature_cols = get_ml_features(df)
    X = df[feature_cols].copy()
    y = df["Q30_Purchase_Intent_Binary"].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler_cls = StandardScaler()
    X_train_sc = scaler_cls.fit_transform(X_train)
    X_test_sc = scaler_cls.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    results = []
    roc_data = {}

    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_sc) if hasattr(model, 'decision_function') else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0

        results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                        "Recall": rec, "F1-Score": f1, "ROC-AUC": auc})

        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_data[name] = (fpr, tpr, auc)

    results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "📈 ROC Curves", "🔍 Confusion Matrices", "⭐ Feature Importance"])

    with tab1:
        st.subheader("Algorithm Performance Comparison")
        st.dataframe(results_df.style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "F1-Score": "{:.4f}", "ROC-AUC": "{:.4f}"
        }).highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                        color="#d4edda"), use_container_width=True)

        # Bar comparison
        metrics_melt = results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                                        var_name="Metric", value_name="Score")
        fig = px.bar(metrics_melt, x="Model", y="Score", color="Metric", barmode="group",
                     title="Model Performance Comparison", color_discrete_sequence=VASTRAA_COLORS)
        fig.update_layout(height=450, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        best_model_name = results_df.iloc[0]["Model"]
        best_f1 = results_df.iloc[0]["F1-Score"]
        st.success(f"**Best Model: {best_model_name}** with F1-Score = {best_f1:.4f}")

    with tab2:
        st.subheader("ROC Curves — All Models")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                  line=dict(dash='dash', color='gray'), name='Random (AUC=0.5)'))
        for i, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                      name=f'{name} (AUC={auc:.3f})',
                                      line=dict(color=VASTRAA_COLORS[i % len(VASTRAA_COLORS)], width=2)))
        fig.update_layout(title="ROC Curves Comparison", xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate", height=500,
                          legend=dict(x=0.55, y=0.05))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Confusion Matrices")
        selected_model = st.selectbox("Select model", list(models.keys()), index=0)
        model = models[selected_model]
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        cm = confusion_matrix(y_test, y_pred)

        fig = go.Figure(data=go.Heatmap(
            z=cm, x=["Not Likely (0)", "Likely (1)"], y=["Not Likely (0)", "Likely (1)"],
            colorscale="Blues", text=cm, texttemplate="%{text}", textfont={"size": 18},
            showscale=True
        ))
        fig.update_layout(title=f"Confusion Matrix — {selected_model}",
                          xaxis_title="Predicted", yaxis_title="Actual", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.text(f"Classification Report — {selected_model}")
        report = classification_report(y_test, y_pred, target_names=["Not Likely", "Likely"])
        st.code(report)

    with tab4:
        st.subheader("Feature Importance — What Drives Purchase?")
        rf_model = models["Random Forest"]
        rf_model.fit(X_train_sc, y_train)
        importances = rf_model.feature_importances_
        feat_imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values("Importance", ascending=True).tail(20)

        fig = px.bar(feat_imp_df, x="Importance", y="Feature", orientation="h",
                     title="Top 20 Features — Random Forest Importance",
                     color="Importance", color_continuous_scale="OrRd")
        fig.update_layout(height=600, margin=dict(l=10))
        st.plotly_chart(fig, use_container_width=True)

        # Gradient Boosting importance
        gb_model = models["Gradient Boosting"]
        gb_model.fit(X_train_sc, y_train)
        gb_imp = pd.DataFrame({"Feature": feature_cols, "Importance": gb_model.feature_importances_}).sort_values("Importance", ascending=True).tail(20)
        fig2 = px.bar(gb_imp, x="Importance", y="Feature", orientation="h",
                      title="Top 20 Features — Gradient Boosting Importance",
                      color="Importance", color_continuous_scale="Viridis")
        fig2.update_layout(height=600, margin=dict(l=10))
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# PAGE 5: SPEND PREDICTION (REGRESSION)
# ============================================================
elif page == "5. Spend Prediction":
    st.markdown('<p class="main-header">💰 Spend Prediction — How Much Will They Spend?</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predictive Analytics — Linear, Ridge & Lasso Regression on 3 spending targets</p>', unsafe_allow_html=True)

    reg_features = ["Q2_Age_Numeric", "Q3_Income_Midpoint_Lakhs", "Q5_City_Tier_Code",
                    "Q13_Frequency_Code", "Q21_Social_Media_Code", "Q22_Online_Trust_Code",
                    "Q23_Sustainability_Code", "Derived_Digital_Readiness_Score",
                    "Derived_Total_Products_Interested", "Derived_Total_Addons",
                    "Derived_Is_Gifter", "Q26_Review_Code", "Q30_Intent_Code"]

    targets = {
        "Annual Textile Spend (₹)": "Q15_Annual_Spend_Midpoint_INR",
        "Budget per Purchase (₹)": "Q9_Budget_Midpoint_INR",
        "Annual Gifting Budget (₹)": "Q16_Gift_Budget_Midpoint_INR"
    }

    target_choice = st.selectbox("Select Target Variable", list(targets.keys()))
    target_col = targets[target_choice]

    X_reg = df[reg_features].copy()
    y_reg = df[target_col].copy()

    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    scaler_reg = StandardScaler()
    X_tr_sc = scaler_reg.fit_transform(X_tr)
    X_te_sc = scaler_reg.transform(X_te)

    reg_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression (α=1.0)": Ridge(alpha=1.0),
        "Lasso Regression (α=1.0)": Lasso(alpha=1.0)
    }

    reg_results = []
    predictions = {}

    for name, model in reg_models.items():
        model.fit(X_tr_sc, y_tr)
        y_pred = model.predict(X_te_sc)
        predictions[name] = y_pred

        r2 = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        reg_results.append({"Model": name, "R² Score": r2, "MAE (₹)": mae, "RMSE (₹)": rmse})

    reg_results_df = pd.DataFrame(reg_results)

    tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "📈 Actual vs Predicted", "🔍 Coefficients"])

    with tab1:
        st.subheader(f"Regression Performance — {target_choice}")
        st.dataframe(reg_results_df.style.format({
            "R² Score": "{:.4f}", "MAE (₹)": "₹{:,.0f}", "RMSE (₹)": "₹{:,.0f}"
        }).highlight_max(subset=["R² Score"], color="#d4edda").highlight_min(
            subset=["MAE (₹)", "RMSE (₹)"], color="#d4edda"), use_container_width=True)

        metrics_m = reg_results_df.melt(id_vars="Model", value_vars=["R² Score"],
                                         var_name="Metric", value_name="Score")
        fig = px.bar(reg_results_df, x="Model", y="R² Score", color="Model",
                     title="R² Score Comparison", color_discrete_sequence=VASTRAA_COLORS[:3])
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Actual vs Predicted Scatter Plots")
        for name, y_pred in predictions.items():
            fig = px.scatter(x=y_te.values, y=y_pred, title=f"{name} — Actual vs Predicted",
                             labels={"x": f"Actual {target_choice}", "y": f"Predicted {target_choice}"},
                             opacity=0.5, color_discrete_sequence=["#C0392B"])
            max_val = max(y_te.max(), y_pred.max())
            fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                      line=dict(dash='dash', color='gray'), name='Perfect Prediction'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Feature Coefficients — What Drives Spending?")
        for name, model in reg_models.items():
            coefs = model.coef_
            coef_df = pd.DataFrame({"Feature": reg_features, "Coefficient": coefs}).sort_values("Coefficient")
            fig = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                         title=f"Coefficients — {name}",
                         color="Coefficient", color_continuous_scale="RdBu_r")
            fig.update_layout(height=450, margin=dict(l=10))
            st.plotly_chart(fig, use_container_width=True)

        st.info("**Lasso Insight:** Features with coefficients shrunk to zero by Lasso are not meaningful predictors of spending. This automatic feature selection helps Vastraa focus resources on what actually drives revenue.")


# ============================================================
# PAGE 6: MARKET BASKET ANALYSIS (ASSOCIATION RULES)
# ============================================================
elif page == "6. Market Basket Analysis":
    st.markdown('<p class="main-header">🛒 Market Basket Analysis — What Goes Together?</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Prescriptive Analytics — Apriori & FP-Growth for bundle strategy, cross-sell, and product recommendations</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎯 Product × Style × Fabric", "🎁 Occasion × Add-ons × Gifts", "📊 Network Visualization"])

    with tab1:
        st.subheader("Association Rules: Product Categories × Textile Styles × Fabric Types")
        prod_cols = [c for c in df.columns if c.startswith("Q6_")]
        style_cols = [c for c in df.columns if c.startswith("Q7_")]
        fab_cols = [c for c in df.columns if c.startswith("Q8_")]

        basket1 = df[prod_cols + style_cols + fab_cols].copy()
        basket1.columns = [c.replace("Q6_", "P:").replace("Q7_", "S:").replace("Q8_", "F:").replace("_", " ") for c in basket1.columns]

        col1, col2 = st.columns(2)
        with col1:
            min_support = st.slider("Minimum Support", 0.02, 0.30, 0.05, 0.01, key="s1")
        with col2:
            min_confidence = st.slider("Minimum Confidence", 0.1, 0.9, 0.3, 0.05, key="c1")

        basket1_bool = basket1.astype(bool)
        freq_items1 = apriori(basket1_bool, min_support=min_support, use_colnames=True)

        if len(freq_items1) > 0:
            rules1 = association_rules(freq_items1, metric="confidence", min_threshold=min_confidence)
            if len(rules1) > 0:
                rules1 = rules1[rules1["lift"] > 1.0].sort_values("lift", ascending=False)
                rules1["antecedents"] = rules1["antecedents"].apply(lambda x: ", ".join(list(x)))
                rules1["consequents"] = rules1["consequents"].apply(lambda x: ", ".join(list(x)))

                display_rules = rules1[["antecedents", "consequents", "support", "confidence", "lift"]].head(25)
                st.dataframe(display_rules.style.format({
                    "support": "{:.3f}", "confidence": "{:.3f}", "lift": "{:.2f}"
                }).background_gradient(subset=["lift"], cmap="YlOrRd"), use_container_width=True)

                st.metric("Total Rules Found", len(rules1))
                st.metric("Avg Lift", f"{rules1['lift'].mean():.2f}")

                # Scatter: Support vs Confidence colored by Lift
                fig = px.scatter(rules1.head(80), x="support", y="confidence", size="lift",
                                 color="lift", hover_data=["antecedents", "consequents"],
                                 title="Rules: Support vs Confidence (size & color = Lift)",
                                 color_continuous_scale="OrRd")
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No rules found with current thresholds. Try lowering support or confidence.")
        else:
            st.warning("No frequent itemsets found. Try lowering minimum support.")

    with tab2:
        st.subheader("Association Rules: Occasions × Add-ons × Gift Categories")
        occ_cols = [c for c in df.columns if c.startswith("Q11_")]
        addon_cols = [c for c in df.columns if c.startswith("Q10_")]
        gift_cols = [c for c in df.columns if c.startswith("Q14_")]

        basket2 = df[occ_cols + addon_cols + gift_cols].copy()
        basket2.columns = [c.replace("Q11_", "Occ:").replace("Q10_", "Add:").replace("Q14_", "Gift:").replace("_", " ") for c in basket2.columns]

        col1, col2 = st.columns(2)
        with col1:
            min_support2 = st.slider("Minimum Support", 0.02, 0.30, 0.05, 0.01, key="s2")
        with col2:
            min_confidence2 = st.slider("Minimum Confidence", 0.1, 0.9, 0.3, 0.05, key="c2")

        basket2_bool = basket2.astype(bool)
        freq_items2 = apriori(basket2_bool, min_support=min_support2, use_colnames=True)

        if len(freq_items2) > 0:
            rules2 = association_rules(freq_items2, metric="confidence", min_threshold=min_confidence2)
            if len(rules2) > 0:
                rules2 = rules2[rules2["lift"] > 1.0].sort_values("lift", ascending=False)
                rules2["antecedents"] = rules2["antecedents"].apply(lambda x: ", ".join(list(x)))
                rules2["consequents"] = rules2["consequents"].apply(lambda x: ", ".join(list(x)))

                display_rules2 = rules2[["antecedents", "consequents", "support", "confidence", "lift"]].head(25)
                st.dataframe(display_rules2.style.format({
                    "support": "{:.3f}", "confidence": "{:.3f}", "lift": "{:.2f}"
                }).background_gradient(subset=["lift"], cmap="YlOrRd"), use_container_width=True)

                fig = px.scatter(rules2.head(80), x="support", y="confidence", size="lift",
                                 color="lift", hover_data=["antecedents", "consequents"],
                                 title="Occasion-Addon Rules: Support vs Confidence",
                                 color_continuous_scale="Viridis")
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No rules found. Try lowering thresholds.")
        else:
            st.warning("No frequent itemsets found. Try lowering support.")

    with tab3:
        st.subheader("Top Association Rules — Network View")
        all_basket = df[[c for c in df.columns if c.startswith(("Q6_","Q7_","Q8_","Q10_"))]].copy()
        all_basket.columns = [c.replace("Q6_","P:").replace("Q7_","S:").replace("Q8_","F:").replace("Q10_","A:").replace("_"," ") for c in all_basket.columns]

        freq_all = apriori(all_basket.astype(bool), min_support=0.05, use_colnames=True)
        if len(freq_all) > 0:
            rules_all = association_rules(freq_all, metric="confidence", min_threshold=0.35)
            rules_all = rules_all[rules_all["lift"] > 1.0].sort_values("lift", ascending=False).head(30)

            if len(rules_all) > 0:
                nodes = set()
                edges = []
                for _, row in rules_all.iterrows():
                    for a in row["antecedents"]:
                        for c in row["consequents"]:
                            nodes.add(a)
                            nodes.add(c)
                            edges.append((a, c, row["lift"], row["confidence"]))

                node_list = list(nodes)
                np.random.seed(42)
                node_x = {n: np.random.uniform(-1, 1) for n in node_list}
                node_y = {n: np.random.uniform(-1, 1) for n in node_list}

                edge_traces = []
                for a, c, lift, conf in edges:
                    edge_traces.append(go.Scatter(
                        x=[node_x[a], node_x[c]], y=[node_y[a], node_y[c]],
                        mode='lines', line=dict(width=max(1, lift), color=f'rgba(192,57,43,{min(conf, 0.8)})'),
                        hoverinfo='text', text=f'{a} → {c} (Lift: {lift:.2f}, Conf: {conf:.2f})',
                        showlegend=False
                    ))

                node_trace = go.Scatter(
                    x=[node_x[n] for n in node_list], y=[node_y[n] for n in node_list],
                    mode='markers+text', text=node_list, textposition="top center",
                    textfont=dict(size=9),
                    marker=dict(size=15, color=VASTRAA_COLORS[:len(node_list)], line=dict(width=1, color='white')),
                    showlegend=False
                )

                fig = go.Figure(data=edge_traces + [node_trace])
                fig.update_layout(title="Association Network (line thickness = Lift, opacity = Confidence)",
                                  height=550, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough rules for network visualization. Try adjusting thresholds above.")
        else:
            st.info("No frequent itemsets found for network.")


# ============================================================
# PAGE 7: PRESCRIPTIVE STRATEGY
# ============================================================
elif page == "7. Prescriptive Strategy":
    st.markdown('<p class="main-header">🎯 Prescriptive Strategy Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Prescriptive Analytics — Actionable recommendations for each customer segment</p>', unsafe_allow_html=True)

    # Run clustering
    cluster_features = ["Q2_Age_Numeric", "Q3_Income_Midpoint_Lakhs", "Q5_City_Tier_Code",
                        "Q9_Budget_Midpoint_INR", "Q13_Frequency_Code", "Q15_Annual_Spend_Midpoint_INR",
                        "Q16_Gift_Budget_Midpoint_INR", "Q21_Social_Media_Code", "Q22_Online_Trust_Code",
                        "Q23_Sustainability_Code", "Derived_Digital_Readiness_Score",
                        "Derived_Total_Products_Interested", "Derived_Total_Addons"]

    X_cl = df[cluster_features].copy()
    scaler_cl = StandardScaler()
    X_cl_sc = scaler_cl.fit_transform(X_cl)

    # Find best K
    sil_scores = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        sil_scores.append(silhouette_score(X_cl_sc, km.fit_predict(X_cl_sc)))
    best_k = range(2, 9)[np.argmax(sil_scores)]

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(X_cl_sc)

    for cl in range(best_k):
        cluster_data = df[df["Cluster"] == cl]
        n = len(cluster_data)
        pct = n / len(df) * 100

        avg_income = cluster_data["Q3_Income_Midpoint_Lakhs"].mean()
        avg_spend = cluster_data["Q15_Annual_Spend_Midpoint_INR"].mean()
        avg_budget = cluster_data["Q9_Budget_Midpoint_INR"].mean()
        avg_trust = cluster_data["Q22_Online_Trust_Code"].mean()
        avg_sm = cluster_data["Q21_Social_Media_Code"].mean()
        avg_sust = cluster_data["Q23_Sustainability_Code"].mean()
        avg_freq = cluster_data["Q13_Frequency_Code"].mean()
        intent_rate = cluster_data["Q30_Purchase_Intent_Binary"].mean() * 100
        avg_age = cluster_data["Q2_Age_Numeric"].mean()

        # Auto name
        if avg_income > 12 and avg_trust > 3.5 and avg_sust > 3.5:
            name = "💎 Premium Heritage Seekers"
        elif avg_sm > 3.5 and avg_age < 32 and avg_trust > 3:
            name = "📱 Digital-First Trendsetters"
        elif avg_freq > 2.5 and avg_spend > 30000:
            name = "🛍️ High-Value Regulars"
        elif cluster_data["Derived_Is_Gifter"].mean() > 0.7:
            name = "🎁 Gifting Champions"
        elif avg_budget < 3000 and avg_income < 6:
            name = "💡 Budget-Conscious Explorers"
        elif avg_trust < 2.5:
            name = "🏪 Offline-First Traditionalists"
        else:
            name = f"👥 Segment {cl}"

        # Strategy card
        with st.expander(f"**Cluster {cl}: {name}** — {n} customers ({pct:.1f}%) | Intent: {intent_rate:.0f}%", expanded=(cl == 0)):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Avg Income", f"₹{avg_income:.1f}L")
            c2.metric("Avg Annual Spend", f"₹{avg_spend:,.0f}")
            c3.metric("Online Trust", f"{avg_trust:.1f}/5")
            c4.metric("Social Media", f"{avg_sm:.1f}/5")
            c5.metric("Purchase Intent", f"{intent_rate:.0f}%")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📦 Recommended Products:**")
                prod_cols_list = [c for c in df.columns if c.startswith("Q6_")]
                prod_names = ["Sarees", "Kurtas & Dupattas", "Stoles & Scarves", "Home Textiles", "Men's Wear"]
                top_prods = [(prod_names[j], cluster_data[prod_cols_list[j]].mean())
                             for j in range(len(prod_cols_list))]
                top_prods.sort(key=lambda x: x[1], reverse=True)
                for pname, pval in top_prods[:3]:
                    st.write(f"  • {pname} ({pval*100:.0f}% interest)")

                st.markdown("**🎨 Top Textile Styles:**")
                style_cols_list = [c for c in df.columns if c.startswith("Q7_")]
                style_names = [c.replace("Q7_", "").replace("_", " ") for c in style_cols_list]
                top_styles = [(style_names[j], cluster_data[style_cols_list[j]].mean())
                              for j in range(len(style_cols_list))]
                top_styles.sort(key=lambda x: x[1], reverse=True)
                for sname, sval in top_styles[:3]:
                    st.write(f"  • {sname} ({sval*100:.0f}%)")

            with col2:
                st.markdown("**💰 Discount Strategy:**")
                top_discount = cluster_data["Q25_Discount_Trigger"].mode().iloc[0] if len(cluster_data) > 0 else "N/A"
                st.write(f"  Preferred: **{top_discount}**")

                st.markdown("**📣 Best Marketing Channel:**")
                ch_cols_list = [c for c in df.columns if c.startswith("Q17_")]
                ch_names = [c.replace("Q17_", "").replace("_", " ") for c in ch_cols_list]
                top_ch = [(ch_names[j], cluster_data[ch_cols_list[j]].mean())
                          for j in range(len(ch_cols_list))]
                top_ch.sort(key=lambda x: x[1], reverse=True)
                st.write(f"  1. {top_ch[0][0]} ({top_ch[0][1]*100:.0f}%)")
                st.write(f"  2. {top_ch[1][0]} ({top_ch[1][1]*100:.0f}%)")

                st.markdown("**🏷️ Price Positioning:**")
                st.write(f"  Avg budget: **₹{avg_budget:,.0f}** per purchase")

            st.markdown("---")


# ============================================================
# PAGE 8: NEW CUSTOMER PREDICTION
# ============================================================
elif page == "8. New Customer Prediction":
    st.markdown('<p class="main-header">🔮 New Customer Prediction Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Score new leads — Single customer form or bulk CSV upload</p>', unsafe_allow_html=True)

    # Train models once
    feature_cols = get_ml_features(df)
    X_full = df[feature_cols].copy()
    y_cls = df["Q30_Purchase_Intent_Binary"].copy()
    y_spend = df["Q15_Annual_Spend_Midpoint_INR"].copy()

    scaler_new = StandardScaler()
    X_full_sc = scaler_new.fit_transform(X_full)

    clf_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf_model.fit(X_full_sc, y_cls)

    reg_features_new = ["Q2_Age_Numeric", "Q3_Income_Midpoint_Lakhs", "Q5_City_Tier_Code",
                        "Q13_Frequency_Code", "Q21_Social_Media_Code", "Q22_Online_Trust_Code",
                        "Q23_Sustainability_Code", "Derived_Digital_Readiness_Score",
                        "Derived_Total_Products_Interested", "Derived_Total_Addons",
                        "Derived_Is_Gifter", "Q26_Review_Code", "Q30_Intent_Code"]

    # Cluster model
    cluster_features = ["Q2_Age_Numeric", "Q3_Income_Midpoint_Lakhs", "Q5_City_Tier_Code",
                        "Q9_Budget_Midpoint_INR", "Q13_Frequency_Code", "Q15_Annual_Spend_Midpoint_INR",
                        "Q16_Gift_Budget_Midpoint_INR", "Q21_Social_Media_Code", "Q22_Online_Trust_Code",
                        "Q23_Sustainability_Code", "Derived_Digital_Readiness_Score",
                        "Derived_Total_Products_Interested", "Derived_Total_Addons"]
    scaler_cluster = StandardScaler()
    X_cl_sc = scaler_cluster.fit_transform(df[cluster_features])
    km_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    km_model.fit(X_cl_sc)

    tab1, tab2 = st.tabs(["👤 Single Customer", "📂 Bulk CSV Upload"])

    with tab1:
        st.subheader("Enter New Customer Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            inp_gender = st.selectbox("Gender", ["Male", "Female", "Non-binary / Prefer not to say"])
            inp_age = st.selectbox("Age Group", ["18-24", "25-34", "35-44", "45-54", "55+"])
            inp_income = st.selectbox("Annual Income", ["Below ₹3 Lakhs", "₹3 - ₹6 Lakhs", "₹6 - ₹10 Lakhs",
                                                        "₹10 - ₹15 Lakhs", "₹15 - ₹25 Lakhs", "Above ₹25 Lakhs"])
            inp_occupation = st.selectbox("Occupation", ["Student", "Salaried Professional",
                                                         "Self-employed / Business Owner", "Homemaker",
                                                         "Freelancer / Gig Worker", "Retired"])
            inp_city = st.selectbox("City Tier", ["Metro", "Tier-1", "Tier-2", "Tier-3 / Small Town / Rural"])

        with col2:
            inp_budget = st.selectbox("Budget per Purchase", ["Below ₹1,000", "₹1,000 - ₹2,500", "₹2,500 - ₹5,000",
                                                              "₹5,000 - ₹10,000", "₹10,000 - ₹20,000", "Above ₹20,000"])
            inp_frequency = st.selectbox("Purchase Frequency", ["1-2 times (Rare)", "3-5 times (Occasional)",
                                                                 "6-10 times (Regular)", "More than 10 times (Frequent)"])
            inp_social = st.selectbox("Social Media Usage", ["Less than 30 min", "30 min - 1 hour",
                                                             "1 - 2 hours", "2 - 4 hours", "More than 4 hours"])
            inp_trust = st.selectbox("Online Purchase Comfort", ["Very Comfortable", "Somewhat Comfortable",
                                                                  "Neutral", "Somewhat Uncomfortable", "Very Uncomfortable"])
            inp_sustainability = st.selectbox("Sustainability Premium", ["Yes, definitely", "Probably yes",
                                                                         "Neutral", "Probably not", "No"])

        with col3:
            inp_products = st.multiselect("Products Interested", ["Sarees", "Kurtas", "Stoles", "Home Textiles", "Men's Wear"], default=["Sarees"])
            inp_addons = st.multiselect("Add-ons", ["Dupatta/Stole", "Jewellery", "Footwear", "Gift Box", "Home Set", "Blouse Piece"], default=["Dupatta/Stole"])
            inp_review = st.selectbox("Review Importance", ["Very Important", "Somewhat Important", "Neutral", "Not Important"])
            inp_household = st.selectbox("Household", ["Single/Living alone", "Newly Married/Couple",
                                                       "Young Family (Children <12)", "Established Family (Children 12+)",
                                                       "Empty Nester", "Joint Family/Multi-generational"])

        if st.button("🔮 Predict Purchase Likelihood", type="primary"):
            # Build feature vector matching training columns
            age_map = {"18-24": 21, "25-34": 30, "35-44": 40, "45-54": 50, "55+": 60}
            income_map = {"Below ₹3 Lakhs": 2, "₹3 - ₹6 Lakhs": 4.5, "₹6 - ₹10 Lakhs": 8,
                          "₹10 - ₹15 Lakhs": 12.5, "₹15 - ₹25 Lakhs": 20, "Above ₹25 Lakhs": 35}
            city_map = {"Metro": 1, "Tier-1": 2, "Tier-2": 3, "Tier-3 / Small Town / Rural": 4}
            budget_map = {"Below ₹1,000": 700, "₹1,000 - ₹2,500": 1750, "₹2,500 - ₹5,000": 3750,
                          "₹5,000 - ₹10,000": 7500, "₹10,000 - ₹20,000": 15000, "Above ₹20,000": 30000}
            freq_map = {"1-2 times (Rare)": 1, "3-5 times (Occasional)": 2,
                        "6-10 times (Regular)": 3, "More than 10 times (Frequent)": 4}
            sm_map = {"Less than 30 min": 1, "30 min - 1 hour": 2, "1 - 2 hours": 3,
                      "2 - 4 hours": 4, "More than 4 hours": 5}
            trust_map = {"Very Comfortable": 5, "Somewhat Comfortable": 4, "Neutral": 3,
                         "Somewhat Uncomfortable": 2, "Very Uncomfortable": 1}
            sust_map = {"Yes, definitely": 5, "Probably yes": 4, "Neutral": 3, "Probably not": 2, "No": 1}
            review_map = {"Very Important": 4, "Somewhat Important": 3, "Neutral": 2, "Not Important": 1}

            # Create a row matching the training feature structure
            new_row = {}
            for col in feature_cols:
                new_row[col] = 0

            # Set numerical features
            new_row["Q2_Age_Numeric"] = age_map[inp_age]
            new_row["Q3_Income_Midpoint_Lakhs"] = income_map[inp_income]
            new_row["Q5_City_Tier_Code"] = city_map[inp_city]
            new_row["Q9_Budget_Midpoint_INR"] = budget_map[inp_budget]
            new_row["Q13_Frequency_Code"] = freq_map[inp_frequency]
            new_row["Q21_Social_Media_Code"] = sm_map[inp_social]
            new_row["Q22_Online_Trust_Code"] = trust_map[inp_trust]
            new_row["Q23_Sustainability_Code"] = sust_map[inp_sustainability]
            new_row["Q26_Review_Code"] = review_map[inp_review]
            new_row["Derived_Digital_Readiness_Score"] = (sm_map[inp_social] + trust_map[inp_trust]) / 2
            new_row["Derived_Total_Products_Interested"] = len(inp_products)
            new_row["Derived_Total_Addons"] = len(inp_addons)
            new_row["Derived_Is_Gifter"] = 1 if "Gift Box" in inp_addons else 0
            new_row["Derived_Estimated_Annual_Value_INR"] = budget_map[inp_budget] * freq_map[inp_frequency]

            # Set estimate for spend and gift budget based on inputs
            est_spend = budget_map[inp_budget] * freq_map[inp_frequency] * 1.2
            new_row["Q15_Annual_Spend_Midpoint_INR"] = est_spend
            new_row["Q16_Gift_Budget_Midpoint_INR"] = est_spend * 0.15 if "Gift Box" in inp_addons else 0
            new_row["Q30_Intent_Code"] = 3  # neutral placeholder

            # Set binary product columns
            prod_map_local = {"Sarees": "Q6_Handloom_Sarees", "Kurtas": "Q6_Block_Print_Kurtas_and_Dupatta_Sets",
                        "Stoles": "Q6_Handloom_Stoles_and_Scarves", "Home Textiles": "Q6_Artisan_Home_Textiles",
                        "Men's Wear": "Q6_Mens_Handloom_Kurtas_and_Nehru_Jackets"}
            for p in inp_products:
                if prod_map_local.get(p) in new_row:
                    new_row[prod_map_local[p]] = 1

            new_df = pd.DataFrame([new_row])[feature_cols]
            new_sc = scaler_new.transform(new_df)

            pred_prob = clf_model.predict_proba(new_sc)[0][1]
            pred_class = "Likely to Purchase ✅" if pred_prob >= 0.5 else "Unlikely to Purchase ❌"

            # Cluster assignment
            cluster_vals = [new_row.get(f, 0) for f in cluster_features]
            cluster_sc = scaler_cluster.transform([cluster_vals])
            cluster_label = km_model.predict(cluster_sc)[0]

            st.markdown("---")
            st.subheader("Prediction Results")
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Purchase Probability", f"{pred_prob*100:.1f}%")
            rc2.metric("Prediction", pred_class)
            rc3.metric("Customer Segment", f"Cluster {cluster_label}")

            if pred_prob >= 0.7:
                st.success("🎯 **High-priority lead!** This customer is a strong fit for Vastraa. Recommend: personalized welcome offer, curated collection based on their style preferences.")
            elif pred_prob >= 0.5:
                st.info("👍 **Moderate-priority lead.** Customer shows interest. Recommend: targeted Instagram/social media ads, first-purchase discount to convert.")
            else:
                st.warning("⚠️ **Lower priority.** Focus on building trust — strong return policy messaging, video reviews, and gradual engagement through content.")

    with tab2:
        st.subheader("Bulk Upload — Score Multiple Customers")
        st.markdown("Upload a CSV file with the **same column structure** as the training dataset. The system will predict purchase likelihood, estimated spend, and cluster for each customer.")

        uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                st.success(f"Uploaded {len(new_data)} records")

                # Check if feature columns exist
                missing_cols = [c for c in feature_cols if c not in new_data.columns]
                if len(missing_cols) > 0:
                    st.error(f"Missing columns: {missing_cols[:10]}... Please ensure your CSV matches the training data structure.")
                else:
                    X_new = new_data[feature_cols].fillna(0)
                    X_new_sc = scaler_new.transform(X_new)

                    # Predictions
                    new_data["Predicted_Purchase_Probability"] = clf_model.predict_proba(X_new_sc)[:, 1]
                    new_data["Predicted_Purchase"] = (new_data["Predicted_Purchase_Probability"] >= 0.5).astype(int)
                    new_data["Predicted_Purchase_Label"] = new_data["Predicted_Purchase"].map({1: "Likely", 0: "Unlikely"})

                    # Cluster
                    cluster_cols_present = [c for c in cluster_features if c in new_data.columns]
                    if len(cluster_cols_present) == len(cluster_features):
                        X_new_cl = scaler_cluster.transform(new_data[cluster_features].fillna(0))
                        new_data["Predicted_Cluster"] = km_model.predict(X_new_cl)

                    # Priority score
                    new_data["Priority_Score"] = (
                        new_data["Predicted_Purchase_Probability"] * 0.6 +
                        (new_data.get("Q15_Annual_Spend_Midpoint_INR", 0) / new_data.get("Q15_Annual_Spend_Midpoint_INR", 1).max()) * 0.4
                    ).round(3)

                    new_data = new_data.sort_values("Priority_Score", ascending=False)

                    # Summary
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Leads", len(new_data))
                    c2.metric("Likely Buyers", f"{new_data['Predicted_Purchase'].sum()} ({new_data['Predicted_Purchase'].mean()*100:.1f}%)")
                    c3.metric("Avg Purchase Probability", f"{new_data['Predicted_Purchase_Probability'].mean()*100:.1f}%")

                    # Show results
                    display_cols_bulk = ["Respondent_ID", "Predicted_Purchase_Label",
                                   "Predicted_Purchase_Probability", "Priority_Score"]
                    if "Predicted_Cluster" in new_data.columns:
                        display_cols_bulk.append("Predicted_Cluster")

                    available_display = [c for c in display_cols_bulk if c in new_data.columns]
                    st.dataframe(new_data[available_display].head(50), use_container_width=True)

                    # Download
                    csv_out = new_data.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Scored Results", csv_out,
                                       "vastraa_scored_leads.csv", "text/csv")

                    # Distribution chart
                    fig = px.histogram(new_data, x="Predicted_Purchase_Probability", nbins=20,
                                       title="Distribution of Purchase Probabilities",
                                       color_discrete_sequence=["#C0392B"])
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

        st.markdown("---")
        st.markdown("**💡 Tip:** You can use the original training dataset format as a template. Download it below:")
        template_csv = df.head(5).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Template CSV (5 sample rows)", template_csv,
                           "vastraa_upload_template.csv", "text/csv")
