import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loader (cached so CSVs load once) ───────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))

    segments = pd.read_csv(os.path.join(base, "data/processed/customer_segments.csv"))
    features = pd.read_csv(os.path.join(base, "data/processed/customer_features.csv"))
    churn    = pd.read_csv(os.path.join(base, "data/processed/customer_churn_scores.csv"))

    # Try loading CLV — may not exist yet if Day 5 skipped
    clv_path = os.path.join(base, "data/processed/customer_clv.csv")
    clv = pd.read_csv(clv_path) if os.path.exists(clv_path) else None

    # Merge into one master DataFrame
    master = segments.merge(
        features[[
            'customer_id', 'recency_days', 'frequency', 'monetary',
            'avg_order_value', 'avg_review_score', 'category_diversity',
            'avg_delivery_days', 'avg_delivery_delta', 'tenure_days',
            'avg_installments', 'freight_to_revenue_ratio',
            'preferred_payment', 'favourite_category',
        ]],
        on='customer_id', how='left', suffixes=('', '_feat')
    )

    master = master.merge(
        churn[[
            'customer_id', 'returned', 'churned',
            'return_probability', 'will_return', 'churn_risk_tier',
        ]],
        on='customer_id', how='left'
    )

    if clv is not None:
        master = master.merge(
            clv[['customer_id', 'clv_12m']],
            on='customer_id', how='left'
        )
    else:
        master['clv_12m'] = master['monetary'] * 0.3  # fallback estimate

    # Derived columns
    master['churn_risk_tier'] = master['churn_risk_tier'].fillna('Unknown')
    master['segment']         = master['segment'].fillna('Unknown')
    master['churn_pct']       = (1 - master['return_probability'].fillna(0.5)) * 100

    return master

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("Filters")
st.sidebar.markdown("---")

# Segment filter
all_segments = sorted(df['segment'].unique())
selected_segments = st.sidebar.multiselect(
    "Customer segment",
    options=all_segments,
    default=all_segments,
)

# Spend tier filter
all_tiers = sorted(df['spend_tier'].dropna().unique())
selected_tiers = st.sidebar.multiselect(
    "Spend tier",
    options=all_tiers,
    default=all_tiers,
)

# Churn risk filter
all_risk = ['Low', 'Medium', 'High', 'Critical']
selected_risk = st.sidebar.multiselect(
    "Churn risk tier",
    options=all_risk,
    default=all_risk,
)

# Monetary range slider
min_mon = float(df['monetary'].min())
max_mon = float(df['monetary'].max())
mon_range = st.sidebar.slider(
    "Total spend range (BRL)",
    min_value=min_mon,
    max_value=max_mon,
    value=(min_mon, max_mon),
    step=10.0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Showing filtered data. "
    f"Total customers: {len(df):,}"
)

# Apply all filters
mask = (
    df['segment'].isin(selected_segments) &
    df['spend_tier'].isin(selected_tiers) &
    df['churn_risk_tier'].isin(selected_risk) &
    df['monetary'].between(mon_range[0], mon_range[1])
)
filtered = df[mask].copy()

# Guard — show warning if filter removes everything
if len(filtered) == 0:
    st.warning("No customers match the current filters. Adjust the sidebar.")
    st.stop()

# ── Page title ────────────────────────────────────────────────
st.title("🛒 E-Commerce Customer Analytics")
st.caption(
    f"Brazilian E-Commerce (Olist) · "
    f"{len(filtered):,} customers shown · "
    f"Snapshot date: 2018-10-18"
)
st.markdown("---")

# ── KPI row ───────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

total_revenue   = filtered['monetary'].sum()
avg_aov         = filtered['avg_order_value'].mean()
avg_churn_risk  = filtered['churn_pct'].mean()
avg_clv         = filtered['clv_12m'].mean()
avg_review      = filtered['avg_review_score'].mean()

k1.metric("Total Revenue (BRL)",   f"{total_revenue:,.0f}")
k2.metric("Avg Order Value",        f"BRL {avg_aov:.2f}")
k3.metric("Avg Churn Risk",         f"{avg_churn_risk:.1f}%")
k4.metric("Avg 12-month CLV",       f"BRL {avg_clv:.2f}")
k5.metric("Avg Review Score",       f"{avg_review:.2f} / 5")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Segment Explorer",
    "⚠️ Churn Risk",
    "💰 CLV Analysis",
    "📋 Business Summary",
])

with tab1:
    st.subheader("Customer Segmentation")
    st.caption("Clusters identified via KMeans on RFM + behavioural features.")

    col1, col2 = st.columns([2, 1])

    with col1:
        # UMAP scatter coloured by segment
        if 'umap_x' in filtered.columns and 'umap_y' in filtered.columns:
            fig_umap = px.scatter(
                filtered.sample(min(len(filtered), 8000), random_state=42),
                x='umap_x',
                y='umap_y',
                color='segment',
                hover_data={
                    'umap_x': False,
                    'umap_y': False,
                    'monetary': ':.2f',
                    'avg_review_score': ':.2f',
                    'churn_risk_tier': True,
                },
                title="UMAP Projection — Customer Segments",
                opacity=0.5,
                height=450,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_umap.update_traces(marker=dict(size=3))
            fig_umap.update_layout(
                legend=dict(orientation='h', yanchor='bottom',
                            y=1.02, xanchor='right', x=1),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_umap, use_container_width=True)
        else:
            st.info("UMAP coordinates not found — rerun Day 3 notebook to generate them.")

    with col2:
        # Segment size donut
        seg_counts = (
            filtered.groupby('segment')
            .agg(customers=('customer_id', 'count'),
                 revenue=('monetary', 'sum'))
            .reset_index()
        )
        fig_donut = px.pie(
            seg_counts,
            names='segment',
            values='customers',
            hole=0.5,
            title="Customers per Segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_donut.update_layout(
            showlegend=True,
            margin=dict(l=0, r=0, t=40, b=0),
            height=220,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # Revenue share donut
        fig_rev = px.pie(
            seg_counts,
            names='segment',
            values='revenue',
            hole=0.5,
            title="Revenue per Segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_rev.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
            height=220,
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("---")

    # Segment profile table
    st.subheader("Segment Profiles")

    profile = (
        filtered.groupby('segment')
        .agg(
            customers          = ('customer_id',            'count'),
            avg_monetary       = ('monetary',               'mean'),
            avg_aov            = ('avg_order_value',        'mean'),
            avg_review         = ('avg_review_score',       'mean'),
            avg_recency        = ('recency_days',           'mean'),
            avg_delivery_delta = ('avg_delivery_delta',     'mean'),
            avg_churn_risk     = ('churn_pct',              'mean'),
            avg_clv            = ('clv_12m',                'mean'),
        )
        .round(2)
        .reset_index()
    )
    profile['revenue_share_%'] = (
        filtered.groupby('segment')['monetary'].sum() /
        filtered['monetary'].sum() * 100
    ).round(1).values

    profile.columns = [
        'Segment', 'Customers', 'Avg Spend (BRL)', 'Avg AOV',
        'Avg Review', 'Avg Recency (days)', 'Avg Delivery Delta',
        'Avg Churn Risk %', 'Avg CLV (BRL)', 'Revenue Share %'
    ]
    st.dataframe(
        profile.sort_values('Avg Spend (BRL)', ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # RFM box plots per segment
    st.subheader("RFM Distribution by Segment")
    rfm_metric = st.selectbox(
        "Select metric",
        ['monetary', 'avg_order_value', 'avg_review_score',
         'recency_days', 'avg_delivery_days'],
        key='rfm_select'
    )

    fig_box = px.box(
        filtered,
        x='segment',
        y=rfm_metric,
        color='segment',
        title=f"{rfm_metric} distribution by segment",
        color_discrete_sequence=px.colors.qualitative.Set2,
        height=400,
    )
    fig_box.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    st.subheader("Churn Risk Analysis")
    st.caption(
        "Model: XGBoost predicting repeat purchase. "
        "Churn risk = 1 − return probability."
    )

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    critical = (filtered['churn_risk_tier'] == 'Critical').sum()
    high     = (filtered['churn_risk_tier'] == 'High').sum()
    med      = (filtered['churn_risk_tier'] == 'Medium').sum()
    low      = (filtered['churn_risk_tier'] == 'Low').sum()

    c1.metric("Critical Risk", f"{critical:,}",
              delta=f"{critical/len(filtered)*100:.1f}% of total",
              delta_color="inverse")
    c2.metric("High Risk",     f"{high:,}",
              delta=f"{high/len(filtered)*100:.1f}% of total",
              delta_color="inverse")
    c3.metric("Medium Risk",   f"{med:,}",
              delta=f"{med/len(filtered)*100:.1f}% of total",
              delta_color="off")
    c4.metric("Low Risk",      f"{low:,}",
              delta=f"{low/len(filtered)*100:.1f}% of total",
              delta_color="normal")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Churn probability histogram
        fig_hist = px.histogram(
            filtered,
            x='return_probability',
            color='churn_risk_tier',
            nbins=50,
            title="Return Probability Distribution",
            labels={'return_probability': 'Return Probability'},
            color_discrete_map={
                'Low':      '#1D9E75',
                'Medium':   '#FAC775',
                'High':     '#F0997B',
                'Critical': '#D85A30',
                'Unknown':  '#B4B2A9',
            },
            height=350,
        )
        fig_hist.update_layout(
            bargap=0.05,
            legend_title="Risk Tier",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Risk tier by segment stacked bar
        risk_seg = (
            filtered.groupby(['segment', 'churn_risk_tier'])
            .size()
            .reset_index(name='count')
        )
        fig_stack = px.bar(
            risk_seg,
            x='segment',
            y='count',
            color='churn_risk_tier',
            title="Churn Risk Tier by Segment",
            color_discrete_map={
                'Low':      '#1D9E75',
                'Medium':   '#FAC775',
                'High':     '#F0997B',
                'Critical': '#D85A30',
                'Unknown':  '#B4B2A9',
            },
            height=350,
        )
        fig_stack.update_layout(
            barmode='stack',
            margin=dict(l=0, r=0, t=40, b=0),
            legend_title="Risk Tier",
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    st.markdown("---")

    # Churn risk vs monetary scatter
    st.subheader("Churn Risk vs Revenue")
    fig_scatter = px.scatter(
        filtered.sample(min(len(filtered), 5000), random_state=42),
        x='monetary',
        y='churn_pct',
        color='churn_risk_tier',
        hover_data={
            'customer_id': True,
            'segment': True,
            'avg_review_score': ':.2f',
            'monetary': ':.2f',
            'churn_pct': ':.1f',
        },
        title="Customer Spend vs Churn Risk %",
        labels={
            'monetary': 'Total Spend (BRL)',
            'churn_pct': 'Churn Risk %',
        },
        color_discrete_map={
            'Low':      '#1D9E75',
            'Medium':   '#FAC775',
            'High':     '#F0997B',
            'Critical': '#D85A30',
            'Unknown':  '#B4B2A9',
        },
        opacity=0.6,
        height=400,
    )
    fig_scatter.update_traces(marker=dict(size=5))
    fig_scatter.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # High-value at-risk customers table
    st.subheader("High-Value Customers at Critical Risk")
    st.caption("Customers with spend > median AND churn risk tier = Critical or High.")

    median_spend = filtered['monetary'].median()
    at_risk_hv = filtered[
        (filtered['monetary'] > median_spend) &
        (filtered['churn_risk_tier'].isin(['Critical', 'High']))
    ].sort_values('monetary', ascending=False)

    display_cols = {
        'customer_id':       'Customer ID',
        'segment':           'Segment',
        'monetary':          'Total Spend (BRL)',
        'avg_order_value':   'AOV (BRL)',
        'churn_pct':         'Churn Risk %',
        'churn_risk_tier':   'Risk Tier',
        'avg_review_score':  'Avg Review',
        'clv_12m':           '12m CLV (BRL)',
    }
    st.dataframe(
        at_risk_hv[list(display_cols.keys())]
        .rename(columns=display_cols)
        .head(100)
        .round(2),
        use_container_width=True,
        hide_index=True,
    )

    csv_at_risk = at_risk_hv[list(display_cols.keys())].to_csv(index=False)
    st.download_button(
        label="Download high-value at-risk list (CSV)",
        data=csv_at_risk,
        file_name="high_value_at_risk_customers.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader("Customer Lifetime Value (CLV)")
    st.caption(
        "12-month CLV estimated via BG/NBD + Gamma-Gamma model "
        "(Day 5) or monetary-based fallback."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Projected CLV (BRL)", f"{filtered['clv_12m'].sum():,.0f}")
    c2.metric("Avg CLV per Customer",       f"BRL {filtered['clv_12m'].mean():.2f}")
    c3.metric("Median CLV",                 f"BRL {filtered['clv_12m'].median():.2f}")
    c4.metric("Top 10% CLV threshold",
              f"BRL {filtered['clv_12m'].quantile(0.9):.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # CLV distribution histogram
        fig_clv_hist = px.histogram(
            filtered,
            x='clv_12m',
            nbins=60,
            title="CLV Distribution (12-month)",
            labels={'clv_12m': '12-month CLV (BRL)'},
            color_discrete_sequence=['#3C3489'],
            height=350,
        )
        fig_clv_hist.update_layout(
            bargap=0.05,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_clv_hist, use_container_width=True)

    with col2:
        # CLV by segment box plot
        fig_clv_box = px.box(
            filtered,
            x='segment',
            y='clv_12m',
            color='segment',
            title="CLV by Segment",
            labels={'clv_12m': '12-month CLV (BRL)'},
            color_discrete_sequence=px.colors.qualitative.Set2,
            height=350,
        )
        fig_clv_box.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_clv_box, use_container_width=True)

    st.markdown("---")

    # Pareto analysis — what % of customers drive what % of CLV
    st.subheader("Pareto Analysis — CLV Concentration")

    clv_sorted = filtered['clv_12m'].sort_values(ascending=False).reset_index(drop=True)
    cumulative_clv = clv_sorted.cumsum() / clv_sorted.sum() * 100
    cumulative_cust = (clv_sorted.index + 1) / len(clv_sorted) * 100

    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Scatter(
        x=cumulative_cust,
        y=cumulative_clv,
        mode='lines',
        name='Cumulative CLV %',
        line=dict(color='#993C1D', width=2),
    ))
    fig_pareto.add_shape(
        type='line', x0=0, y0=0, x1=100, y1=100,
        line=dict(color='gray', dash='dash'),
    )
    # Mark the 20% customer line
    idx_20 = int(len(cumulative_cust) * 0.2)
    clv_at_20 = cumulative_clv.iloc[idx_20]
    fig_pareto.add_annotation(
        x=20, y=clv_at_20,
        text=f"Top 20% customers<br>= {clv_at_20:.1f}% of CLV",
        showarrow=True, arrowhead=2,
        bgcolor='white', bordercolor='gray',
    )
    fig_pareto.update_layout(
        title="Pareto Curve — Customer % vs CLV %",
        xaxis_title="Cumulative % of Customers",
        yaxis_title="Cumulative % of CLV",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown("---")

    # CLV vs churn risk scatter
    st.subheader("CLV vs Churn Risk")
    fig_clv_churn = px.scatter(
        filtered.sample(min(len(filtered), 5000), random_state=42),
        x='clv_12m',
        y='churn_pct',
        color='segment',
        size='monetary',
        size_max=12,
        hover_data={
            'customer_id': True,
            'segment': True,
            'clv_12m': ':.2f',
            'churn_pct': ':.1f',
            'monetary': ':.2f',
        },
        title="12-month CLV vs Churn Risk % (bubble size = total spend)",
        labels={
            'clv_12m':   '12-month CLV (BRL)',
            'churn_pct': 'Churn Risk %',
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
        opacity=0.65,
        height=420,
    )
    fig_clv_churn.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_clv_churn, use_container_width=True)

with tab4:
    st.subheader("Executive Business Summary")
    st.caption("Aggregated insights across all filtered customers.")

    # Revenue by segment bar
    rev_seg = (
        filtered.groupby('segment')['monetary']
        .sum()
        .reset_index()
        .sort_values('monetary', ascending=True)
    )
    fig_rev_bar = px.bar(
        rev_seg,
        x='monetary',
        y='segment',
        orientation='h',
        title="Total Revenue by Segment (BRL)",
        labels={'monetary': 'Total Revenue (BRL)', 'segment': ''},
        color='monetary',
        color_continuous_scale='Teal',
        height=300,
    )
    fig_rev_bar.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_rev_bar, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Average review score by segment
        review_seg = (
            filtered.groupby('segment')['avg_review_score']
            .mean()
            .reset_index()
            .sort_values('avg_review_score', ascending=True)
        )
        fig_review = px.bar(
            review_seg,
            x='avg_review_score',
            y='segment',
            orientation='h',
            title="Avg Review Score by Segment",
            labels={'avg_review_score': 'Avg Review (1–5)', 'segment': ''},
            color='avg_review_score',
            color_continuous_scale='Greens',
            range_x=[1, 5],
            height=280,
        )
        fig_review.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_review, use_container_width=True)

    with col2:
        # Preferred payment method breakdown
        pay_counts = (
            filtered['preferred_payment']
            .value_counts()
            .reset_index()
        )
        pay_counts.columns = ['payment_type', 'count']
        fig_pay = px.pie(
            pay_counts,
            names='payment_type',
            values='count',
            title="Preferred Payment Method",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
            height=280,
        )
        fig_pay.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_pay, use_container_width=True)

    st.markdown("---")

    # Top product categories
    st.subheader("Top Product Categories")
    cat_counts = (
        filtered['favourite_category']
        .value_counts()
        .head(15)
        .reset_index()
    )
    cat_counts.columns = ['category', 'customers']
    cat_counts['revenue'] = cat_counts['category'].map(
        filtered.groupby('favourite_category')['monetary'].sum()
    )

    fig_cat = px.bar(
        cat_counts.sort_values('customers'),
        x='customers',
        y='category',
        orientation='h',
        title="Top 15 Categories by Customer Count",
        labels={'customers': 'Customers', 'category': ''},
        color='revenue',
        color_continuous_scale='Oranges',
        height=420,
    )
    fig_cat.update_layout(
        coloraxis_colorbar_title="Revenue (BRL)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("---")

    # Full data export
    st.subheader("Export Data")
    export_cols = [
        'customer_id', 'segment', 'spend_tier',
        'monetary', 'avg_order_value', 'recency_days',
        'avg_review_score', 'churn_risk_tier', 'churn_pct',
        'return_probability', 'clv_12m',
    ]
    export_df = filtered[export_cols].round(3)

    st.download_button(
        label=f"Download filtered dataset ({len(export_df):,} rows)",
        data=export_df.to_csv(index=False),
        file_name="ecommerce_customer_analytics.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # Model metadata
    with st.expander("Model & Data Details"):
        st.markdown(f"""
**Dataset:** Brazilian E-Commerce Public Dataset (Olist)
**Orders:** 99,441 delivered orders · 2016–2018
**Customers:** {len(df):,} unique customers in feature store

**Segmentation model**
- Algorithm: KMeans (k=6, elbow method)
- Features: RFM + 15 behavioural features
- Validation: DBSCAN sanity check + Kruskal-Wallis

**Churn model**
- Algorithm: XGBoost (binary:logistic)
- Target: Repeat purchase within observation window
- Tuning: Optuna (50 trials, PR-AUC objective)
- Imbalance handling: scale_pos_weight + SMOTE comparison

**CLV model**
- Algorithm: BG/NBD (frequency) + Gamma-Gamma (monetary)
- Horizon: 12-month projection
- Library: lifetimes

**Snapshot date:** 2018-10-18
        """)


