# Wine-analysis

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

# Helper functions
def load_data(path):
    return pd.read_csv(path)


def basic_stats(df):
    mean = df.mean()
    var = df.var(ddof=0)
    return mean, var


def freq_distribution(series):
    counts = series.value_counts().sort_index()
    explanation = f"Counted each unique '{series.name}' value to form its frequency distribution."
    return counts, explanation


def compute_from_distribution(counts):
    x = counts.index.values.astype(float)
    weights = counts.values.astype(float)
    mean_dist = np.average(x, weights=weights)
    var_dist = np.average((x - mean_dist)**2, weights=weights)
    return mean_dist, var_dist


def compute_intervals(data):
    n = len(data)
    m = data.mean()
    v = data.var(ddof=1)
    ci_mean = stats.t.interval(0.95, n-1, loc=m, scale=stats.sem(data))
    ci_var = ((n-1)*v / stats.chi2.ppf(0.975, n-1), (n-1)*v / stats.chi2.ppf(0.025, n-1))
    factor = stats.t.ppf(0.975, n-1) * np.sqrt((n-1)/stats.chi2.ppf(0.95, n-1))
    tol_int = (m - factor*np.sqrt(v), m + factor*np.sqrt(v))
    return ci_mean, ci_var, tol_int


def one_sample_test(data, popmean=10.5):
    t_stat, p_value = stats.ttest_1samp(data, popmean, alternative='greater')
    return t_stat, p_value

# Streamlit app configuration
st.set_page_config(page_title="Wine Data Analysis", layout="wide")
st.title("🍷 Wine Data Analysis & Visualization")
st.markdown("---")

# 1. Load data and basic stats
st.header("1. Load Data & Compute Basic Statistics")
df = load_data("C:/Users/hp/Downloads/winequality-red.csv")
mean_vals, var_vals = basic_stats(df.select_dtypes(include=np.number))
col1, col2 = st.columns([2, 1])
col1.subheader("Dataset Preview")
col1.dataframe(df.head(6), use_container_width=True)
col2.subheader("Mean & Variance (Full Dataset)")
col2.dataframe(pd.DataFrame({'Mean': mean_vals, 'Variance': var_vals}))
st.markdown("---")

# 2. Frequency distribution of 'quality'
st.header("2. Frequency Distribution of 'quality' (Full Dataset)")
counts_full, explanation_full = freq_distribution(df['quality'])
st.write(counts_full)
st.info(explanation_full)
col1, col2 = st.columns(2)
col1.plotly_chart(px.histogram(df, x='quality', nbins=len(counts_full), title='Histogram of Quality'), use_container_width=True)
col2.plotly_chart(px.pie(names=counts_full.index, values=counts_full.values, title='Pie Chart of Quality'), use_container_width=True)
st.markdown("---")

# 3. Sample-based stats and comparison
st.header("3. Sample vs. Full Dataset: Mean & Variance")

# Sample creation
sample_size = st.sidebar.number_input("Sample Size", min_value=5, max_value=len(df), value=30, step=1)
sample = df['quality'].sample(sample_size, random_state=42)

# Direct stats on sample
direct_mean, direct_var = sample.mean(), sample.var(ddof=0)

# Distribution-based stats on sample
counts_sample, _ = freq_distribution(sample)
dist_mean, dist_var = compute_from_distribution(counts_sample)

# Display metrics
col1, col2 = st.columns(2)
col1.metric("Sample Mean (Direct)", f"{direct_mean:.2f}")
col1.metric("Sample Mean (Dist)", f"{dist_mean:.2f}")
col2.metric("Sample Var (Direct)", f"{direct_var:.2f}")
col2.metric("Sample Var (Dist)", f"{dist_var:.2f}")

# Show full dataset for reference
st.write(f"**Full Dataset Mean:** {mean_vals['quality']:.2f}, **Variance:** {var_vals['quality']:.2f}")
st.markdown("---")

# 4. Confidence and tolerance intervals for 'alcohol'
st.header("4. Confidence & Tolerance Intervals for 'alcohol'")
shuffled = df['alcohol'].sample(frac=1, random_state=42)
train, test = np.split(shuffled, [int(0.8*len(shuffled))])
ci_mean, ci_var, tol_interval = compute_intervals(train)
st.write({
    '95% CI for Mean': ci_mean,
    '95% CI for Variance': ci_var,
    '95% Tolerance Interval': tol_interval
})
coverage = ((test >= tol_interval[0]) & (test <= tol_interval[1])).mean()
st.metric("Tolerance Coverage", f"{coverage:.2%}")
st.markdown("---")

# 5. Hypothesis test on 'alcohol'
st.header("5. Hypothesis Test: Mean 'alcohol' > 10.5")
t_stat, p_value = one_sample_test(train)
st.write({'t-statistic': t_stat, 'p-value': p_value})
if p_value < 0.05:
    st.success("Reject H0: Evidence that mean 'alcohol' > 10.5")
else:
    st.warning("Fail to reject H0: Insufficient evidence")
