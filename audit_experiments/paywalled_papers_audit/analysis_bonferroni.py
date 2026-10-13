import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy import stats


def bonferroni_summary(result):
    """Apply Bonferroni correction to a statsmodels logit result."""
    summary_df = pd.DataFrame({
        'Coefficient': result.params,
        'Std Err': result.bse,
        'P-value': result.pvalues,
    })

    n_tests = len(summary_df)
    adjusted_alpha = 0.05 / n_tests
    z_crit = stats.norm.ppf(1 - adjusted_alpha / 2)

    summary_df['Bonferroni P-value'] = (summary_df['P-value'] * n_tests).clip(upper=1.0)
    summary_df['Lower CI'] = summary_df['Coefficient'] - z_crit * summary_df['Std Err']
    summary_df['Upper CI'] = summary_df['Coefficient'] + z_crit * summary_df['Std Err']

    return summary_df


df = pd.read_csv('paywalled_papers_audit_reply_data.csv')

# ── Regression models estimating likelihood of any reply ──────────────────────

base = (
    "C(sender_name, Treatment(reference='Karl Muller')) + "
    "C(sender_status, Treatment(reference='Journal Article')) + "
    "C(sender_university, Treatment(reference='New York University'))"
)

controls = (
    "C(paper_domain, Treatment(reference='Health Sciences')) + "
    "h_index + academic_age + "
    "C(matching_continent, Treatment(reference='No Match'))"
)

# Model 1: Base (reply)
formula_base = "replied ~ " + base
model_base = smf.logit(formula_base, data=df)
result_base = model_base.fit()
print("=== Reply Model: Base ===")
print(result_base.summary())
print("\n--- Bonferroni-corrected ---")
bonf_base = bonferroni_summary(result_base)
print(bonf_base)

# Model 2: Controls (reply)
formula_controls = "replied ~ " + base + " + " + controls
model_controls = smf.logit(formula_controls, data=df)
result_controls = model_controls.fit()
print("\n=== Reply Model: Controls ===")
print(result_controls.summary())
print("\n--- Bonferroni-corrected ---")
bonf_controls = bonferroni_summary(result_controls)
print(bonf_controls)

# Model 3: Sender x University interaction (reply)
formula_sr = (
    "replied ~ " + base + " + " + controls + " + " +
    "C(sender_name, Treatment(reference='Karl Muller')):"
    "C(sender_university, Treatment(reference='New York University'))"
)
model_sr = smf.logit(formula_sr, data=df)
result_sr = model_sr.fit()
print("\n=== Reply Model: Sender x University ===")
print(result_sr.summary())
print("\n--- Bonferroni-corrected ---")
bonf_sr = bonferroni_summary(result_sr)
print(bonf_sr)


# ── Regression models estimating likelihood of positive reply ─────────────────

# Model 1: Base (positive reply)
formula_base_pos = "is_positive_reply ~ " + base
model_base_pos = smf.logit(formula_base_pos, data=df)
result_base_pos = model_base_pos.fit()
print("\n=== Positive Reply Model: Base ===")
print(result_base_pos.summary())
print("\n--- Bonferroni-corrected ---")
bonf_base_pos = bonferroni_summary(result_base_pos)
print(bonf_base_pos)

# Model 2: Controls (positive reply)
formula_controls_pos = "is_positive_reply ~ " + base + " + " + controls
model_controls_pos = smf.logit(formula_controls_pos, data=df)
result_controls_pos = model_controls_pos.fit()
print("\n=== Positive Reply Model: Controls ===")
print(result_controls_pos.summary())
print("\n--- Bonferroni-corrected ---")
bonf_controls_pos = bonferroni_summary(result_controls_pos)
print(bonf_controls_pos)

# Model 3: Sender x University interaction (positive reply)
formula_sr_pos = (
    "is_positive_reply ~ " + base + " + " + controls + " + " +
    "C(sender_name, Treatment(reference='Karl Muller')):"
    "C(sender_university, Treatment(reference='New York University'))"
)
model_sr_pos = smf.logit(formula_sr_pos, data=df)
result_sr_pos = model_sr_pos.fit()
print("\n=== Positive Reply Model: Sender x University ===")
print(result_sr_pos.summary())
print("\n--- Bonferroni-corrected ---")
bonf_sr_pos = bonferroni_summary(result_sr_pos)
print(bonf_sr_pos)


# ── Figure 2B (Bonferroni-corrected) ─────────────────────────────────────────

mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['pdf.fonttype'] = 42
cm = 1 / 2.54

# Save corrected results from the controls model for the forest plot
bonf_controls_plot = bonf_controls.reset_index()
bonf_controls_plot.columns = ['index', 'Coefficient', 'Std Err', 'P-value',
                              'Bonferroni P-value', 'Lower CI', 'Upper CI']
bonf_controls_plot.to_csv('log_odds_model_2_bonferroni.csv', index=False)

log_odds = bonf_controls_plot.copy()
log_odds['color'] = log_odds['Bonferroni P-value'].apply(
    lambda x: 'tab:orange' if x < 0.05 else 'black'
)
log_odds = log_odds.iloc[::-1].reset_index(drop=True)

fig, axs = plt.subplots(1, 1, figsize=(9 * cm, 13 * cm))

for i, row in log_odds.iterrows():
    if pd.isnull(row['Coefficient']):
        continue

    coef = row['Coefficient']
    lower = row['Lower CI']
    upper = row['Upper CI']

    axs.errorbar(
        y=i, x=coef,
        xerr=[[coef - lower], [upper - coef]],
        fmt='o', color=row['color'],
        markersize=3, capsize=2, linewidth=0.5
    )

axs.set_yticks(range(len(log_odds)))
axs.set_yticklabels(log_odds['index'].values, fontsize=7)
axs.axvline(x=0, linestyle='--', linewidth=0.5, alpha=0.25)
axs.set_ylim(-0.5, len(log_odds) - 0.5)
axs.set_xlabel('Log odds ratio of reply rate', fontsize=8)
axs.tick_params(axis='both', labelsize=8)
sns.despine()

fig.show()
fig.savefig('figure_2b_bonferroni.pdf', dpi=300, transparent=True, bbox_inches='tight')


# ── Figure 2A ────────────────────────────────────────────────────────────────

condition_means = df.groupby(['sender_name', 'sender_status', 'sender_university'])['replied'].mean().reset_index()
condition_stds = df.groupby(['sender_name', 'sender_status', 'sender_university'])['replied'].sem().reset_index()
condition_means = condition_means.merge(condition_stds, on=['sender_name', 'sender_status', 'sender_university'], how='left')
condition_means.columns = ['sender_name', 'sender_status', 'sender_university', 'mean', 'sem']
condition_means = condition_means.iloc[::-1].reset_index()

color_map = {
    'Karl Muller': 'tab:green',
    'Faisal Khan': 'tab:red',
    'Olu Adeyemi': 'tab:blue',
}

marker_map = {
    'New York University': 'o',
    'San Diego State University': '^',
    'Louisiana State University': 's',
}

condition_means['color'] = condition_means['sender_name'].apply(lambda x: color_map[x])
condition_means['marker'] = condition_means['sender_university'].apply(lambda x: marker_map[x])

fig, axs = plt.subplots(1, 1, figsize=(9 * cm, 13 * cm))

for i, row in condition_means.iterrows():
    fill = 'none' if row['sender_status'] == 'Class project' else row['color']
    axs.errorbar(
        y=i, x=row['mean'],
        xerr=[[row['sem']], [row['sem']]],
        fmt=row['marker'], color=row['color'],
        markerfacecolor=fill, markersize=7,
        capsize=2, linewidth=0.5
    )

axs.set_yticklabels([])
axs.set_yticks([])
sns.despine(left=True)
axs.tick_params(axis='both', labelsize=8)
axs.set_xlabel('Reply rate', fontsize=8)

color_legend = [
    Line2D([0], [0], color='tab:green', lw=3, label='German'),
    Line2D([0], [0], color='tab:red', lw=3, label='Pakistani'),
    Line2D([0], [0], color='tab:blue', lw=3, label='Nigerian'),
]

marker_legend = [
    Line2D([0], [0], marker='o', color='black', linestyle='None', label='New York University', markersize=6),
    Line2D([0], [0], marker='^', color='black', linestyle='None', label='San Diego State University', markersize=6),
    Line2D([0], [0], marker='s', color='black', linestyle='None', label='Louisiana State University', markersize=6),
]

fill_legend = [
    Line2D([0], [0], marker='o', color='black', markerfacecolor='black', linestyle='None', label='Journal Article', markersize=6),
    Line2D([0], [0], marker='o', color='black', markerfacecolor='none', linestyle='None', label='Class Project', markersize=6),
]

all_legends = color_legend + marker_legend + fill_legend
axs.legend(
    handles=all_legends,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    fontsize=7,
    frameon=False
)

fig.show()
fig.savefig('figure_2a_bonferroni.pdf', bbox_inches='tight', dpi=300)
