import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D


df = pd.read_csv('dataset_audit_data.csv')

# Common part of the model
base = (
    "C(sender_country, Treatment(reference='England')) + "
    "C(university_country, Treatment(reference='England')) + "
    "C(university_ranking, Treatment(reference='High')) + "
    "C(request_purpose, Treatment(reference='journal article'))")

controls = ("C(paper_domain, Treatment(reference='Health Sciences')) + "
    "h_index + academic_age"
)

formula_base = (
    "replied ~ " + base
)
model_base = smf.logit(formula_base, data=df)
result_base = model_base.fit()
print(result_base.summary())


formula_controls = (
    "replied ~ " + base + " + " + controls
)
model_controls = smf.logit(formula_controls, data=df)
result_controls = model_controls.fit()
print(result_controls.summary())



formula_sr = (
    "replied ~ " + base + " + " + controls + " + " +
    "C(sender_country, Treatment(reference='England')):"
    "C(university_ranking, Treatment(reference='High'))"
)

model_sr = smf.logit(formula_sr, data=df)
result_sr = model_sr.fit()
print(result_sr.summary())

formula_sc = (
    "replied ~ " + base + " + " + controls + " + " +
    "C(sender_country, Treatment(reference='England')):"
    "C(university_country, Treatment(reference='England'))"
)

model_sc = smf.logit(formula_sc, data=df)
result_sc = model_sc.fit()
print(result_sc.summary())

formula_cr = (
    "replied ~ " + base + " + " + controls + " + " +
    "C(university_country, Treatment(reference='England')):"
    "C(university_ranking, Treatment(reference='High'))"
)

model_cr = smf.logit(formula_cr, data=df)
result_cr = model_cr.fit()
print(result_cr.summary())


formula_all = (
    "replied ~ " + base + " + " + controls + " + " +
    "C(sender_country, Treatment(reference='England')):"
    "C(university_ranking, Treatment(reference='High')) + "
    "C(sender_country, Treatment(reference='England')):"
    "C(university_country, Treatment(reference='England')) + "
    "C(university_country, Treatment(reference='England')):"
    "C(university_ranking, Treatment(reference='High'))"
)

model_all = smf.logit(formula_all, data=df)
result_all = model_all.fit()
print(result_all.summary())



#Figure 4A
mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['pdf.fonttype'] = 42

cm = 1/2.54
df = pd.read_csv('dataset_audit_data.csv')
df['university_total'] = df['university_ranking'] + '-' + df['university_country']
condition_means = df.groupby(['sender', 'request_purpose', 'university_total'])['replied'].mean().reset_index()
condition_stds = df.groupby(['sender', 'request_purpose', 'university_total'])['replied'].sem().reset_index()
condition_means = condition_means.merge(condition_stds, on = ['sender', 'request_purpose','university_total'], how = 'left')
condition_means.columns =['sender', 'request_purpose', 'university_total', 'mean', 'sem']
condition_means = condition_means.sort_values(by = ['sender', 'request_purpose', 'university_total'])
condition_means.replace({'James Whitfield' : 'British', 'Kabelo Molefe' : 'South African'}, inplace = True)

color_map = {
    'British' : 'tab:green',
    'South African' : 'tab:blue',
}

marker_map = {
    'High-England' : 'o',
    'Low-England' : '^',
    'High-South Africa' : 's',
    'Low-South Africa': 'D'
}

condition_means['color'] = condition_means['sender'].apply(lambda x: color_map[x])
condition_means['marker'] = condition_means['university_total'].apply(lambda x: marker_map[x])

condition_means = condition_means.iloc[::-1].reset_index()

cm = 1/2.54
fig, axs = plt.subplots(1, 1, figsize = (9*cm, 13*cm))

for i, row in condition_means.iterrows():
    if row['request_purpose'] == 'final course project':
        axs.errorbar(y = i, x = row['mean'], xerr = [[row['sem']], [row['sem']],], fmt = row['marker'], color = row['color'], markerfacecolor = 'none', markersize = 7, capsize = 2, linewidth = 0.5)

    if row['request_purpose'] == 'journal article':
        axs.errorbar(y = i, x = row['mean'], xerr = [[row['sem']], [row['sem']],], fmt = row['marker'], color = row['color'], markerfacecolor = row['color'], markersize = 7, capsize = 2, linewidth = 0.5)

axs.set_yticklabels([])
sns.despine(left = True)
axs.tick_params(axis = 'both', labelsize = 8)
axs.set_xlabel('Reply rate', fontsize = 8)
axs.set_xlim(0, 0.2)



# Legend for sender nationality (color)
color_legend = [
    Line2D([0], [0], color='tab:green', lw=3, label='White-presenting'),
    Line2D([0], [0], color='tab:blue', lw=3, label='Black-presenting'),
]


# 'High-England' : 'o',
# 'Low-England' : '^',
# 'High-South Africa' : 's',
# 'Low-South Africa': 'D'
# Legend for sender university (marker shape)
marker_legend = [
    Line2D([0], [0], marker='o', color='black', linestyle='None', label='~150 ranked university in the UK', markersize=6),
    Line2D([0], [0], marker='^', color='black', linestyle='None', label='1001-1200 ranked university in the UK', markersize=6),
    Line2D([0], [0], marker='s', color='black', linestyle='None', label='~150 ranked university in South Africa', markersize=6),
    Line2D([0], [0], marker='D', color='black', linestyle='None', label='1001-1200 ranked university in South Africa', markersize=6),
]

# Legend for sender status (marker fill)
fill_legend = [
    Line2D([0], [0], marker='o', color='black', markerfacecolor='black', linestyle='None', label='Journal Article', markersize=6),
    Line2D([0], [0], marker='o', color='black', markerfacecolor='none', linestyle='None', label='Class Project', markersize=6),
]

# Combine all legend entries
all_legends = color_legend + marker_legend + fill_legend

# Add legend below the plot in 3 columns
axs.legend(
    handles=all_legends,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    fontsize=7,
    frameon=False
)

fig.show()

# fig.savefig('figure_4a.pdf', bbox_inches = 'tight', dpi = 300)



#Figure 4B
log_odds = pd.read_csv('logistic_regression_reply_rate.csv')
log_odds['color'] = log_odds['P-value'].apply(lambda x: 'tab:orange' if x < 0.05 else 'black')
log_odds = log_odds.iloc[::-1].reset_index()

cm = 1/2.54
fig, axs = plt.subplots(1, 1, figsize = (7*cm, 9*cm))

for i, row in log_odds.iterrows():
    if pd.isnull(row['Coefficient']):
        continue

    coef = row['Coefficient']
    lower = row['Lower CI']
    upper = row['Upper CI']

    axs.errorbar(y = i, x = row['Coefficient'], xerr = [[coef - lower], [upper - coef],], fmt = 'o', color =row['color'], markersize = 3, capsize = 1, linewidth = 0.25)

axs.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
axs.set_yticklabels(log_odds['index'].values, fontsize = 7)
axs.axvline(x = 0, linestyle = '--', linewidth = 0.5, alpha = 0.25)

axs.set_ylim(-0.5, 14.5)

axs.set_xlabel('Log odds ratio of reply rate', fontsize = 8)
axs.tick_params(axis = 'both', labelsize = 8)
sns.despine()

fig.show()
# fig.savefig('figure_4b.pdf', bbox_inches = 'tight', dpi = 300)
