import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib as mpl
from matplotlib.lines import Line2D



df = pd.read_csv('paywalled_papers_audit_reply_data.csv')

#log odds estimate on the probability of receiving a reply
logit_model = smf.logit("replied ~ C(sender_name, Treatment(reference='Karl Muller')) + C(sender_status, Treatment(reference='Journal Article')) + C(sender_university, Treatment(reference='New York University')) + C(paper_domain, Treatment(reference='Health Sciences')) + h_index + academic_age + C(matching_continent, Treatment(reference='No Match'))",
    data=df)

res = logit_model.fit()
print(res.summary())



#log odds estimate on the probability of receiving the paper in question
logit_model = smf.logit("is_positive_reply ~ C(sender_name, Treatment(reference='Karl Muller')) + C(sender_status, Treatment(reference='Journal Article')) + C(sender_university, Treatment(reference='New York University')) + C(paper_domain, Treatment(reference='Health Sciences')) + h_index + academic_age + C(matching_continent, Treatment(reference='No Match'))",
    data=df)

res = logit_model.fit()
print(res.summary())



#Figure 2B
log_odds = pd.read_csv('log_odds.csv')
mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['pdf.fonttype'] = 42


cm = 1/2.54
fig, axs = plt.subplots(1, 1, figsize = (9*cm, 13*cm))

for i, row in log_odds.iterrows():
    if pd.isnull(row['Coefficient']):
        continue

    coef = row['Coefficient']
    lower = row['Lower CI']
    upper = row['Upper CI']

    axs.errorbar(y = i, x = row['Coefficient'], xerr = [[coef - lower], [upper - coef],], fmt = 'o', color =row['color'], markersize = 3, capsize = 2, linewidth = 0.5)

axs.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
axs.set_yticklabels(['Academic Age', 'h index', '', 'Nigerian Sender x African University', 'German Sender x West Europe University', 'Pakistani Sender x South Asian University', '', 'Social Sciences', 'Physical Sciences', 'Life Sciences', '', 'San Diego State University', 'Louisiana State University', '', 'Class Project', '','Pakistani', 'Nigerian'], fontsize = 7)
axs.axvline(x = 0, linestyle = '--', linewidth = 0.5, alpha = 0.25)

axs.set_ylim(-0.5, 17.5)

axs.set_xlabel('Log odds ratio of reply rate', fontsize = 8)
axs.tick_params(axis = 'both', labelsize = 8)
sns.despine()

# fig.savefig('figure_2b.pdf', dpi = 300, transparent = True, bbox_inches = 'tight')


condition_means = df.groupby(['sender_name', 'sender_status', 'sender_university'])['replied'].mean().reset_index()
condition_stds = df.groupby(['sender_name', 'sender_status', 'sender_university'])['replied'].sem().reset_index()
condition_means = condition_means.merge(condition_stds, on = ['sender_name', 'sender_status', 'sender_university'], how = 'left')
condition_means.columns = ['sender_name', 'sender_status', 'sender_university', 'mean', 'sem']
# condition_means = condition_means.sort_values(by = ['sender_name', 'sender_status', 'sender_university'])
condition_means = condition_means.iloc[::-1].reset_index()


color_map = {
    'Karl Muller' : 'tab:green',
    'Faisal Khan' : 'tab:red',
    'Olu Adeyemi' : 'tab:blue',
}

marker_map = {
    'New York University' : 'o',
    'San Diego State University' : '^',
    'Louisiana State University' : 's',
}

condition_means['color'] = condition_means['sender_name'].apply(lambda x: color_map[x])
condition_means['marker'] = condition_means['sender_university'].apply(lambda x: marker_map[x])

condition_means.head()


fig, axs = plt.subplots(1, 1, figsize = (9*cm, 13*cm))

for i, row in condition_means.iterrows():
    if row['sender_status'] == 'Class project':
        axs.errorbar(y = i, x = row['mean'], xerr = [[row['sem']], [row['sem']],], fmt = row['marker'], color = row['color'], markerfacecolor = 'none', markersize = 7, capsize = 2, linewidth = 0.5)

    if row['sender_status'] == 'Journal Article':
        axs.errorbar(y = i, x = row['mean'], xerr = [[row['sem']], [row['sem']],], fmt = row['marker'], color = row['color'], markerfacecolor = row['color'], markersize = 7, capsize = 2, linewidth = 0.5)


axs.set_yticklabels([])
axs.set_yticks([])
sns.despine(left = True)
axs.tick_params(axis = 'both', labelsize = 8)
axs.set_xlabel('Reply rate', fontsize = 8)

# Legend for sender nationality (color)
color_legend = [
    Line2D([0], [0], color='tab:green', lw=3, label='German'),
    Line2D([0], [0], color='tab:red', lw=3, label='Pakistani'),
    Line2D([0], [0], color='tab:blue', lw=3, label='Nigerian'),
]

# Legend for sender university (marker shape)
marker_legend = [
    Line2D([0], [0], marker='o', color='black', linestyle='None', label='New York University', markersize=6),
    Line2D([0], [0], marker='^', color='black', linestyle='None', label='San Diego State University', markersize=6),
    Line2D([0], [0], marker='s', color='black', linestyle='None', label='Louisiana State University', markersize=6),
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

# fig.savefig('figure_2a.pdf', bbox_inches = 'tight', dpi = 300)
