import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import statsmodels.formula.api as smf


df = pd.read_csv('dataset_audit_reply_data.csv')

df.head()

#model 1: log odds estimate on the probability of receiving a reply given email is received
logit_result = smf.logit("replied ~ C(Sender, Treatment(reference='pakistani_nyu')) + C(Venue) + C(paper_domain) + C(matching_country, Treatment(reference='No Match')) + h_index + academic_age",
    data=df)

res = logit_result.fit()
print(res.summary())

#model 2: log odds estimate on the probability of receiving a positive reply given email is received
logit_result = smf.logit("positive_reply ~ C(Sender, Treatment(reference='pakistani_nyu')) + C(Venue) + C(paper_domain) + C(matching_country, Treatment(reference='No Match')) + h_index + academic_age",
    data=df)

res = logit_result.fit()
print(res.summary())


#Figure 4A
mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['pdf.fonttype'] = 42
cm = 1/2.54
fig, axs = plt.subplots(1, 1, figsize = (7*cm, 9*cm))

sns.pointplot(data = df, y = 'Sender', x = 'replied', ax = axs, order = ['german', 'nigerian', 'pakistani_nyu', 'pakistani_lums'], linewidth = 0, errwidth = 1, ms = 6, capsize = 0.1)
axs.set_yticklabels(['German (NYU)', 'Nigerian (NYU)', 'Pakistani (NYU)', 'Pakistani (LUMS)'])
axs.set_xlabel('Reply rate', fontsize = 8)
axs.set_ylabel('Sender condition', fontsize = 8)
axs.tick_params(axis = 'both', which = 'major', labelsize = 8)
sns.despine()
plt.show()

# fig.savefig('figure_4a.pdf', bbox_inches = 'tight', dpi = 300)



#Figure 4B
log_odds = pd.read_csv('logistic_regression_reply_rate.csv')
log_odds['color'] = log_odds['p_value'].apply(lambda x: 'tab:orange' if x < 0.05 else 'black')

cm = 1/2.54
fig, axs = plt.subplots(1, 1, figsize = (7*cm, 9*cm))

for i, row in log_odds.iterrows():
    if pd.isnull(row['Coefficient']):
        continue

    coef = row['Coefficient']
    lower = row['Lower CI']
    upper = row['Upper CI']

    axs.errorbar(y = i, x = row['Coefficient'], xerr = [[coef - lower], [upper - coef],], fmt = 'o', color =row['color'], markersize = 3, capsize = 1, linewidth = 0.25)

axs.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
axs.set_yticklabels(log_odds['index'].values, fontsize = 7)
axs.axvline(x = 0, linestyle = '--', linewidth = 0.5, alpha = 0.25)

axs.set_ylim(-0.5, 20.5)

axs.set_xlabel('Log odds ratio of reply rate', fontsize = 8)
axs.tick_params(axis = 'both', labelsize = 8)
sns.despine()
# fig.savefig('figure_4b.pdf', bbox_inches = 'tight', dpi = 300)
