import numpy as np
import pandas as pd
from utils import *

import matplotlib.pyplot as plt

import seaborn as sns
#%%

# LOAD TEST DF WITH PREDICTIONS

test_df = pd.read_csv('PATH TO TEST DF')

test_df['race_categorize_demographics'] = test_df.apply(
    lambda row: race_categorize(row), axis=1)
test_df['race_ethnicity_demographics'] = test_df.apply(
    lambda row: ethnicity_categorize(row), axis=1)

#%%




get_performance_metrics_descriptive(test_df,'HCM','Pred_CNN',0.15)

get_performance_metrics_age(test_df,'HCM','Pred_CNN',0.15)


############# Make ROC Curves ############
# If making plot for 4 formats of images
fpr_sig = dict()
tpr_sig = dict()
roc_auc_sig = dict()
recall_sig = dict()
precision_sig = dict()
pr_auc_sig = dict()
f1_sig = dict()
formats = ['stand','alter','two_r','shuff']
for i in range(4):
    fpr_sig[i], tpr_sig[i], _ = roc_curve(test_df.loc[test_df['image_format']==formats[i]]['HCM'], test_df.loc[test_df['image_format']==formats[i]]['Pred_CNN'])
    roc_auc_sig[i] = auc(fpr_sig[i], tpr_sig[i])
    recall_sig[i], precision_sig[i], _ = precision_recall_curve(test_df.loc[test_df['image_format']==formats[i]]['HCM'], test_df.loc[test_df['image_format']==formats[i]]['Pred_CNN'])
    pr_auc_sig[i] = average_precision_score(test_df.loc[test_df['image_format']==formats[i]]['HCM'], test_df.loc[test_df['image_format']==formats[i]]['Pred_CNN'])
    f1_sig[i] = max(np.multiply(2, np.divide(np.multiply(precision_sig[i], recall_sig[i]), np.add(recall_sig[i], precision_sig[i]))))


plt.figure(figsize=(8, 8), dpi=80)
lw = 1.5
plt.style.use('classic')
plt.plot(fpr_sig[0], tpr_sig[0], 
         lw=lw, label='Standard (AUROC = %0.3f)' % (roc_auc_sig[0]))
plt.plot(fpr_sig[1], tpr_sig[1],
         lw=lw, label='Alternate (AUROC = %0.3f)' % (roc_auc_sig[1]))
plt.plot(fpr_sig[2], tpr_sig[2],
         lw=lw, label='Two Rhythm (AUROC = %0.3f)' % (roc_auc_sig[2]))
plt.plot(fpr_sig[3], tpr_sig[3],
         lw=lw, label='Standard Shuffled (AUROC = %0.3f)' % (roc_auc_sig[3]))

plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# %%


### FALSE POSITIVE ANALYSIS #####
# LOAD ECG ECHO PAIRS AND IVSd data
echo_data = pd.read_csv('PATH TO DATA')


# %%
merged = test_df.merge(echo_data[['FileID','cleanedIVSd']],how='left',on='FileID')
# %%
trueneg = merged.loc[(merged['HCM']==False)&(merged['Pred_CNN']<.15)]
falsepos = merged.loc[(merged['HCM']==False)&(merged['Pred_CNN']>=.15)]
trueneg_IVSD = trueneg.dropna(subset = ['cleanedIVSd'])
falsepos_IVSD = falsepos.dropna(subset = ['cleanedIVSd'])

# Plotting the histograms
sns.kdeplot(trueneg_IVSD[['cleanedIVSd']].squeeze(), color = 'green', shade = True, label='True Negatives')  
sns.kdeplot(falsepos_IVSD[['cleanedIVSd']].squeeze(), color = 'red', shade = True, label='False Positives')
plt.xlabel('IVSd')
plt.ylabel('Proportion of ECGs')
plt.legend()
plt.show()
# %%
u_stat, p_val = stats.mannwhitneyu(trueneg_IVSD[['cleanedIVSd']], falsepos_IVSD[['cleanedIVSd']])

print("U-statistic:", u_stat)
print("p-value:", p_val)

#### Code for predictions by time from diagnosis
all_ecgs_pos = pd.read_csv('Path To Data')
# %%
bins = [-10000,-1095,-365, 0, 365, 1095,100000]  # Adjust bins as needed
labels = ['<-1095','-1095 to -365','-365 to 0', '0 to 365', '365 to 1095', '1095+']

all_ecgs_pos['time_bin'] = pd.cut(all_ecgs_pos['timedelta_days'], bins=bins, labels=labels, right=False)

# Step 2: Generate the boxplot
plt.figure(figsize=(10, 6))
all_ecgs_pos.boxplot(column='Pred_CNN', by='time_bin', grid=False, showmeans=False)

# Step 3: Customize the plot
plt.xlabel('Time between ECG and Date of Diagnosis')
plt.ylabel('Probability of HCM')
plt.title('Predictions by Time from Diagnosis')
plt.suptitle('')  # Removes the default subtitle added by pandas
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/root/Desktop/HCM_Predictions_time.png')
# %%

all_ecgs_pos = pd.read_csv('/mnt/nfs_yale_ecg/HCM/data/final_model_all_standard_preds_all_test_pos_MRN_ECGs.csv')

bins = [-10000,-1095,-365, 0, 365, 1095,100000]  # Adjust bins as needed
labels = ['<-3 years','-3 to -1 years','-1 to 0 years', '0 to 1 years', '1 to 3 years', '>3 years']

all_ecgs_pos['time_bin'] = pd.cut(all_ecgs_pos['timedelta_days'], bins=bins, labels=labels, right=False)
counts = all_ecgs_pos['time_bin'].value_counts(sort=False)
# Update labels to include counts
updated_labels = [f"{label}\n(n={count})" for label, count in zip(labels, counts)]

# Generate the boxplot
plt.figure(figsize=(10, 6))
all_ecgs_pos.boxplot(column='Pred_CNN', by='time_bin', grid=False, showmeans=False)

# Customize the plot
plt.xlabel('Time between ECG and Date of Diagnosis (years)')
plt.ylabel('AI Probability of HCM (Median, IQR)')
plt.title('Predictions by Time from Diagnosis')
plt.suptitle('')  # Removes the default subtitle added by pandas

# Update x-axis tick labels
plt.xticks(ticks=range(1, len(updated_labels) + 1), labels=updated_labels, rotation=45)

plt.tight_layout()
plt.show()
