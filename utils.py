#%%
# Load require packages and libraries
import numpy as np
import math
from skimage.transform import resize, rotate
from PIL import Image
import tensorflow as tf
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc, average_precision_score, precision_recall_curve
import pandas as pd

#%%


### Code to load data into the model for Training
class DataSequenceTrain(tf.keras.utils.Sequence):
    """
    Keras Sequence object to train a model on larger-than-memory data.
    """
    def __init__(self, df, batch_size, mode):
        self.df = df # your pandas dataframe
        self.bsz = batch_size # batch size
        self.mode = mode # image or signal

        # Take labels and a list of image locations in memory
        class_names = ['HCM']
        #class_names = ['Under40']
        self.labels = self.df[class_names].values
        if self.mode == 'image':
            self.im_list = self.df['image_path'].tolist()

    def __len__(self):
        # compute number of batches to yield
        return int(math.ceil(len(self.df) / float(self.bsz)))


    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return self.labels[idx * self.bsz: (idx + 1) * self.bsz]

    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        if self.mode == 'image':
            return np.array([rotate(resize(np.array(Image.open(im).convert('L').convert('RGB')), (300,300)), np.random.uniform(-10,10,1)[0]) for im in self.im_list[idx * self.bsz: (1 + idx) * self.bsz]])
    def __getitem__(self, idx):
        if self.mode=='image':
            batch_x = self.get_batch_features(idx)
            batch_y = self.get_batch_labels(idx)
            return batch_x, batch_y



### Code to load data into the model for Testing
class DataSequenceTest(tf.keras.utils.Sequence):
    """
    Keras Sequence object to train a model on larger-than-memory data.
    """
    def __init__(self, df, batch_size, mode):
        self.df = df # your pandas dataframe
        self.bsz = batch_size # batch size
        self.mode = mode # image or signal

        # Take labels and a list of image locations in memory
        class_names = ['HCM']
        self.labels = self.df[class_names].values
        if self.mode == 'image':
            self.im_list = self.df['image_path'].tolist()
    def __len__(self):
        # compute number of batches to yield
        return int(math.ceil(len(self.df) / float(self.bsz)))
    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return self.labels[idx * self.bsz: (idx + 1) * self.bsz]
    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        if self.mode == 'image':
            return np.array([resize(np.array(Image.open(im).convert('L').convert('RGB')), (300,300)) for im in self.im_list[idx * self.bsz: (1 + idx) * self.bsz]])
    def __getitem__(self, idx):
        if self.mode=='image':
            batch_x = self.get_batch_features(idx)
            batch_y = self.get_batch_labels(idx)
            return batch_x, batch_y

##### Stats Metrics Helpers
def TP(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 1))

def TN(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 0))

def FP(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 0))

def FN(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 1))

def accuracy(y, pred, th=0.5):
    tp = TP(y,pred,th)
    fp = FP(y,pred,th)
    tn = TN(y,pred,th)
    fn = FN(y,pred,th)
    return (tp+tn)/(tp+tn+fp+fn)

def precision(y, pred, th=0.5):
    tp = TP(y,pred,th)
    fp = FP(y,pred,th)
    return tp/(tp+fp)

def recall(y, pred, th=0.5):
    tp = TP(y,pred,th)
    fn = FN(y,pred,th)
    
    return tp/(tp+fn)

def sensitivity(y, pred, th=0.5):
    tp = TP(y,pred,th)
    fn = FN(y,pred,th)
    
    return tp/(tp+fn)

def specificity(y, pred, th=0.5):
    tn = TN(y,pred,th)
    fp = FP(y,pred,th)
    return tn/(tn+fp)

def fscore(y,pred,beta=1.0):
    p = precision(y,pred)
    r = recall(y,pred)
    return (1+beta**2)*p*r/((1+beta**2)*p+r)
    
def f1(y,pred,beta=1.0):
    p = precision(y,pred)
    r = recall(y,pred)
    return (1+beta**2)*p*r/((1+beta**2)*p+r)

def prcurve(y,pred):
    P = []
    R = []
    thresholds = np.arange(0.0,1.0+0.01,0.01)
    for th in thresholds:
        P.append(precision(y,pred,th))
        R.append(recall(y,pred,th))
    plt.figure(figsize=(10,10))
    plt.plot(R,P,linewidth=2)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Recall",fontsize=12)
    plt.ylabel("Precision",fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.title("Precision Recall Curve",fontsize=12)
    plt.savefig('Precision_Recall_Curve.png')

def fpr(y,pred,th=0.5):
    fp = FP(y,pred,th)
    tn = TN(y,pred,th)
    return fp/(tn+fp)

def prevalence(y,pred, th=.5):
    tp = TP(y,pred,th)
    fp = FP(y,pred,th)
    tn = TN(y,pred,th)
    fn = FN(y,pred,th)
    
    return (tp+fn)/(tp+fp+tn+fn)
    
def ppv(y, pred, th=0.5):
    tp = TP(y,pred,th)
    fp = FP(y,pred,th)
    return tp/(tp+fp)

def npv(y, pred, th=0.5):
    tn = TN(y,pred,th)
    fn = FN(y,pred,th)
    return tn/(tn+fn)

## Helper Function to print out all metrics for a set of predictions
def get_performance_metrics(y, pred, class_labels, thresholds=[]):
    
    if len(thresholds) == len(class_labels):
        thresholds = thresholds
    else:
        thresholds = [.5] * len(class_labels)
        for i in range(len(class_labels)):
            threshes = np.arange(.01,.99,.01)
            maxf1 = 0
            bestthresh = 0
            for thresh in threshes:
                testf1 = f1_score(y.iloc[:, i], pred.iloc[:, i]>thresh)
                if testf1 > maxf1:
                    bestthresh = thresh
                    maxf1 = testf1
            thresholds[i] = bestthresh

    columns = ["Labels","Cutoff", "TP","TN","FP","FN","accuracy","ppv", "npv","specificity","sensitivity","AUROC", "F1", "AUPRC","MaxF1"]
    
    df = pd.DataFrame(columns=columns)
	
    
    for i in range(len(class_labels)):
        df.loc[i] = [""] + [0] * (len(columns) - 1)
        df.loc[i][0] = class_labels[i]
        for j in range(1,len(columns)):
            try:
                if columns[j] == 'Cutoff':
                    df.loc[i][j] = round(thresholds[i],3)
                elif columns[j] == 'AUROC': 
                    df.loc[i][j] = round(roc_auc_score(y.iloc[:, i], pred.iloc[:, i]), 3)
                elif columns[j] == 'F1':
                    df.loc[i][j] = round(f1_score(y.iloc[:, i], pred.iloc[:, i]>thresholds[i]), 3)
                elif columns[j] == 'AUPRC':
                    df.loc[i][j] = round(average_precision_score(y.iloc[:, i], pred.iloc[:, i]), 3)
                elif columns[j] == 'MaxF1':
                    recall_sig, precision_sig, _ = precision_recall_curve(y.iloc[:,i], pred.iloc[:,i])
                    df.loc[i][j] = max(np.multiply(2, np.divide(np.multiply(precision_sig, recall_sig), np.add(recall_sig, precision_sig))))
                else:
                    df.loc[i][j] = round(globals()[columns[j]](y.iloc[:, i], pred.iloc[:, i], thresholds[i]), 3)
            except:
                df.loc[i][j] = "error"
    df.loc[len(class_labels)] = [""] + [0]*(len(columns) - 1) 
    df.loc[len(class_labels)][0] = "Weigted Average"
    total_labels = np.sum(np.sum(y.iloc[:,1:]))
    for j in range(1,len(columns)):
        try:
            sum = 0
            for i in range(1,len(class_labels)):
                sum += df.loc[i][j] *(np.sum(y.iloc[:,i] == 1)/total_labels)
            df.loc[len(class_labels)][j] = round(sum,3)
        except:
            df.loc[len(class_labels)][j] = ""

    return df




# from https://github.com/yandexdataschool/roc_comparison/blob/master/compare_auc_delong_xu.py

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight=None):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    sample_weight = None
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    sample_weight = None
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


def calc_auc_ci(y_true, y_pred, alpha=0.95):
    auc, auc_cov = delong_roc_variance(y_true,y_pred)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    ci[ci > 1] = 1
    return auc, ci

def get_performance_metrics_descriptive(data, class_label,pred_name, threshold = .1):
    columns = ["Labels","Cutoff", "TP","TN","FP","FN","accuracy","ppv", "npv","specificity","sensitivity","AUROC","AUROC_L",'AUROC_U', "F1", "AUPRC","AUPRC_L",'AUPRC_U',"MaxF1"]
    df = pd.DataFrame(columns=columns)
	
    breakdowns = ['All','Male','Female','>=65','<65','Hispanic','White','Black','Asian','Other','Unknown',
                  'Paced','NotPaced','Afib/Flutter','No Afib/Flutter','LBBB','No LBBB',
                  'RBBB','No RBBB','LVH','No LVH']
    
    for i in range(len(breakdowns)):
        if i == 0:
            y_new = data[class_label]
            pred_new = data[pred_name]
        elif i == 1:
            y_new = data.loc[data['PatientSex']=='M'][class_label]
            pred_new = data.loc[data['PatientSex']=='M'][pred_name]
        elif i == 2:
            y_new = data.loc[data['PatientSex']=='F'][class_label]
            pred_new = data.loc[data['PatientSex']=='F'][pred_name]
        elif i == 3:
            y_new = data.loc[data['NewAge']>=65][class_label]
            pred_new = data.loc[data['NewAge']>=65][pred_name]
        elif i == 4:
            y_new = data.loc[data['NewAge']<65][class_label]
            pred_new = data.loc[data['NewAge']<65][pred_name]
        elif i == 5:
            y_new = data.loc[data['race_ethnicity_demographics']=='Hispanic'][class_label]
            pred_new = data.loc[data['race_ethnicity_demographics']=='Hispanic'][pred_name]           
        elif i == 6:
            y_new = data.loc[data['race_ethnicity_demographics']=='White'][class_label]
            pred_new = data.loc[data['race_ethnicity_demographics']=='White'][pred_name]  
        elif i == 7:
            y_new = data.loc[data['race_ethnicity_demographics']=='Black'][class_label]
            pred_new = data.loc[data['race_ethnicity_demographics']=='Black'][pred_name]  
        elif i == 8:
            y_new = data.loc[data['race_ethnicity_demographics']=='Asian'][class_label]
            pred_new = data.loc[data['race_ethnicity_demographics']=='Asian'][pred_name]  
        elif i == 9:
            y_new = data.loc[data['race_ethnicity_demographics']=='Others'][class_label]
            pred_new = data.loc[data['race_ethnicity_demographics']=='Others'][pred_name]  
        elif i == 10:
            y_new = data.loc[data['race_ethnicity_demographics']=='Missing'][class_label]
            pred_new = data.loc[data['race_ethnicity_demographics']=='Missing'][pred_name]  
        elif i == 11:
            y_new = data.loc[data['Diagnosis'].str.contains("pacing|paced",case=False)][class_label]
            pred_new = data.loc[data['Diagnosis'].str.contains("pacing|paced",case=False)][pred_name]  
        elif i == 12:
            y_new = data.loc[~data['Diagnosis'].str.contains("pacing|paced",case=False)][class_label]
            pred_new = data.loc[~data['Diagnosis'].str.contains("pacing|paced",case=False)][pred_name]  
        elif i == 13:
            y_new = data.loc[data['Diagnosis'].str.contains("afib|flutter|fibrillation",case=False)][class_label]
            pred_new = data.loc[data['Diagnosis'].str.contains("afib|flutter|fibrillation",case=False)][pred_name]  
        elif i == 14:
            y_new = data.loc[~data['Diagnosis'].str.contains("afib|flutter|fibrillation",case=False)][class_label]
            pred_new = data.loc[~data['Diagnosis'].str.contains("afib|flutter|fibrillation",case=False)][pred_name]  
        elif i == 15:
            y_new = data.loc[data['Diagnosis'].str.contains("lbbb|left bundle branch block",case=False) & ~(data['Diagnosis'].str.contains("incomplete",case=False))][class_label]
            pred_new = data.loc[data['Diagnosis'].str.contains("lbbb|left bundle branch block",case=False) & ~(data['Diagnosis'].str.contains("incomplete",case=False))][pred_name]  
        elif i == 16:
            y_new = data.loc[~data['Diagnosis'].str.contains("lbbb|left bundle branch block",case=False) & ~(data['Diagnosis'].str.contains("incomplete",case=False))][class_label]
            pred_new = data.loc[~data['Diagnosis'].str.contains("lbbb|left bundle branch block",case=False) & ~(data['Diagnosis'].str.contains("incomplete",case=False))][pred_name]  
        elif i == 17:
            y_new = data.loc[data['Diagnosis'].str.contains("rbbb|right bundle branch block",case=False) & ~(data['Diagnosis'].str.contains("incomplete",case=False))][class_label]
            pred_new = data.loc[data['Diagnosis'].str.contains("rbbb|right bundle branch block",case=False) & ~(data['Diagnosis'].str.contains("incomplete",case=False))][pred_name]  
        elif i == 18:
            y_new = data.loc[~data['Diagnosis'].str.contains("rbbb|right bundle branch block",case=False) & ~(data['Diagnosis'].str.contains("incomplete",case=False))][class_label]
            pred_new = data.loc[~data['Diagnosis'].str.contains("rbbb|right bundle branch block",case=False) & ~(data['Diagnosis'].str.contains("incomplete",case=False))][pred_name]  
        elif i == 19:
            y_new = data.loc[data['Diagnosis'].str.contains('LVH|left ventricular hypertrophy',case=False)][class_label]
            pred_new = data.loc[data['Diagnosis'].str.contains('LVH|left ventricular hypertrophy',case=False)][pred_name]  
        elif i == 20:
            y_new = data.loc[~data['Diagnosis'].str.contains('LVH|left ventricular hypertrophy',case=False)][class_label]
            pred_new = data.loc[~data['Diagnosis'].str.contains('LVH|left ventricular hypertrophy',case=False)][pred_name]                                  
     
        
        df.loc[i] = [""] + [0] * (len(columns) - 1)
        df.loc[i][0] = breakdowns[i]
        for j in range(1,len(columns)):
            try:
                if columns[j] == 'Cutoff':
                    df.loc[i][j] = round(threshold,3)
                elif columns[j] == 'AUROC': 
                    df.loc[i][j] = round(roc_auc_score(y_new, pred_new), 3)
                elif columns[j] == 'F1':
                    df.loc[i][j] = round(f1_score(y_new, pred_new>threshold), 3)
                elif columns[j] == 'AUPRC':
                    df.loc[i][j] = round(average_precision_score(y_new, pred_new), 3)
                elif columns[j] == 'MaxF1':
                    recall_sig, precision_sig, _ = precision_recall_curve(y_new, pred_new)
                    df.loc[i][j] = max(np.multiply(2, np.divide(np.multiply(precision_sig, recall_sig), np.add(recall_sig, precision_sig))))
                elif columns[j] == "AUROC_L":
                    findci = calc_auc_ci(y_new,pred_new)
                    df.loc[i][j] = round(findci[1][0],4)
                elif columns[j] == "AUROC_U":
                    findci = calc_auc_ci(y_new,pred_new)
                    df.loc[i][j] = round(findci[1][1],4)
                elif columns[j] == "AUPRC_L":
                    df.loc[i][j] = CI_AUPRC(y_new,pred_new)[1]
                elif columns[j] == "AUPRC_U":
                    df.loc[i][j] = CI_AUPRC(y_new,pred_new)[0]
                else:
                    df.loc[i][j] = round(globals()[columns[j]](y_new, pred_new, threshold), 3)
            except:
                df.loc[i][j] = "error"
    return df

def get_performance_metrics_age(data, class_label,pred_name, threshold = .1):
    columns = ["Labels","Cutoff", "TP","TN","FP","FN","accuracy","ppv", "npv","specificity","sensitivity","AUROC","AUROC_L",'AUROC_U', "F1", "AUPRC","AUPRC_L",'AUPRC_U',"MaxF1"]
    data['NewAge'] = np.nan
    data.loc[(data['PatientAge']!='None')&(data['PatientAge']!='')&(~data['PatientAge'].isnull()),'NewAge']= data[(data['PatientAge']!='None')&(data['PatientAge']!='')&(~data['PatientAge'].isnull())]['PatientAge'].str[0:3].astype(int)
    df = pd.DataFrame(columns=columns)
	
    breakdowns = ['All','18-30','31-40','41-50','51-60','61-70','71-80','81-90']
    
    for i in range(len(breakdowns)):
        if i == 0:
            y_new = data[class_label]
            pred_new = data[pred_name]
        elif i == 1:
            y_new = data.loc[(data['NewAge']>=18)&(data['NewAge']<=30)][class_label]
            pred_new = data.loc[(data['NewAge']>=18)&(data['NewAge']<=30)][pred_name]
        elif i == 2:
            y_new = data.loc[(data['NewAge']>30)&(data['NewAge']<=40)][class_label]
            pred_new = data.loc[(data['NewAge']>30)&(data['NewAge']<=40)][pred_name]
        elif i == 3:
            y_new = data.loc[(data['NewAge']>40)&(data['NewAge']<=50)][class_label]
            pred_new = data.loc[(data['NewAge']>40)&(data['NewAge']<=50)][pred_name]
        elif i == 4:
            y_new = data.loc[(data['NewAge']>50)&(data['NewAge']<=60)][class_label]
            pred_new = data.loc[(data['NewAge']>50)&(data['NewAge']<=60)][pred_name]
        elif i == 5:
            y_new = data.loc[(data['NewAge']>60)&(data['NewAge']<=70)][class_label]
            pred_new = data.loc[(data['NewAge']>60)&(data['NewAge']<=70)][pred_name]      
        elif i == 6:
            y_new = data.loc[(data['NewAge']>70)&(data['NewAge']<=80)][class_label]
            pred_new = data.loc[(data['NewAge']>70)&(data['NewAge']<=80)][pred_name]  
        elif i == 7:
            y_new = data.loc[(data['NewAge']>80)&(data['NewAge']<=90)][class_label]
            pred_new = data.loc[(data['NewAge']>80)&(data['NewAge']<=90)][pred_name]  
       
        
        df.loc[i] = [""] + [0] * (len(columns) - 1)
        df.loc[i][0] = breakdowns[i]
        for j in range(1,len(columns)):
            try:
                if columns[j] == 'Cutoff':
                    df.loc[i][j] = round(threshold,3)
                elif columns[j] == 'AUROC': 
                    df.loc[i][j] = round(roc_auc_score(y_new, pred_new), 3)
                elif columns[j] == 'F1':
                    df.loc[i][j] = round(f1_score(y_new, pred_new>threshold), 3)
                elif columns[j] == 'AUPRC':
                    df.loc[i][j] = round(average_precision_score(y_new, pred_new), 3)
                elif columns[j] == 'MaxF1':
                    recall_sig, precision_sig, _ = precision_recall_curve(y_new, pred_new)
                    df.loc[i][j] = max(np.multiply(2, np.divide(np.multiply(precision_sig, recall_sig), np.add(recall_sig, precision_sig))))
                elif columns[j] == "AUROC_L":
                    findci = calc_auc_ci(y_new,pred_new)
                    df.loc[i][j] = round(findci[1][0],4)
                elif columns[j] == "AUROC_U":
                    findci = calc_auc_ci(y_new,pred_new)
                    df.loc[i][j] = round(findci[1][1],4)
                elif columns[j] == "AUPRC_L":
                    df.loc[i][j] = CI_AUPRC(y_new,pred_new)[1]
                elif columns[j] == "AUPRC_U":
                    df.loc[i][j] = CI_AUPRC(y_new,pred_new)[0]
                else:
                    df.loc[i][j] = round(globals()[columns[j]](y_new, pred_new, threshold), 3)
            except:
                df.loc[i][j] = "error"
    return df


def calc_auc_ci(y_true, y_pred, alpha=0.95):
    auc, auc_cov = delong_roc_variance(y_true,y_pred)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    ci[ci > 1] = 1
    return auc, ci

def CI_AUPRC(y, pred):
    track = []
    both = pd.DataFrame({'y':y,'pred':pred})
    for i in range(1000):
        new_both = both.sample(n=len(both),replace=True)
        track.append(average_precision_score(new_both['y'], new_both['pred']))
    
    ordered = np.sort(track)
    lower = np.percentile(ordered, 97.5)
    upper = np.percentile(ordered, 2.5)
    return(lower,upper)

def race_categorize(row):
    if row['PATIENT_RACE_ALL'] == 'White or Caucasian':
        return 'White'
    if row['PATIENT_RACE_ALL'] == 'White':
        return 'White'
    elif row['PATIENT_RACE_ALL'] == 'Black or African American':
        return 'Black'
    elif row['PATIENT_RACE_ALL'] == 'Unknown':
        return 'Missing'
    elif row['PATIENT_RACE_ALL'] == 'Not Listed':
        return 'Missing'
    elif row['PATIENT_RACE_ALL'] == 'Asian':
        return 'Asian'
    # ADD IN OTHERS AS APPROPRIATE
    return 'Others'



# Combine race and ethnicity in one column
def ethnicity_categorize(row):
    if row['PATIENT_ETHNICITY'] == 'Hispanic or Latino':
        return 'Hispanic'
    if row['PATIENT_ETHNICITY'] == 'Hispanic or Latina/o/x':
        return 'Hispanic'
    return row['race_categorize_demographics']