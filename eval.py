from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 22

'''
Measure performance of model by its true positive rate
Predicting Eye-Fixations
'''

def main(pred_path):
    plt.figure(figsize=(9,12))
    for j,f in enumerate(pred_path):
        partitions = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        preds = np.load(f)
        prob = preds[0]
        gt = preds[1]
        for i in range(prob.shape[0]):

            # get ground truth and prediction values
            test = gt[i]
            pred = prob[i]

            tpr = dict()

            for partition in partitions:

                # calculate the k largest values according to threshold
                k = int(partition * pred.shape[0] * pred.shape[1])
                flat_pred = pred.flatten()
                idx = np.argpartition(flat_pred, -k)
                thresh = np.min(flat_pred[idx[-k:]])
                flat_pred[flat_pred < thresh] = 0 #threshold values
                # calculate true positive rate
                # num true positives
                num_true = np.dot(flat_pred, test.flatten())
                # num false negatives
                num_false = len(np.intersect1d(np.where(flat_pred==0),np.where(test!=0)))
                tpr[(partition,i)] = num_true / (num_true + num_false)


        mean_tprs = []
        keys = list(tpr.keys())    
        for partition in partitions:
            # all dict values that correspond to this partition
            part_keys = [key for key in keys if key[0] == partition]
            # find the average true positive rate
            mean_tpr = sum(tpr[key] for key in part_keys)/len(part_keys)
            mean_tprs.append(mean_tpr)

        # plot the true positive rate vs the percent salient of the image
        labels = ["Bayes Model", "Baseline"]
        plt.plot(partitions, mean_tprs, label=labels[j])
    plt.title("Performance for Saliency Thresholds")
    plt.xlim([0.0,0.35])
    plt.ylim([0.0,1.05])
    plt.xlabel("Percent Salient")
    plt.ylabel("True Positive Rate")
    plt.legend()
    #plt.tight_layout()
    plt.savefig("/Users/Lauren/eye-movements/tex/final_report/figures/tpr.pdf")

if __name__ =="__main__":
    main(['/Users/Lauren/eye-movements/_prob_baye.npy','/Users/Lauren/eye-movements/baseline_200_prob.npy'])
