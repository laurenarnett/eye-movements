from PIL import Image
import numpy as np
import sys

'''
Measure performance of model by its true positive rate
Predicting Eye-Fixations
'''

def main(ground_truth_path, pred_path):
    partitions = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    with np.load(preds_file) as preds:
        for idx,img in enumerate(os.listdir(ground_truth_path)):

            # get ground truth and prediction values
            test = np.asarray(Image.open(ground_truth_path + img))
            pred = preds[idx]
            tpr = dict(dict())

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
            fig,ax = plt.subplots()
            ax.plot(partition, mean_tprs)
            plt.xlim([0.0,1.0])
            plt.ylim([0.0,1.05])
            plt.xlabel("Percent Salient")
            plt.ylabel("True Positive Rate")

if __name__ =="__main__":
    main(sys.argv[1],sys.argv[2])
