import numpy as np
import os

acc = 0
total = 0

max_per = 0

pred_dir = "./msg_outputs/"
label_dir = "./msg_clusters_labels/"
res_dir = "./msg_results/"
#counts_dir = "./counts/"

preds_lst = sorted(os.listdir(pred_dir))


for i in range(len(preds_lst)):
    pred_file = preds_lst[i]
    
    preds = open(pred_dir + pred_file).readlines()
    
    label_file = pred_file.replace(".txt", "_labels.txt")
    
    labels = open(label_dir + label_file).readlines()
    #counts = open(counts_dir + pred_file).readlines()
    
    file_acc = 0
    file_total = 0
    
    res_file = open(res_dir + pred_file, "w")
    

    for i in range(len(preds)):
        pred_line = preds[i].strip("\n").strip(" ")
        label_line = labels[i].strip("\n").strip(" ")
        #count = int(counts[i])
        
        tmp = pred_line.split(" ")
        
        vals = []
        for val in tmp:
            vals.append(float(val))
        
        avg = np.mean(vals)
        avg1 = 1 / len(vals)
        sum_sqr_err = 0
        
        for val in vals:
            sum_sqr_err += np.power(val - avg, 2)
        
        sd = np.sqrt(sum_sqr_err / len(vals))
        #print(avg, sd)
        tmp_2 = []
        
        for val in tmp:
            float_val = float(val)
            if (float_val - avg) / sd < 2.7:
                tmp_2.append("key")
            else:
                tmp_2.append("parameter")
            
        pred = " ".join(tmp_2)
        
        res_file.write(pred + "\n")
        #labels_token = label_line.split(" ")
        #equal = True
        #for k in range(len(tmp_2)):
        #    if labels_token[k] == "key":
        #        if tmp_2[k] != "key":
        #            equal = False
        #            break
        #if equal:
        #    file_acc += 1
        #file_total += 1
        
        if pred == label_line:
            file_acc += 1
        #else:
            #print(pred + " : " + label_line)
        file_total += 1
    acc += file_acc
    total += file_total
    print(pred_file, len(preds), file_acc / file_total)
        
print(acc / total)
