from cluster import *
import math
import numpy as np

kmeans_model, data, all_logs, all_labels, uni_log_count_table = build_cluster("./msg-msg.txt", "./msg_labels.txt")
log_outputs = []
labels_outputs = []
ids_outputs = []


for i in range(len(kmeans_model.cluster_centers_)):
    f_log = open("./msg_clusters/cluster" + str(i) + ".txt", "w")
    log_outputs.append(f_log)
    
    f_label = open("./msg_clusters_labels/cluster" + str(i) + "_labels.txt", "w")
    labels_outputs.append(f_label)
    
    #f_count = open("./counts/cluster" + str(i) + ".txt", "w")
    #counts_outputs.append(f_count)

for i in range(len(data)):
    sen_vec = data[i]
    log = all_logs[i]
    labels = all_labels[i]
    
    min_dis = math.inf
    min_ind = None
    
    for j in range(len(kmeans_model.cluster_centers_)):
        center = kmeans_model.cluster_centers_[j]
        diff = sen_vec - center
        dis = np.sum(np.square(diff))
        
        if dis < min_dis:
            min_dis = dis
            min_ind = j
    #print(min_ind)
    log_out = log_outputs[min_ind]
    log_out.write(log)
    
    label_out = labels_outputs[min_ind]
    label_out.write(labels)

    
    #count_out = counts_outputs[min_ind]
    #count_out.write(str(uni_log_count_table[log]) + "\n")


for i in range(len(kmeans_model.cluster_centers_)):
    log_outputs[i].close()
    labels_outputs[i].close()

for i in range(len(kmeans_model.cluster_centers_)):
    f_id = open("./msg_clusters_ids/cluster" + str(i) + ".txt", "w")
    ids_outputs.append(f_id)

for i in range(len(data)):
    sen_vec = data[i]
    log = all_logs[i]
    labels = all_labels[i]
    
    min_dis = math.inf
    min_ind = None
    
    for j in range(len(kmeans_model.cluster_centers_)):
        center = kmeans_model.cluster_centers_[j]
        diff = sen_vec - center
        dis = np.sum(np.square(diff))
        
        if dis < min_dis:
            min_dis = dis
            min_ind = j
    
    id_out = ids_outputs[min_ind]
    id_out.write(str(i) + "\n")

for i in range(len(kmeans_model.cluster_centers_)):
    ids_outputs[i].close()


