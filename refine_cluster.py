import numpy as np
import os
import math

cluster_dir = "./clusters/"
label_dir = "./clusters_labels/"
refined_cluster_dir = "./refined_clusters/"

cluster_lst = sorted(os.listdir(cluster_dir))

for i in range(len(cluster_lst)):
    cluster_file = cluster_lst[i]
    
    cluster_lines = open(cluster_dir + cluster_file).readlines()
    
    label_file = cluster_file.replace(".txt", "_labels.txt")
    
    labels = open(label_dir + label_file).readlines()
    
    standard = cluster_lines[0].strip("\n").split("\t")[0].split(" ")
    
    max_diff = 0
    
    diff_lst = [math.inf]
    
    for j in range(1, len(cluster_lines)):
        tmp = cluster_lines[j].strip("\n").split("\t")[0].split(" ")
        if len(tmp) != len(standard):
            diff_lst.append(math.inf)
            continue
        
        diff = 0
        
        for k in range(len(tmp)):
            if tmp[k] != standard[k]:
                diff += 1
        diff_lst.append(diff)
        if diff > max_diff:
            max_diff = diff
    
    thres = max(0, max_diff - 3)
    
    refined_cluster_out = open(refined_cluster_dir + cluster_file, "w")
    
    refined_cluster_out.write(cluster_lines[0])
    
    for j in range(1, len(cluster_lines)):
        if diff_lst[j] > thres:
             refined_cluster_out.write(cluster_lines[j])
    
    refined_cluster_out.close()
    
    


