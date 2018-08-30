import numpy as np
import os

res_dir = "./results/"
final_res_dir = "./final_results/"
log_dir = "./clusters/"

res_lst = sorted(os.listdir(res_dir))
output = None

for res in res_lst:
    res_lines = open(res_dir + res, "r").readlines()
    #log_lines = open(log_dir + res).readlines()
    
    #log_count_table = dict()
    #max_occur = 0
    #max_occcur_idx = 0
    
    #for i in range(len(log_lines)):
    #    log_line = log_lines[i]
    #    count = log_count_table.get(len(log_line), 0)
    #    count += 1
    #    log_count_table[len(log_line)] = count
    #    
    #    if count > max_occur:
    #        max_occcur_idx = i
    
    
    #const_param_arr = [0 for i in range(len(res_lines[max_occcur_idx].split(" ")))]
    const_param_arr = [0 for i in range(len(res_lines[0].split(" ")))]
    
    #param_count_table = dict()
    #for k in range(len(const_param_arr)):
        #param_count_table[k] = 0
    
    #thres = int(len(res_lines) / 500)
    
    for line in res_lines:
        tmp = line.strip("\n").split(" ")
        
        if len(tmp) != len(const_param_arr):
            continue
        
        for j in range(len(tmp)):
            if tmp[j] == "parameter":
                const_param_arr[j] = 1
            #if param_count_table[j] > thres:
             #   const_param_arr[j] = 1
    
    #final_res_arr = [0 for i in range(len(res_lines[max_occcur_idx].split(" ")))]
    final_res_arr = [0 for i in range(len(res_lines[0].split(" ")))]
    
    for j in range(len(const_param_arr)):
        if const_param_arr[j] == 1:
            final_res_arr[j] = "parameter"
        else:
            final_res_arr[j] = "key"
    
    final_res = " ".join(final_res_arr)
    output = open(final_res_dir + res, "w")
    
    for j in range(len(res_lines)):
        tmp = line.strip("\n").split(" ")
        
        if len(tmp) != len(const_param_arr): #or len(log_lines[j]) != len(log_lines[max_occcur_idx]):
            output.write(line)
        else:
            output.write(final_res + "\n")
    

