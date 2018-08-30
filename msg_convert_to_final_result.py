import numpy as np
import os

res_dir = "./msg_results/"
final_res_dir = "./msg_final_results/"

res_lst = sorted(os.listdir(res_dir))
output = None

for res in res_lst:
    res_lines = open(res_dir + res, "r").readlines()
    const_param_arr = [0 for i in range(len(res_lines[0].split(" ")))]
    
    for line in res_lines:
        tmp = line.strip("\n").split(" ")
        
        if len(tmp) != len(const_param_arr):
            continue
        
        for j in range(len(tmp)):
            if tmp[j] == "parameter":
                const_param_arr[j] = 1
    
    final_res_arr = [0 for i in range(len(res_lines[0].split(" ")))]
    
    for j in range(len(const_param_arr)):
        if const_param_arr[j] == 1:
            final_res_arr[j] = "parameter"
        else:
            final_res_arr[j] = "key"
    
    final_res = " ".join(final_res_arr)
    output = open(final_res_dir + res, "w")
    
    for line in res_lines:
        tmp = line.strip("\n").split(" ")
        
        if len(tmp) != len(const_param_arr):
            output.write(line)
        else:
            output.write(final_res + "\n")
    

