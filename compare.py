import os

final_res_dir = "./final_results/"
label_dir = "./clusters_labels/"
count_dir = "./counts/"
log_dir = "./clusters/"

final_res_lst = sorted(os.listdir(final_res_dir))

total = 0
acc = 0

all_total = 0
all_acc = 0

for res in final_res_lst:
    res_lines = open(final_res_dir + res).readlines()
    
    label_file = res.replace(".txt", "_labels.txt")
    label_lines = open(label_dir + label_file).readlines()
    
    count_lines = open(count_dir + res).readlines()
    
    log_lines = open(log_dir + res).readlines()
    
    file_total = 0
    file_acc = 0
    
    for i in range(len(res_lines)):
        if res_lines[i] == label_lines[i]:
            file_acc += 1
            all_acc += int(count_lines[i])
        elif int(count_lines[i]) > 10000:
            print(log_lines[i].split("\t")[0], int(count_lines[i]), res) 
        file_total += 1
        all_total += int(count_lines[i])
    
    #print(res, file_acc / file_total)
    
    acc += file_acc
    total += file_total
    
print(acc / total)
print(all_acc / all_total)
    
