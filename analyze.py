import os

final_res_dir = "./final_results/"
cluster_dir = "./clusters/"
label_dir = "./clusters_labels/"
count_dir = "./counts/"

all_msg_type = dict()
all_msg_info = dict()

file_lst = sorted(os.listdir(final_res_dir))

for f in file_lst:
    res_lines = open(final_res_dir + f).readlines()
    cluster_lines = open(cluster_dir + f).readlines()
    label_lines = open(label_dir + f.replace(".txt", "_labels.txt")).readlines()
    count_lines = open(count_dir + f).readlines()
    
    print(f, len(res_lines))
    
    for j in range(len(res_lines)):
        res_line = res_lines[j].strip("\n")
        cluster_line = cluster_lines[j].strip("\n").split("\t")[0]
        label_line = label_lines[j].strip("\n")
        count = int(count_lines[j])
        
        r_tmp = res_line.split(" ")
        c_tmp = cluster_line.split(" ")
        
        if len(r_tmp) != len(c_tmp):
            continue
        
        for k in range(len(r_tmp)):
            if r_tmp[k] == "parameter":
                c_tmp[k] = "*"
        
        
        msg_type = " ".join(c_tmp)
        if msg_type not in all_msg_type:
            all_msg_type[msg_type] = label_line
        
        if msg_type not in all_msg_info:
            all_msg_info[msg_type] = []
        
        all_msg_info[msg_type].append((cluster_line, label_line, res_line, count))
        

all_msg_type_lst = sorted(all_msg_type.keys(), key=lambda msg_type: len(msg_type))

print(len(all_msg_type_lst))
output = open("msg-msg.txt", "w")
label_output = open("msg_labels.txt", "w")

for i in range(len(all_msg_type_lst)):
    msg_type = all_msg_type_lst[i]
    output.write(msg_type + "\t" + msg_type + "\n")
    label = all_msg_type[msg_type]
    label_output.write(label + "\n")
    
    
    msg_type_output = open("./msg_types/" + str(i)+".txt", "w")
    msg_type_label_output = open("./msg_types_labels/" + str(i)+".txt", "w")
    msg_type_res_output = open("./msg_types_final_res/" + str(i)+".txt", "w")
    msg_type_counts_output = open("./msg_types_counts/" + str(i)+".txt", "w")
    
    tup_lst = all_msg_info[msg_type]
    
    for tup in tup_lst:
        msg_type_output.write(tup[0] + "\n")
        msg_type_label_output.write(tup[1] + "\n")
        msg_type_res_output.write(tup[2] + "\n")
        msg_type_counts_output.write(str(tup[3]) + "\n")
    
    msg_type_output.close()
    msg_type_label_output.close()
    msg_type_res_output.close()
    msg_type_counts_output.close()
    
    
    
output.close()
label_output.close()







