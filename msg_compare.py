import os

final_res_dir = "./msg_final_results/"
id_dir = "./msg_clusters_ids/"

msg_type_dir = "./msg_types/"
msg_type_label_dir = "./msg_types_labels/"
msg_type_count_dir = "./msg_types_counts/"
msg_type_final_res_dir = "./msg_types_final_res/"

final_res_lst = sorted(os.listdir(final_res_dir))

total = 0
acc = 0



for res in final_res_lst:
    res_lines = open(final_res_dir + res).readlines()

    id_lines = open(id_dir + res).readlines()
    
    file_total = 0
    file_acc = 0
    
    for i in range(len(res_lines)):
        res_line = res_lines[i].strip("\n")
        type_id = id_lines[i].strip("\n")
        
        msg_type_logs = open(msg_type_dir + type_id + ".txt").readlines()
        msg_type_labels = open(msg_type_label_dir + type_id + ".txt").readlines()
        msg_type_counts = open(msg_type_count_dir + type_id + ".txt").readlines()
        msg_type_results = open(msg_type_final_res_dir + type_id + ".txt").readlines()
        
        for j in range(len(msg_type_logs)):
            msg_log = msg_type_logs[j].strip("\n")
            msg_label = msg_type_labels[j].strip("\n")
            msg_count = int(msg_type_counts[j])
            msg_final_res = msg_type_results[j].strip("\n")
            
            tmp = msg_log.split(" ")
            tmp2 = []
            
            r_tmp = res_line.split(" ")
            msg_type_final_tmp = msg_final_res.split(" ")
            
            for k in range(len(tmp)):
                #print(res, type_id,j,k, msg_log)
                if msg_type_final_tmp[k] != "parameter" and r_tmp[k] == "parameter":
                    tmp2.append("parameter")
                else:
                    tmp2.append(msg_type_final_tmp[k])
            
            last_res = " ".join(tmp2)
            if last_res == msg_label:
                file_acc += msg_count
            elif msg_count > 10000:
                print(msg_count, msg_log, res, type_id) 
            file_total += msg_count
    
    #print(res, file_acc / file_total)
    
    acc += file_acc
    total += file_total
    
print(acc / total)

    
