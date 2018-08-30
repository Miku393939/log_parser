from sklearn.cluster import KMeans
import numpy as np
from gensim.models import Word2Vec
LOGLEVEL = set(["INFO", "FATAL", "ERROR", "WARNING", "SEVERE", "FAILURE"])
import time

def build_cluster(file_name, label_file):
    sentences = []

    # This is log levels, and they are subject to change by dataset
    
    
    all_logs = open(file_name, "r").readlines()[:3516040]
    all_labels = open(label_file).readlines()[:3516040]
    
    
    all_uni_logs = []
    all_uni_labels = []
    
    uni_log_count_table = dict()
    
    for i in range(len(all_logs)):
        log = all_logs[i]
        if log not in uni_log_count_table:
            all_uni_logs.append(log)
            all_uni_labels.append(all_labels[i])
            uni_log_count_table[log] = 1
        else:
            uni_log_count_table[log] += 1
        
    all_logs = all_uni_logs
    all_labels = all_uni_labels
    
    print(len(all_logs))
    log_types_table = dict()
    
    print(len(all_logs))
    c = 0
    for log in all_logs:
        log = log.strip().strip("\n")

        tmp = log.split("\t")[0].split(" ")
    
        if tmp[-1] in LOGLEVEL or tmp[-1].find('FATAL')>=0 or tmp[-1].find('INFO')>=0:
            continue
    
        start=0
        while len(tmp)>start and (tmp[start] not in LOGLEVEL):
            start += 1
    
        if len(tmp) > start and tmp[start] in LOGLEVEL:
            start += 1
            log = " ".join(tmp[start:])  # not including LOGLEVEL here
        elif len(tmp) <= start:
            #print ("tmp[-1]", tmp[-1], "no log level: ", log)
            start = 0
    
        if log.strip().isdigit():
            continue
    
    
    
        sentence = []
        for j in range(start, len(tmp)):
            sentence.append(tmp[j])
    
        sentences.append(sentence)
        c += 1
        print(c)
    
    print("finish reading sentences", len(sentences))


    vec_size = 400
    word2vec_model = Word2Vec(sentences, size = vec_size, min_count=1)
    print("finish constructing word2vec")
    #va = word2vec_model["core.5962"]
    #vb = word2vec_model["core.5705"]
    #vc = word2vec_model["interrupt"]
    #print(np.sum(np.absolute(va-vb)))
    #print(np.sum(np.absolute(va-vc)))
    #time.sleep(3)

    data = np.zeros((len(sentences), vec_size))

    for i in range(len(sentences)):
        print(i)
        sentence = sentences[i]
        sentence_vec = np.zeros(vec_size)
    
        for word in sentence:
            sentence_vec += word2vec_model[word]
        sentence_vec /= len(sentence)
        
        
        data[i] = sentence_vec

    kmeans_model = KMeans(n_clusters=350, n_init=200, max_iter=600, random_state=0).fit(data)

    #print(kmeans_model.cluster_centers_)
    assert len(kmeans_model.cluster_centers_) == 350
    #print(type(kmeans_model.cluster_centers_[0]))
    
    return kmeans_model, data, all_logs, all_labels, uni_log_count_table
