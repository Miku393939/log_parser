from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import math
import string
import re
import random
import glob

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda")
print(device)

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        words = sentence.split(" ")
        for word in words:
            self.addWord(word)
        return len(words)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = s.lower().strip()
    #s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, f_name, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(f_name, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs



MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, f_name, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, f_name, reverse)
    print("Read %s sentence pairs" % len(pairs))
    #pairs = filterPairs(pairs)
    #print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    
    MAX = 0
    for pair in pairs:
        length = input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        if length > MAX:
            MAX = length
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs, MAX

cluster_lst = glob.glob("./msg_clusters/*.txt")
input_lang_lst = []
output_lang_lst = []
pairs_lst = []
max_lst = []

MAX_all = -math.inf

for cluster in cluster_lst:
    input_lang, output_lang, pairs, MAX = prepareData('ori', 'recon', cluster)
    
    if MAX_all < MAX:
        MAX_all = MAX
    
    input_lang_lst.append(input_lang)
    output_lang_lst.append(output_lang)
    pairs_lst.append(pairs)
    max_lst.append(MAX)

# + 1 because appending EOS token during training
MAX_LENGTH = MAX_all + 1
print(MAX_LENGTH)

ori_cluster_lst = glob.glob("./msg_clusters/*.txt")
ori_pairs_lst = []

for ori_cluster in ori_cluster_lst:
    input_lang, output_lang, ori_pairs, MAX = prepareData('ori', 'recon', ori_cluster)
    ori_pairs_lst.append(ori_pairs)

class Embedding_layer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Embedding_layer, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.M = torch.randn(hidden_size, hidden_size, device=device)
        
    def forward(self, input_tensor):
        embeddings = torch.zeros(input_tensor.shape[0], self.hidden_size, device=device)
        weights = torch.zeros(input_tensor.shape[0], device=device)
        total = torch.zeros(1, self.hidden_size, device=device)
        
        for i in range(input_tensor.shape[0]):
            embedding = self.embedding(input_tensor[i])
            total = total + embedding.view(1, -1)
            embeddings[i] = embeddings[i] + embedding
        
        ys = total / input_tensor.shape[0]
        
        for i in range(input_tensor.shape[0]):
            weights[i] = weights[i] + torch.mm(torch.mm(embeddings[i].view(1, -1), self.M), torch.t(ys))
        
        return embeddings, weights
        
        

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, bidirectional=True)

    def forward(self, input_tensor, hidden, attention):
        embedded = input_tensor.view(1, 1, -1)
        embedded = embedded * attention
        output = embedded
        #print("embed shape", embedded.shape)
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.randn((2, 1, self.hidden_size), device=device), torch.randn((2, 1, self.hidden_size), device=device))


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=5, bidirectional=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return (torch.randn((10, 1, self.hidden_size), device=device), torch.randn((10, 1, self.hidden_size), device=device))


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1, bidirectional=True)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)
            
        #print("1", attn_weights.shape)
        
        #print("4", encoder_outputs.shape)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        #print("2", attn_applied.shape)
        
        
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        
        #print("3", output.shape)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return (torch.randn((2, 1, self.hidden_size), device=device), torch.randn((2, 1, self.hidden_size), device=device))


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, embed, encoder, decoder, embed_optimizer, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    
    embed_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size * 2, device=device)

    loss = 0
    
    embeddings, weights = embed(input_tensor)
    max_v = torch.max(torch.abs(weights))
    weights = weights / max_v
    all_weight = torch.sum(torch.exp(weights)[:len(weights) - 1])
    weights = torch.exp(weights) / all_weight
    
    
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            embeddings[ei], encoder_hidden, weights[ei])
        #print(encoder_outputs.shape, encoder_output.shape)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    
    embed_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(embed, encoder, decoder, pairs, n_iters, print_every=1000, plot_every=100, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    embed_optimizer = optim.Adam(embed.parameters(), lr=learning_rate)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        #if iter % 1 == 0:
        #    print(iter)
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, embed, encoder,
                     decoder, embed_optimizer, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))




import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)



def evaluate(embed, encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, 2 * encoder.hidden_size, device=device)
        
        embeddings, weights = embed(input_tensor)
        max_v = torch.max(torch.abs(weights))
        weights = weights / max_v
        all_weight = torch.sum(torch.exp(weights)[:len(weights) - 1])
        weights = torch.exp(weights) / all_weight
        
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(embeddings[ei],
                                                     encoder_hidden, weights[ei])
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            #print(decoder_output.shape)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                #decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
        #print(decoded_words)
        return decoded_words, weights


def evaluateRandomly(embed, encoder, decoder, pairs, outfile, maximum_length, n=10):
    acc = 0
    total = 0
    
    all_attentions = torch.zeros([len(pairs), maximum_length])
    
    length_lst = []
    
    count = 0
    for pair in pairs:
        #print('>', pair[0])
        #print('=', pair[1])
        output_words, attentions = evaluate(embed, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        #print('<', output_words)
        
        
        #att_arr = attentions.data.cpu().numpy()
        #for att in att_arr[: len(attentions) - 1]:
        #    out_file.write(str(att) + " ")
        #out_file.write("\n")
        length_lst.append(len(attentions) - 1)
        for j in range(len(attentions)):
            all_attentions[count][j] = attentions[j]
        count += 1
        
        correct = True
        label_words = pair[1].split(" ")
        #print(label_words)
        
        #print('')
        if len(label_words) != len(output_words):
            correct = False
        else:
            for i in range(len(label_words)):
                if label_words[i] != output_words[i]:
                    correct = False
                    break
        if correct:
            acc += 1
        total += 1
    print("training accuracy", acc / total)
    
    out_file = open(outfile, "w")
    all_attentions_data = all_attentions.data.cpu().numpy()
    for i in range(len(length_lst)):
        att_arr = all_attentions_data[i]
        length = length_lst[i]
        
        for att in att_arr[: length]:
            out_file.write(str(att) + " ")
        out_file.write("\n")    
    out_file.close()

hidden_size = 512

for i in range(len(cluster_lst)):
    input_lang = input_lang_lst[i]
    output_lang = output_lang_lst[i]
    pairs = pairs_lst[i]
    ori_pairs = ori_pairs_lst[i]
    maximum = max_lst[i] + 1
    
    embed1 = Embedding_layer(input_lang.n_words, hidden_size).to(device)
    encoder1 = EncoderRNN(hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
    epoch = 0
    
    if len(pairs) < 50:
        epoch = 150
    elif len(pairs) < 800:
        epoch = int(len(pairs) / 2)
    else:
        epoch = int(len(pairs) / 7)
    
    
    trainIters(embed1, encoder1, attn_decoder1, ori_pairs, epoch, print_every=1000)
    
    cluster = cluster_lst[i]
    outfile = cluster.replace("clusters", "outputs")
    
    evaluateRandomly(embed1, encoder1, attn_decoder1, pairs, outfile, maximum)
    
