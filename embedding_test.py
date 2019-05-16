# -*- coding=utf-8 -*-
import pickle as pkl
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy

embedding_layer = nn.Embedding(90000, 50)
fileAddr1 = '/deeperpool/lixs/sessionST/GraRanker/data/vocab.dict.9W.pkl'
fileAddr2 = '/deeperpool/lixs/sessionST/GraRanker/data/emb50_9W.pkl'
f = open(fileAddr1,'rb')
word2id1,dic2word = pkl.load(f, encoding='bytes')
word2id = {}
f.close()
for k,v in word2id1.items():
    try:
        t = bytes(k).decode('utf8')
    except:
        continue
    word2id[t] = v
f = open(fileAddr2,'rb')
pre_word_embeds = pkl.load(f, encoding='bytes')
f.close()
print(pre_word_embeds[38488])
print('pre_word_embeds size: '+str(pre_word_embeds.shape))
embedding_layer.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
with open('embedding_word2id.txt','w', encoding='utf-8') as f:
    for k,v in word2id.items():
        f.write(str(k) + ' '+ str(v) +'\t')

xiamen = (embedding_layer(Variable(torch.LongTensor([word2id['厦门市']])))).detach().numpy()
gulangsu = embedding_layer(Variable(torch.LongTensor([word2id['鼓浪屿']]))).detach().numpy()
houwei = embedding_layer(Variable(torch.LongTensor([word2id['后卫']]))).detach().numpy()
zhongfeng = embedding_layer(Variable(torch.LongTensor([word2id['中锋']]))).detach().numpy()
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
# print(cos(xiamen,gulangsu))
# print(cos(xiamen,zhongfeng))
# print(cos(xiamen,houwei))
# print(cos(houwei,zhongfeng))
print(zhongfeng)
print(numpy.sqrt(numpy.sum(numpy.square(xiamen - gulangsu))))
print(numpy.sqrt(numpy.sum(numpy.square(xiamen - zhongfeng))))
print(numpy.sqrt(numpy.sum(numpy.square(xiamen - houwei))))
print(numpy.sqrt(numpy.sum(numpy.square(houwei - zhongfeng))))
print(numpy.sqrt(numpy.sum(numpy.square(houwei - gulangsu))))
# print(dic2word)

