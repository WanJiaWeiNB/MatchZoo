# -*- coding: UTF-8 -*-
import matchzoo as mz
import pandas as pd
import os
import pickle as pkl
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# str = train/ test/ valid
# test is special with human
def get_train_data(str1 = "train/",all_data = True):
    pd.set_option('display.width', None)
    inputData = []
    filePath = "/deeperpool/lixs/sessionST/ad-hoc-udoc/"
    testFilePath = filePath + str1
    testPathDir = os.listdir(testFilePath)
    for j,fileName in enumerate(testPathDir):
        with open(testFilePath + fileName, 'r') as testFile:
            for i, line in enumerate(testFile.readlines()):
                if (i == 0):
                    continue
                inputData.append(line.split('\t'))
        if all_data == False and j > 2:
            break

    input_data_frame = pd.DataFrame({
        "text_left": [x[2] for x in inputData],
        "text_right": [x[3] for x in inputData],
        "label": [x[5] for x in inputData]
    })
    # my_pack = mz.pack(input_data_frame)
    # with open("Mtest2", "w") as f:
    #     f.write(str(my_pack.frame()))
    return input_data_frame

# TODO: check the embedding result is right
# return word2id id2word embedding
# usage embedding_layer(Variable(torch.LongTensor([word2id['厦门市']])))
def get_embedding():
    embedding_layer = nn.Embedding(92402, 50)
    fileAddr1 = '/deeperpool/lixs/sessionST/GraRanker/data/vocab.dict.9W.pkl'
    fileAddr2 = '/deeperpool/lixs/sessionST/GraRanker/data/emb50_9W.pkl'
    f = open(fileAddr1, 'rb')
    word2id1, dic2word = pkl.load(f, encoding='bytes')
    word2id = {}
    f.close()
    for k, v in word2id1.items():
        try:
            t = bytes(k).decode('utf8')
        except:
            continue
        word2id[t] = v
    f = open(fileAddr2, 'rb')
    pre_word_embeds = pkl.load(f, encoding='bytes')
    f.close()
    print('pre_word_embeds size: ' + str(pre_word_embeds.shape))
    embedding_layer.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
    return embedding_layer,word2id, dic2word

def trans_form_to_id(str1,word2id):
    import re
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")  # 匹配不是中文、大小写、数字的其他字符
    str1 = cop.sub('', str1)  # 将string1中匹配到的字符替换成空字符
    list1 = str1.split(' ')
    for i,j in enumerate(list1):
        if j in word2id.keys():
            list1[i] = word2id[j]
        else:
            list1[i] = 1
        if(i > 1):
            print("lu")
    return list1

def my_data_process(data_pack, word2id):
    for i,j in enumerate(data_pack['text_left']):
        data_pack['text_left'][i] = trans_form_to_id(j,word2id)


if __name__ == '__main__':
    train_data_pack = get_train_data(all_data=False)
    valid_data_pack = get_train_data('valid/',False)
    embedding_layer,word2id, dic2word = get_embedding()
    print("-----------data is load------------")
    # Preprocess your input data in three lines of code, keep track parameters to be passed into the model.
    print(train_data_pack)
    my_data_process(train_data_pack, word2id)
    my_data_process(valid_data_pack, word2id)
    print(train_data_pack)
    print(valid_data_pack)

    data = pd.DataFrame(data=[[0, 1], [2, 3]], index=['A', 'B'])
    embedding = mz.Embedding(data)
    matrix = embedding.build_matrix({'A': 2, 'B': 1})
    matrix.shape == (3, 2)

    print("---------preprocess is ok-----------")
    # Make use of MatchZoo customized loss functions and evaluation metrics:
    ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=4))
    ranking_task.metrics = [
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
        mz.metrics.MeanAveragePrecision()
    ]
    print("-----------ranktask is prepared----------")
    # Initialize the model, fine-tune the hyper-parameters.
    model = mz.models.DSSM()
    model.params['input_shapes'] = preprocessor.context['input_shapes']
    model.params['task'] = ranking_task
    model.params['mlp_num_layers'] = 3
    model.params['mlp_num_units'] = 300
    model.params['mlp_num_fan_out'] = 128
    model.params['mlp_activation_func'] = 'relu'
    model.guess_and_fill_missing_params()
    model.build()
    model.compile()
    print("--------------model init------------------")
    # Generate pair-wise training data on-the-fly, evaluate model performance using customized callbacks on validation data.
    train_generator = mz.PairDataGenerator(train_processed, num_dup=1, num_neg=4, batch_size=64, shuffle=True)
    valid_x, valid_y = valid_processed.unpack()
    print("-----------generator is ok-----------------")
    evaluate = mz.callbacks.EvaluateAllMetrics(model, x=valid_x, y=valid_y, batch_size=64)

    history = model.fit_generator(train_generator, epochs=20, callbacks=[evaluate], workers=5,
                                  use_multiprocessing=False)
    print("---------------done!-----------------------")

