# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
#
# word_to_ix = {'hello': 0, 'world': 1}
# embeds = nn.Embedding(2, 5)
# hello_idx = torch.LongTensor([word_to_ix['hello']])
# print(hello_idx)
# hello_idx = Variable(hello_idx)
# print(hello_idx)
# hello_embed = embeds(hello_idx)
# print(hello_embed)
import pandas as pd

inputData = [[1,2,3],[4,5,6],[7,8,9]]
input_data_frame = pd.DataFrame({
        "text_left": [x[0] for x in inputData],
        "text_right": [x[1] for x in inputData],
        "label": [x[2] for x in inputData]
    })
for i,j in enumerate(input_data_frame["text_left"]):
    input_data_frame["text_left"][i] = j + 1
print(input_data_frame["text_left"][0])
