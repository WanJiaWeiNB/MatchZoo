# import os
#
#
# filePath = "/deeperpool/lixs/sessionST/ad-hoc-udoc/"
# trainFilePath = filePath + "train/"
# testFilePath = filePath + "test/"
# testPathDir = os.listdir(testFilePath)
# print(testPathDir)
# for fileName in testPathDir:
#     with open(testFilePath + fileName, 'r') as testFile:
#         for i, line in enumerate(testFile.readlines()):
#             if(i == 0):
#                 continue
#             print(line.split('\t'))
#     break
# a = 1
# print(f'{a}+nihao')
import matchzoo as mz
import pandas as pd
task = mz.tasks.Ranking()
print(task)
train_raw = mz.datasets.toy.load_data(stage='train', task=task)
test_raw = mz.datasets.toy.load_data(stage='test', task=task)
type(train_raw)
print(train_raw.left)
print(train_raw.right)
print(train_raw.relation)
data = pd.DataFrame({
    'text_left': list('ARSAARSA'),
    'text_right': list('arstenus')
})
my_pack = mz.pack(data)
print(my_pack.frame())
print(list('AAAAA'))




