# import matchzoo and prepare input data
import matchzoo as mz
# import os
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
train_pack = mz.datasets.wiki_qa.load_data('train', task='ranking')
valid_pack = mz.datasets.wiki_qa.load_data('dev', task='ranking')
predict_pack = mz.datasets.wiki_qa.load_data('test', task='ranking')
print("^^load data")
print(train_pack)
# Preprocess your input data in three lines of code, keep track parameters to be passed into the model.
preprocessor = mz.preprocessors.DSSMPreprocessor()
train_processed = preprocessor.fit_transform(train_pack)
valid_processed = preprocessor.transform(valid_pack)
print(train_processed.left.head())

print("^^pack data")

# Make use of MatchZoo customized loss functions and evaluation metrics:
ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=4))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print("^^rank task")

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
print("^^ model init")

# Generate pair-wise training data on-the-fly, evaluate model performance using customized callbacks on validation data.
train_generator = mz.PairDataGenerator(train_processed, num_dup=1, num_neg=4, batch_size=64, shuffle=True)
print("^^ gen")
valid_x, valid_y = valid_processed.unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=valid_x, y=valid_y, batch_size=64)

history = model.fit_generator(train_generator, epochs=20, callbacks=[evaluate], workers=5, use_multiprocessing=False)
print("^^ history")