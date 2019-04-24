import os


filePath = "/deeperpool/lixs/sessionST/ad-hoc-udoc/"
trainFilePath = filePath + "train/"
testFilePath = filePath + "test/"
testPathDir = os.listdir(testFilePath)
print(testPathDir)
for fileName in testPathDir:
    with open(testFilePath + fileName, 'r') as testFile:
        for i, line in enumerate(testFile.readlines()):
            if(i == 0):
                continue
            print(line.split('\t'))
    break

# print(f'{a}+nihao')