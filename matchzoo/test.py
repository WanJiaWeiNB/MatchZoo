import matchzoo as mz
raw_data = mz.datasets.toy.load_data()
preprocessor = mz.preprocessors.BasicPreprocessor(
    fixed_length_left=10,
    fixed_length_right=40,
    remove_stop_words=True)
processed_data = preprocessor.fit_transform(raw_data)
data_generator = DPoolDataGenerator(processed_data, 3, 10,
     batch_size=3, shuffle=False)