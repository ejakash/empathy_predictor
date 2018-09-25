To Run the code.
python main.py
To Re calibrate the hyper parameters
python main.py true

Handle_dataset.py takes care of the input reading and pre processing.

On initialization, the load_data method is invoked which read the data from the input file. This data is cleaned by changing the text values to numbers and filling missing values and scaling.

Then the values are stored in peopleData which could be accessed by main.

All the constants are saved in  constants.py. These values are used by handleData mostly.

The model trained after tuning hyper parameters is serialized and saved to trained_model file.

main has the work flow of training and evaluation.

In main, the main method is defined.
It loads the people data and run recursive feature elimination. The output is mapped to PCA fitting. This output is used to train the model and make the prediction. Sklearn DummyClassifier is used as a baseline classifier. It uses stratified strategy.

If tune_hyper_parameters is enabled, the model runs GridCVSearch with a range of parameters. This could take some time to run. It took about an hour in my machine. Once the tuning is complete, the model is saved to a file using pickle serializing module. In case the tune_hyper_parameters is disabled, we try to find this file and load the model from it. However, if the file could not be found due to some reason, we load the known tuned model.

Once the training is complete, we use compute_accuracy() to do the evaluations.

Libraries used are numpy, sklearn for models, PCA, RFE, preprocessing and dummt Classifier.
Pickle for data serialization. Pandas for data preprocessing.

