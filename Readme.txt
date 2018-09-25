To Run the code.
python main.py
To Re calibrate the hyper parameters
python main.py true

HandleData.py takes care of the input reading and pre processing.
All the constants are saved in  constants.py. These values are used by handleData mostly.

The model trained after tuning hyper parameters is serialized and saved to trained_model file.

main has the work flow of training and evaluation.