## A BenchMark for Multivariate Time Series 

### Introduction to this BenchMark

This library aims to provide an open source library for multivariate time series prediction. Consulting https://github.com/LibCity/Bigscity-LibCity, We simply divide the whole process into the following six parts: config, data, evaluator, executor, model and raw_data.

###### config

The parameters of models and datasets.

###### data

This part aims to provide the <u>dataloader</u> and <u>data features</u> of the dataset for the model and model executor.

###### evaluator

###### executor

The executor of a specific model. 

###### model

The set of different models.

###### raw_data

The original dataset.

### How to use

eg. python run_model.py --task single_step --model LSTNET  --dataset traffic

the type of dataset is up to the demand of specific model, for example, traffic.txt, traffic.hdf5, traffic.npz...

### How to add model

If you want to add some models to this benchmark, you should take the following items into consider.

###### config

Check the config of dataset and add the configuration of your model into folder "model".

Modify the file "task_config.json"， add your model and configuration in the appropriate place in this file.

Make sure that you can get the whole configuration of the prediction through this part.

###### data

You should add a dataset file into the folder "dataset". Simply name it "model_name_dataset", put it into fittable folder(single_step_dataset or multi_step_dataset). And this file should provide two functions, one is "get_data" and the other is "get_data_feature".

Then modify the file "__init__.py" which is under folder "dataset".

###### model

Add your model into folder "model". It makes no sense whether your model inherits the father model file "abstract_model.py". But you should make sure your model can "forward" ~~.

Then modify the "__init__.py" which is in folder "model".

###### executor

It's better your model executor inherits the father executor "abstract_executor" and implement the four functions in this file.

Note that one func is named as "evaluate", which indicates you should also complete the "evaluate" process in this part. So it's up to you whether to establish a part "evaluator" as an element of the model executor. Simply make sure your executor can finish the four masks fluently.

Modify the file "__init__.py" in the folder "executor".

###### raw_data

If you need to run other datasets, you can add it into folader "raw_data".























