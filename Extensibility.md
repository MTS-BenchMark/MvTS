## Development of MvTS

Thanks so much for your interest in MvTS!

### Workflow of MvTS

We first briefly introduce how `MvTS` completes the entire training process of a model. To be convenient, we take the LSTNET model for an example.

```
Python run_model.py --task single_step --model LSTNET --dataset solar
```

The directory of the main files participating in the process is shown as below.

```Python
[root@#######/MTS_Library/]# tree
.
├── mvts
│   └── config
│		└── data
│			└── solar.json
│   	├── model
│			└── single_step
│				└── LSTNET.json
│   	└── task_config.json
│   ├── data
│		└── dataset
│			└── single_step_dataset
│   			├── _init_.py
│				└── single_step_dataset.py
│   ├── model
│		└── single_step_model
│   		├── _init_.py
│			└── LSTNET.py
│   ├── executor
│		└── single_step_executor
│   		├── _init_.py
│			└── single_step_executor.py
│   ├── evaluator
│		└── evaluator.py
│   ├── pipeline
│		└── pipeline.py
│   ├── raw_data
 │		└── solar.h5
│   └── utils
├── run_model.py

```

**Key Steps:**

- `MvTS` first get the parameter configuration in Config, including the parameters of dataset and the training parameters of model. Meanwhile, `task_config.json` shows the models and datasets supported by the library, and specifies the corresponding  **dataloader** and **executor** for a specific model. (For `LSTNET`, the dataloader and executor are `single_step_dataset` and `single_step_executor`).
- The dataloader (`singel_step_dataset`) loads dataset `solar`, and performs normalization, segmentation and other operations, and finally is used to provide well processed data during the training process.
- `LSTNET.py` stores the original implementation of the model `LSTNET`.
- The executor(`single_step_executor`) completes the training and prediction process of `LSTNET`, and finally stores the well trained model at the specified location.
- **evaluator **provides a variety of evaluation functions to evaluate the performance of the model.

### New Datasets

If you want to extend new datasets into `MvTS`, then you can follow the methods below:

1. Record the rawdata, time, adjacency matrix(*if not available, then set it zero-matrix*) information of the dataset and integrate them in the h5 file. And put it into `./mvts/raw_data/`
2. Add `newdataset.json` in directory `./mvts/config/data/`.
3. Modify `./mvts/config/task_config.json` to support the new dataset.

### New Models

If you want to develop new datasets into `MvTS`, then you can follow the methods below (lets' take the model `LSTNET` for example):

1. Add the `LSTNET.py` into `./mvts/model/single_step_model/`, and modify the `./mvts/model/single_step_model/_inti_.py` to link the new model.
2. Add `LSTNET.json` into `./mvts/config/model/single_step/`.
3. Construct correct **dataloader** and **executor**. 
   - You can select a correct one from the existing modules, all you need is to make sure the methods that the module loads data and train the model  are the same.
   -  If there are not modules available, you can just add yours into `./mvts/data/dataset/single_step_dataset/` and  `./mvts/executor/single_step_executor/`. Meanwhile, remember to modify the `_init_.py` files in the two directories.
4. Finally, modify the `task_config.json` to support the new model and set the corresponding **dataloader** and **executor**.













