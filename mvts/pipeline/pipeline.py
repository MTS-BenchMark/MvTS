import os

from mvts.config import ConfigParser
from mvts.data import get_dataset
from mvts.utils import get_executor, get_model, get_logger, ensure_dir


def run_model(task=None, model_name=None, dataset_name= None, config_file=None,
              saved_model=True, train=True, other_args=None):
    
    # load config
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args)
    # logger
    logger = get_logger(config)
    logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}'.
                format(str(task), str(model_name), str(dataset_name)))
    # 加载数据集
    dataset = get_dataset(config)

    # 转换数据，并划分数据集
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    # 加载执行器
    model_cache_file = './mvts/cache/model_cache/{}_{}.m'.format(
        model_name, dataset_name)
    model = get_model(config, data_feature)

    executor = get_executor(config, model)
    # 训练
    if train or not os.path.exists(model_cache_file):
        executor.train(train_data, valid_data)
        if saved_model:
            executor.save_model(model_cache_file)
    else:
        executor.load_model(model_cache_file)
    # 评估，评估结果将会放在 cache/evaluate_cache 下
    executor.evaluate(test_data)
