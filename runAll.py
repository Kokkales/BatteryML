from batteryml.pipeline import Pipeline
from batteryml.visualization.plot_helper import plot_capacity_degradation, plot_cycle_attribute, plot_result

import pandas as pd
import os
import optuna

config_dir = 'configs/baselines/'
data = 'hust'
result=[]
# for method_type in os.listdir(config_dir):
method_type = 'sklearn'
for method in os.listdir(os.path.join(config_dir, method_type)):
    print(method)
    config_path = os.path.join(config_dir, method_type, method, f'{data}.yaml')
    pipeline = Pipeline(config_path=config_path,
                workspace=f'workspaces/{method}')
    model, dataset = pipeline.train(device='cpu', skip_if_executed=False)
    train_prediction = model.predict(dataset, data_type='train')
    train_loss = dataset.evaluate(train_prediction, 'RMSE', data_type='train')
    test_prediction = model.predict(dataset, data_type='test')
    test_loss = dataset.evaluate(test_prediction, 'RMSE', data_type='test')
    result.append([method, train_loss, test_loss])

res = pd.DataFrame(data=result, columns=['method', 'train_RMSE', 'test_RMSE'])

print(res.head)