import sys
from batteryml.pipeline import Pipeline
from batteryml.visualization.plot_helper import plot_capacity_degradation, plot_cycle_attribute, plot_result

pipeline = Pipeline(config_path='configs/baselines/nn_models/lstm/hust.yaml',workspace='workspaces')

model, dataset = pipeline.train(device='cpu', skip_if_executed=False)

train_prediction = model.predict(dataset, data_type='train')
train_loss = dataset.evaluate(train_prediction, 'RMSE', data_type='train')
test_prediction = model.predict(dataset, data_type='test')
test_loss = dataset.evaluate(test_prediction, 'RMSE', data_type='test')
print(test_prediction)
print(train_prediction)
print(f'RMSE: Train {train_loss:.2f}, test {test_loss:.2f}')
# feature_importance = model.feature_importances_

# Print feature importance scores
# for i, importance in enumerate(feature_importance):
#     print(f'Feature {i}: Importance: {importance}')

pipeline.evaluate(model=model, dataset=dataset, skip_if_executed=False)
