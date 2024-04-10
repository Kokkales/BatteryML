# batteryml/model/rul_predictor/lightgbm.py
from lightgbm import LGBMRegressor
# from sklearn import
from batteryml.builders import MODELS
from batteryml.models.sklearn_model import SklearnModel
@MODELS.register()
class LightgbmRULPredictor(SklearnModel):
    def __init__(self, *args, workspace: str = None, **kwargs):
        SklearnModel.__init__(self, workspace)
        self.model = LGBMRegressor(*args, **kwargs)