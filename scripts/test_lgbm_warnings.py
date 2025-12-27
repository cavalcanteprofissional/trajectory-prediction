# Short test to train LightGBM on dummy data and observe warnings
import numpy as np
from models.model_factory import ModelFactory
from training.cross_validation import CrossValidator

np.random.seed(42)
# create dummy features with some constant columns to simulate real case
X = np.random.randn(500, 10)
# add constant column
const_col = np.ones((500,1)) * 3.14
X = np.hstack([X, const_col])
# target lat/lon around a city
y = np.random.randn(500,2) * 0.01 + np.array([-23.55, -46.63])

factory = ModelFactory(n_samples=500)
model = factory.create_model('LightGBM')

validator = CrossValidator(n_splits=3, random_state=42, shuffle=True)
res = validator.cross_validate(model, X, y, model_name='LightGBM_test', verbose=True)
print('Result:', res['mean_error'], 'km')
