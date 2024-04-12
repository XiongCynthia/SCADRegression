## Linear Regression with SCAD penalty

[SCADRegression](https://github.com/XiongCynthia/SCADRegression/blob/main/SCADRegression.py) is a class for fitting and predicting data on a linear model with smoothly clipped absolute deviation (SCAD) penalties.

### Usage

```python
from SCADRegression import SCADRegression
scad = SCADRegression()
scad.fit(x_train, y_train)
y_pred = scad.predict(x_test)
```

More example usages are included in [SCADRegression_examples.ipynb](https://github.com/XiongCynthia/SCADRegression/blob/main/SCADRegression_examples.ipynb), which additionally showcases performance comparisons of the model against the ElasticNet and SqrtLasso regularized regression models.
