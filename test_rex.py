import numpy as np
from rex import train_simple_linear, train_and_evaluate, train_polynomial

def test_simple_linear_prediction():
    height = [[4.0],[5.0],[6.0],[7.0],[8.0],[9.0],[10.0],[11.0]]
    weight = [8, 10, 12, 14, 16, 18, 20, 22]

    model = train_simple_linear(height, weight)
    pred = model.predict([[12.0]])[0]
    
    # Expected ~24 (since it's a perfect line: weight = 2 * height)
    assert np.isclose(pred, 24.0, atol=0.1)

def test_train_and_evaluate_accuracy():
    X = [[4.0],[5.0],[6.0],[7.0],[8.0],[9.0],[10.0]]
    y = [8, 10, 12, 14, 16, 18, 20]

    model, score, X_test, y_test = train_and_evaluate(X, y)
    
    # Linear data should give R^2 score very close to 1
    assert score > 0.9

def test_polynomial_regression():
    X = [[4.0],[5.0],[6.0],[7.0],[8.0],[9.0],[10.0]]
    y = [16, 25, 36, 49, 64, 81, 100]  # y = x^2

    model = train_polynomial(X, y, degree=2)
    pred = model.predict([[11]])[0]
    
    # Expected ~121 for 11^2
    assert np.isclose(pred, 121.0, atol=1.0)
