import pandas as pd
import joblib
import numpy as np

def test_scoring():
    """Test the scoring process with three assert statements"""
    
    model_path = './models/geyser_model.pkl'
    pipeline = joblib.load(model_path)
    
    
    df = pd.DataFrame({"eruptions": [1.5, 2.0, 3.0]})
    preds = pipeline.predict(df[["eruptions"]])
    
    # Assert 1: Number of predictions matches input
    assert len(preds) == len(df["eruptions"]), \
        "Number of predictions must match number of input values"
    
    # Assert 2: All predictions are finite (not NaN or inf)
    assert np.all(np.isfinite(preds)), \
        "All predicted values must be finite (not NaN or inf)"
    
    # Assert 3: All predictions are positive (greater than zero)
    assert np.all(preds > 0), \
        "All predicted values must be positive (greater than zero)"
