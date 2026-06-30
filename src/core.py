# Rural credit helpers: seasonal adjustment and crop risk weights
import numpy as np

def seasonal_adjustment(predicted, season_factor):
    adjusted = []
    for p, s in zip(predicted, season_factor):
        adjusted.append(float(p * s))
    return adjusted

def crop_risk_weight(crop_type, crop_risks):
    weights = []
    for c in crop_type:
        weights.append(crop_risks.get(c, 1.0))
    return np.array(weights)