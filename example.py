import pandas as pd
from feature_engineering import create_features

# Synthetic examples
shots = pd.DataFrame({
    'shottype': ['layup', 'jumper', 'jumper'],
    'three': [False, False, True],
    'distance': [3.0, 14.0, 23.0],
    'locationx': [-42.0, -30.0, -25.0],
    'locationy': [0.0, 9.0, 22.0],
    'shooter_id': ['player_a', 'player_b', 'player_a'],
    'closestdefapproach': [
        '{5,4,3,2}',
        '{8,7,6,5}',
        '{6,5,4,3}'
    ]
})

result = create_features(shots)

print(result[
    ['shooter_id', 'shot_zone', 'shooter_zone', 'defender_closing']
].to_string(index=False))
