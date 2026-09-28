import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from synthetic_data import generate_synthetic_shots
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
feature_cols = [
    'distance',
    'shottype',
    'three',
    'closestdefdist',
    'locationx',
    'locationy',
    'contested',
    'gamestate',
    'dribblesbefore',
    'shot_zone',
    'num_contesters',
    'shotclock',
    'defender_closing',
    'defdist_1sec',
    'defdist_075sec',
    'defdist_05sec',
    'defdist_025sec',
    'shooter_zone'
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            'categorical',
            OneHotEncoder(handle_unknown='ignore'),
            ['shottype', 'shot_zone', 'gamestate']
        ),
        (
            'shooter_zone',
            TargetEncoder(),
            ['shooter_zone']
        )
    ],
    remainder='passthrough'
)

gb_model = HistGradientBoostingClassifier(
    max_leaf_nodes=64,
    learning_rate=0.05,
    max_iter=200,
    random_state=42
)

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', gb_model)
])

if __name__ == '__main__':
    shots = generate_synthetic_shots(n=500)
    X = shots[feature_cols]
    y = shots['outcome']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print('Starting model training...', flush=True)
    gb_pipeline.fit(X_train, y_train)
    print('Training complete!', flush=True)

    val_probs = gb_pipeline.predict_proba(X_val)[:, 1]

    print(f'Validation log loss: {log_loss(y_val, val_probs):.4f}')
    print('\nFirst five predicted make probabilities:')

    for probability in val_probs[:5]:
        print(f'{probability:.3f}')
