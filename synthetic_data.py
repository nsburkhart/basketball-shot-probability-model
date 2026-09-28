import numpy as np
import pandas as pd

from feature_engineering import create_features


def generate_synthetic_shots(n=500, random_state=42):
    rng = np.random.default_rng(random_state)

    # Generate made-up shot information.
    shottype = rng.choice(
        ['layup', 'dunk', 'floater', 'jumper', 'post'],
        size=n,
        p=[0.25, 0.05, 0.10, 0.50, 0.10]
    )

    three = rng.random(n) < 0.35
    three[np.isin(shottype, ['layup', 'dunk'])] = False

    # Keep layups and dunks near the rim.
    distance = np.where(
        np.isin(shottype, ['layup', 'dunk']),
        rng.uniform(1, 5, n),
        np.where(
            three,
            rng.uniform(22, 30, n),
            rng.uniform(5, 21, n)
        )
    )

    locationx = -distance
    locationy = rng.uniform(-25, 25, n)
    closestdefdist = rng.uniform(1, 10, n)

    # Invent four defender-distance measurements.
    defdist_1sec = closestdefdist + rng.uniform(0, 4, n)
    defdist_075sec = closestdefdist + rng.uniform(0, 3, n)
    defdist_05sec = closestdefdist + rng.uniform(0, 2, n)
    defdist_025sec = closestdefdist + rng.uniform(0, 1, n)

    closestdefapproach = [
        f'{{{a:.2f},{b:.2f},{c:.2f},{d:.2f}}}'
        for a, b, c, d in zip(
            defdist_1sec,
            defdist_075sec,
            defdist_05sec,
            defdist_025sec
        )
    ]

    shots = pd.DataFrame({
        'shottype': shottype,
        'three': three,
        'distance': distance,
        'locationx': locationx,
        'locationy': locationy,
        'shooter_id': rng.choice(
            ['player_a', 'player_b', 'player_c', 'player_d'],
            size=n
        ),
        'closestdefapproach': closestdefapproach,
        'closestdefdist': closestdefdist,
        'contested': closestdefdist < 4,
        'gamestate': rng.choice(
            ['regular', 'transition', 'late_clock'],
            size=n
        ),
        'dribblesbefore': rng.integers(0, 7, size=n),
        'num_contesters': rng.integers(0, 4, size=n),
        'shotclock': rng.uniform(1, 24, n)
    })

    # Illustrative outcomes, not real NBA shot probabilities.
    make_prob = (
        0.65
        - 0.012 * shots['distance']
        + 0.015 * shots['closestdefdist']
        - 0.05 * shots['contested'].astype(int)
    )
    make_prob = np.clip(make_prob, 0.05, 0.95)
    shots['outcome'] = rng.binomial(1, make_prob)

    return create_features(shots)


if __name__ == '__main__':
    shots = generate_synthetic_shots()
    print(shots[
        ['shottype', 'distance', 'shot_zone', 'defender_closing', 'outcome']
    ].head())
    print(f'\nGenerated {len(shots)} synthetic shots.')
