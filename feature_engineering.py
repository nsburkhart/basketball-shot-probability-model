import numpy as np
def create_features(df):
  # creating zones for all shots based on distance and location
    conditions = [

        # HEAVE
        (df['shottype'] == 'heave'),

        # AT RIM
        (df['three'] == False) &
        (df['distance'] <= 5),

        # LOW POST
        (df['three'] == False) &
        (df['locationx'] >= -43) &
        (df['locationx'] <= -35) &
        (df['locationy'].abs() >= 8.3) &
        (df['locationy'].abs() <= 15),

        # ELBOW
        (df['three'] == False) &
        (df['locationx'] > -35) &
        (df['locationx'] <= -25) &
        (df['locationy'].abs() >= 6) &
        (df['locationy'].abs() <= 12),

        # HIGH POST
        (df['three'] == False) &
        (df['locationx'] > -35) &
        (df['locationx'] <= -25) &
        (df['locationy'].abs() < 6),

        # SHORT 2
        (df['three'] == False) &
        (df['distance'] > 5) &
        (df['distance'] <= 10),

        # MIDRANGE
        (df['three'] == False) &
        (df['distance'] > 10),

        # CORNER 3
        (df['three'] == True) &
        (df['locationy'].abs() > 20) &
        (df['locationx'] < -18),

        # WING 3
        (df['three'] == True) &
        (df['locationy'].abs() > 8),

        # TOP 3
        (df['three'] == True) &
        (df['locationy'].abs() <= 8)
    ]

    choices = [
        'Heave',
        'At Rim',
        'Low Post',
        'Elbow',
        'High Post',
        'Short 2',
        'Midrange',
        'Corner 3',
        'Wing 3',
        'Top 3'
    ]

    df['shot_zone'] = np.select(
        conditions,
        choices,
        default='Other'
    )

    # shooter_id and shot_zone combination
    df['shooter_zone'] = (
        df['shooter_id'] + '_' + df['shot_zone']
    )

    # str.strip('{}')        removes { }
    # str.split(',')         separates the 4 measurements
    # expand=True            puts them into 4 columns
    # astype(float)          converts strings → actual numbers
    approach = df['closestdefapproach'].str.strip('{}').str.split(',', expand=True).astype(float)

    # approach columns labels
    approach.columns = ['defdist_1sec', 'defdist_075sec',
                        'defdist_05sec', 'defdist_025sec']
    df[approach.columns] = approach

    # Change in defender distance leading up to the shot
    df['defender_closing'] = (
        df['defdist_1sec'] - df['defdist_025sec']
    )
    return df
