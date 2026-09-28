# Basketball Shot Probability Modeling

A Python machine-learning project exploring how shot location, shooter tendencies, and defensive pressure can help predict the probability of a basketball shot going in.

I developed this project as part of a basketball analytics internship assessment. This public repository demonstrates my approach without including the original assessment dataset or submission files.

## What I Worked On

I started with logistic regression because I wanted a simple baseline. I then tested gradient boosting to see whether a model that could capture more complex relationships between shot characteristics would produce better probability predictions.

I also experimented with feature engineering, including:

- **Shot zones:** Grouping shots into areas such as the corner three, midrange, and low post.
- **Shooter-specific zones:** Combining shooter identity with shot zone to explore whether players have different strengths in different areas.
- **Defender movement:** Using changes in defender distance leading up to a shot, rather than only the defender's distance at the time of the shot.

## Tools and Methods

Python, pandas, NumPy, scikit-learn, logistic regression, histogram-based gradient boosting, target encoding, cross-validation, and permutation importance.

I used log-loss to evaluate the models because the goal was to predict probabilities, not just classify shots as makes or misses.

## What I Learned

One of the biggest things I learned was the importance of keeping training and validation data separate. I also learned how repeatedly making decisions based on the same validation set can make a model's performance look better than it might be on completely new data.

Permutation importance helped me understand which features the model relied on. It also showed me why overlapping features can be difficult to evaluate individually.

## What's in This Repository

- **`feature_engineering.py`** — Creates shot zones, shooter-specific zones, and defender-movement features.
- **`example.py`** — Demonstrates feature engineering on three made-up shots.
- **`synthetic_data.py`** — Generates synthetic shots with the columns needed to run the model.
- **`model.py`** — Trains a gradient-boosting model and outputs predicted shot-make probabilities.

## Try the Feature Engineering

This repository includes a small demonstration using synthetic basketball shots. No data from the internship assessment is included.

First, install the required packages:

```bash
pip install pandas numpy scikit-learn
```

Run the example:

```bash
python example.py
```

The example shows how the code assigns shot zones, combines shooter IDs with those zones, and calculates how much distance a defender closed before a shot.

## Run the Full Model

The repository also includes a runnable demonstration of the modeling pipeline.

Run:

```bash
python model.py
```

The script:

1. Generates 500 synthetic shots.
2. Creates additional features using the feature-engineering code.
3. Splits the data into training and validation sets.
4. Trains a histogram-based gradient-boosting classifier.
5. Prints the validation log loss and five predicted shot-make probabilities.

The model uses one-hot encoding for categorical shot information and target encoding for shooter-specific zones.

## About the Data and Results

The original assessment dataset, instructions, and submission files are not included in this public repository.

The synthetic data is only intended to demonstrate how the code works. Results from running `model.py` are not measurements of model performance on real NBA shots.

The public demonstration uses the same general feature-engineering and modeling approach I developed for the assessment, but it is not a reproduction of the original assessment results.
