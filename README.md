# Basketball Shot Probability Modeling

A Python machine-learning project exploring how shot location, shooter tendencies, and defensive pressure can help predict the probability of a basketball shot going in.

I developed this project as part of a basketball analytics internship assessment. This public repository is a portfolio overview of my approach, not a copy of the private assessment or its data.

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

## Project Availability

The original dataset, assessment instructions, and submission files are not included in this public repository. A public-data demonstration may be added separately.

## Try the Feature Engineering

This repository includes a small demonstration using synthetic basketball shots. No data from the internship assessment is included.

To run it:

1. Install the required packages:
   `pip install pandas numpy`

2. Run the example:
   `python example.py`

The example shows how the code assigns shot zones, combines shooter IDs with those zones, and calculates how much distance a defender closed before a shot.
