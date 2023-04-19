# Tests to get statistical evidences

## Play-time tests
These are the ones intended to show how quickly the enemies would learn, from scratch, by playing along with the player.
We should have a way to convert cycles to seconds, and strive for learning non-random strategies within 20 seconds, and challengin strategies within 3 minutes.
Each one of the tests should be repeated on as many batches as available to increase statistical significance

Base tests: for these we just run the trained and collect the reward results, to then analyze peaks

- Random player
- Evasive player
- Hiding player
- Shifty player

Cross tests: for every learned enemy net, test on the other strategies (learning OFF) and see average score


## Long-term tests
These are intended to show the long term results of our setup, run on subsequent episodes as if the player played more sessions.
We are less interested in time here, we should instead see when the rewards reach a plateau

Base tests: train to reach the plateau on these strategy settings

- Random player
- Evasive player
- Hiding player
- Shifty player
- Multiple strategies/players (generic learning)

Cross tests: for every learned enemy net, test on the other strategies (learning OFF) and see average score

Cross learning: for every learned enemy net, train on the other strategies (learning ON) and see if and how quickly scores rise back to the corresponding base test


# Hyper parameter optimization

HPO to be performed on

- Batch size
- Gamma
- Episode start
- Episode end
- Episode decay
- Temperature Tau
- Learning rate
- Replay memory

with [bayesian optimization](https://github.com/fmfn/BayesianOptimization) 
OR
with [genetic algorithms](https://optuna.readthedocs.io/en/stable/index.html)

[More Examples](https://docs.ray.io/en/latest/tune/examples/hpo-frameworks.html)



# Results

Before tuning

Long-term
    Evasive: 48 tests of 100 eps x 1000 steps 
        avg -0.68 reached in 3 eps
        std 0.13 reached in 5 eps
        final result: avg -0.36, std 0.09