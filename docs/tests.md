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

Settings: 1 episode of 5000 cycles (tot playtime: 5000 cycles)



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

Settings: 100 episodes of 1000 cycles (tot playtine: 100'000 cycles)


# Hyper parameter optimization

HPO to be performed on

- Gamma
- Episode start
- Episode end
- Episode decay
- Temperature Tau
- Learning rate

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
    Multiple: 48 tests of 100 eps x 1000 steps (cycling evasive, hiding, shifty by episode)
        Oscillatory results between avg -0.5 and -0.66
        Third episode (on shifty) works well already after the first two episodes on evasive and hiding
        Shifty is always the better performing one
        At the end it oscillates between avg -0.44 and -0.5, std doesn't stabilize



Tuning

Settings: 6 episodes of 1000 cycles (tot playtime: 6000 cycles)
100 instances
ranges: {
    "gamma": [0.9, 0.99],
    "eps_start": [0.8, 0.9],
    "eps_end": [0.02, 0.05],
    "eps_decay": [200, 1000],
    "tau": [0.005, 0.01],
    "learning_rate": [1e-4, 1e-3],
}

2023-04-26 12:56:00,795 both: Best metrics: 
{'avg_ep_avg_reward': -0.4728836830810709,
 'config': {'eps_decay': 913.0213920325779,
            'eps_end': 0.04546406201073101,
            'eps_start': 0.8255359867903806,
            'gamma': 0.9434996340408109,
            'learning_rate': 0.00026195288980869773,
            'tau': 0.007382362148858288},
 'time_total_s': 208.89277482032776,


 Settings: 6 ep of 1000 cycles (tot 6000)
 1000 instances

ranges: {
    "gamma": [0.9, 0.99],
    "eps_start": [0.8, 0.9],
    "eps_end": [0.02, 0.05],
    "eps_decay": [200, 2000],
    "tau": [0.005, 0.01],
    "learning_rate": [1e-4, 1e-3],
}

2023-04-26 13:25:38,177 both: Best metrics: 
{'avg_ep_avg_reward': -0.40767106496635425,
 'config': {'eps_decay': 1213.7774149761788,
            'eps_end': 0.04360871119949491,
            'eps_start': 0.8535821424779334,
            'gamma': 0.9419305967430943,
            'learning_rate': 0.000433329981053619,
            'tau': 0.008404270072628766},
 'time_total_s': 331.25559759140015,

 This took around 1h10m