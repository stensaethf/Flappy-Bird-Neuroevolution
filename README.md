Flappy-Bird-Neuroevolution
==========

Neural networks learning to play Flappy Bird.

#### Parameters
Note: tags with a single dash require a value after it (example: -p 20 -c 0.5). Tags with a double dash should not have a value after it (example: --crossover-section --perfect).

| Tag                 | Default | Description  |
| :-----------------: | :-----: | :----------: |
| -b                  | 25      | Fitness boost. Boost to fitness for passing through set of pipes. All values are valid. |
| -c                  | 0.15    | Crossover rate. Range: [0, 1] |
| -e                  | 2       | Number of elite clones. Range [0,] |
| -fps                | 30      | Max frames per second. Range: [0,] |
| -g                  | 1       | Save every x generations. Range: [0,] |
| -l                  | 2       | Difficulty level. 0: easy, 1: medium, 2: hard. |
| -m                  | 0.15    | Mutation rate. Range: [0, 1] |
| -n                  | 3/7/1   | Structure. Form: x/y/.../z (x is number of inputs, y is number of nodes in first hidden layer, z is number of outputs). Requires 2 or 3 inputs and 1 ouput. |
| -p                  | 50      | Population size. Range: [2,]. |
| -s                  | 0       | Strategy. 0: neural networks, 1: always flap, 2: never flap, 3: randomly flap. |
| --crossover-section | off     | Crossover section. Crossover a random section of the genome instead of the first hidden layer. |
| --perfect           | off     | Perfect game. Increases the distance between pipes so that it is possible to have a perfect game (never die). |
| --print             | off     | Print. Saves the generation number, min fitness, max fitness and average fitness for each generation in data/results.txt |
| --random-nn         | off     | Random neural networks. Population will consist of neural networks of varying structures. Crossover is automatically set to --crossover-section. |
