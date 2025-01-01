# Reinforcement-Learning

The intention of this repository is show the various reinforcement learning (RL) algorithms and how they perform on different benchmarks. The algorithms implemented are tested on classical RL benchmarks and environments. The environments used are either implemented from scratch or from [Gymnasium](https://gymnasium.farama.org/).

The structure is roughly as follows. Mostly it is about solving different environments, which are increasing in complexity and difficulty. As the repository grows more challenging environments will be tackeled and added. The approach is to use what works on the simpler benchmarks, from which we already know that the algorithm implementation works and if the agent does not perform well then it might be that we need to adapt the algortihm or use an entirely different one.

After an environment has been solved, one can find a report inside the respected folder which informs about the results and shows various metrics and describes them. Also the algorithms used are explained in the report as well and links to further resources are provided.

We focus mainly on continuous control, meaning that we are dealing with either a continuous action space, or state space or both. Therefore the folder **Continuous-Control** is the one where most is happening. The **RL-Concepts** folder contains discrete problems and their solutions, but it will probably not be as much developed as the **Continuous-Control** folder.

## Continuous-Control
The first problem is the [Cart Pole](https://github.com/Massive-Rabbit-8223/Reinforcement-Learning/tree/main/Continuous-Control/CartPole) problem:

![Agent-solving-CartPole][def]

[def]: https://github.com/Massive-Rabbit-8223/Reinforcement-Learning/blob/main/GIFs/CartPole.gif