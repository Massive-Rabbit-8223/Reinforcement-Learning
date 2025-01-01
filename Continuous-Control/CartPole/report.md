For solving the Cart Pole problem I used vanilla policy gradient (VPG) as baseline and further added two modifications as comparison. First I describe VPG and then the modifications. For my VPG implementation I relied hevily on [OpenAI's Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) introduction to policy optimization. Basicially what I write here is a summary of their introduction together with how the algorithm and its variations perform on the Cart Pole problem.

Why VPG? The policy of *Action-Value Methods* selects an action $a_t$ based on the values of all the reachable next states $v(S_{t+1})$ given the current state $S_t$, where the value of a state is the expected return from that state onwards:

$$
\begin{align}
v(s) &:= \mathbb{E}[G_t|S_t=s]\\ 
       &=\mathbb{E}\Bigl[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}|S_t=s\Bigr] \text{, for all }s \in \mathcal{S}.
\end{align}
$$

A greedy action choice, given the value function of the current policy, denoted by $v_{\pi}$, can be found by comparing which of the possible actions leads to the state with the highest estimated value:

$$
\pi(s_t) = \argmax_{a \in \mathcal{A}(s_t)}\mathbb{E}\Bigl[R_{t+1}+\gamma v_{\pi}(S_{t+1}) | S_t=s, A_t=a \Bigr].
$$
 
A problem we face in the Cart Pole problem and in general in continuous cases is that we have infinitely many possible states to be in and therefore cannot learn a value for each state. In the Cart Pole problem we have a continuous state space in that a state consists of the following four features:

Observation | Min | Max
---|:---:|---:
Cart Position | -4.8 | 4.8
Cart Velocity | -$\infty$ | $\infty$
Pole Angle | -24° | 24°

Instead of learning a value function to determine an action to take, we want to learn the policy directly. *Policy-Gradient Methods* do exactly that. We have a parameterized policy with $\theta \in \mathbb{R}^d$ being the parameter-vector. We can write the policy as follows:

$$
\pi(a|s, \theta) = \text{Pr}\{A_t=a | S_t=s, \theta_t=\theta\}.
$$
For shorthand notation we write $\pi_{\theta}(a|s)$ instead.

Now the question is how to find a suitable $\theta$ such that the policy chooses actions which maximize the expected return. We define a performance measure for our policy, which we want to maximize through gradient ascent, hence the name *Policy-Gradient*. We denote the performance measure by $J(\theta)$ with respect to the policy parameter. We choose the expected return as our performance measure since this is the quantity we want to maximize. We write:

$$
J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}\big[R(\tau)\bigr].
$$

where $R(\tau)$ gives the finite-horizon undiscounted return. We use the following update rule to optimize our policy via gradient ascent:

$$
\theta_{t+1} = \theta_t + \alpha \nabla_{\theta}J(\pi_{\theta})|_{\theta_{t}}.
$$