## Attention Head

### Layers

-

## Positional Embeddings vs. Rotary Embeddings

## RMSNorm vs LayerNorm

### LayerNorm

_Variables_

- $\gamma$:
- $\beta$:
- $\epsilon$:
- n: The number of features in the layer
- x: The input vector

### RMSNorm

_Variables_

- $\gamma$: A learnable scale parameter
- $\epsilon$: A small constant (used for stability)
- n: The number of features in the layer
- x: The input vector

_Equation_\
 $y = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}} \cdot \gamma$

**What's the deal with $\epsilon$?**
Epsilon is used for numerical stability.
Gee thanks! What the hell is that?

So imagine you divide a number by 0? That gives you infinity, exploding your computation!
So let's run through RMS without epsilon using [0.000001, 0.000000, 0.000002].

- First you square them:

  - [1e-12, 0, 4e-12]

- Then find the mean (keeping the dimension):

  - [1.66666667e-12]

- Then take the reciprocol sqrt 1 / sqrt(1.66666667e-12), lets call it rsqrt

  - 1e-12 / rsqrt = 1e-2
  - 0 / rsqrt = 0.1
  - 4e-12 / rsqrt = 10
  - This is

- Epsilon essentially sets a minimum for how small the denomenator can get.
- This protects from overflow since the minimum for rsqrt = 1 / sqrt($\epsilon$)
