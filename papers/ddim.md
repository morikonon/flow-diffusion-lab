# DDIM (Denoising Diffusion Implicit Models)

## Idea
Instead of samplig with stochastic noise at each step (as in DDPM), DDIM uses a determenistic mapping to generate samples:

x_t -> x_{t - 1}

## Key Equation
x_{t-1} = √α_{t-1} * x_0 + √(1 - α_{t-1}) * ε_θ(x_t, t)

## Why it's better
- faster sampling (fewers steps needed)
- determenistic generation (same input -> same output)
- controllable trade-off between speed and quality

## Used in this project
We use DDIM sampling to accelerate image generation during inference.
