# Flow Matching

## Idea 
Instead of predicting noise(e), the model predicts a velocity field:

v(x, t)

## Key Equation
dx/dt = v(x, t)

## Why it's better
-- More stable training
-- no need for variance schedule
-- better theoretical grounding 

## Used in this project
We use flow matching loss instead of MSE noise prediction.

