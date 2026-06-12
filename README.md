# Uncertainty-Aware Digital Twin for 2D Heat Diffusion

This project builds a simplified digital-twin workflow for a 2D heat-diffusion system. It combines simulation data, surrogate modeling, sparse-sensor calibration, uncertainty estimation, and validation against a finite-difference solver.

## Motivation

Scientific and experimental systems often require fast surrogate models that can approximate expensive simulations, update from sparse experimental observations, and quantify uncertainty. This project demonstrates that workflow on a controlled physical system: heat diffusion on a 2D plate.

## Physical System

The system is a 2D plate with randomized heat sources and varying thermal diffusivity. A finite-difference simulator generates temperature fields over time.

## ML Task

Input:
- initial temperature field
- heat-source field
- thermal diffusivity parameter

Output:
- future temperature field

## Planned Workflow

1. Generate simulation data with a finite-difference heat equation solver.
2. Train a PyTorch surrogate model to predict future temperature fields.
3. Validate the surrogate against held-out simulations.
4. Calibrate an unknown physical parameter from sparse noisy sensor observations.
5. Estimate predictive uncertainty using an ensemble of models.
6. Test reliability on out-of-distribution heater configurations.

## Results

Coming soon.

## Limitations

This is a simplified synthetic system. The purpose is to demonstrate an end-to-end scientific ML workflow, not to model a full experimental platform.