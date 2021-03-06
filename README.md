# Fractals
Fractals generated by randomly initialized neural networks.

## How does this work?
Traditional fractals like the Mandelbrot set rely on a recursive function applied to complex numbers. The Mandelbrot set relies on a relatively simple equation: z{t + 1} = z{t}^2 + c, where z{0} = 0. The goal of this project was to utilize much more complicated functions to generate random fractals that are interesting and bizarre, while still maintaining the infinite complexity and self-similarity the Mandelbrot set is so well known for. 

The general description of z{t + 1} is that it is some function of z{t} and c. For example, in the Mandelbrot set, the function is z{t}^2 + c. This project makes the function f_nn(z{t}, c), where f_nn is a "neural network" (it isn't really a neural network because it isn't trying to solve a specific problem, but it's much easier to say it is). 

I randomly initialize 3 "weight" matrices of random complex numbers and make interesting models, combining matrix multiplication with interesting non-linearities like trigonometric functions and piecewise functions like min/max. I initialize z{0} to a flattened array of all zeros and initalize c to a grid of complex numbers. I recursively calculate z{t + 1} using the neural network model and generate the fractal at the end by calculating for each pixel how many iterations it spent above the magnitude threshold.

A slightly more elaborate explanation is [here](https://nathan-yan.github.io/fractals).

## Dependencies
- numpy>=1.16.0
- matplotlib>=3.0.0
- tqdm>=4.0.0

## How to use
To use run:
```
  python fractal.py
```

You'll be prompted with an input asking for a random seed. Press enter if you just want to use a random one. A fractal will be shown to you, at a fairly low resolution. The reason for the low resolution is because fractal generation is compute intensive, as you need to compute the trajectory of every point on the complex plane; the fact that a neural network is involved greatly increases the compute. 

Click on a region to select a zoom location. A rectangle will show you the bounds of the new fractal. Once you have picked a zoom location you like, type
```
  zoom
```
in the prompt to render the zoomed fractal. The default zoom is 2x, but if you would like to zoom in more you can set the zoom factor by typing
```
  setzoom #scale
```
where #scale is the factor we scale by. If scale is 1, the fractal is not zoomed at all, and is the equivalent of pannning. If scale is 0.5, it will actually zoom out by a factor of 2.

To refine the image (increase resolution), type
```
  refine #resolution_scale
```
where #resolution_scale is the factor we increase the resolution by (2 would bring a 400 x 400 resolution to a 800 x 800 resolution). This command may take a lot of time to compute, because we need to completely re-graph the function. Working on a faster way of doing this. 

To perform more iterations (adds more detail to the fractal), type

```
  iterate #num_iterations
```

where #num_iterations is the number of *extra* iterations we perform. By default rendering does 30 iterations, and you shouldn't really ever need more than this. 
