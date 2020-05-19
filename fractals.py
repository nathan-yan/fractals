import numpy as np
import time
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pprint
from tqdm import tqdm

"""
    DEFINE YOUR MODEL HERE!
    things to try:
        - change the weight variances
        - change how many layers there are in the network
        - change up the non-linearities. As a general rule, adding piecewise functions like maximum and clipping create these really jagged structures I think look very cool
        - something very interesting is to actually optimize this model on a task, maybe the MNIST dataset, and see how the fractal evolves as the model is optimize. Not sure if this is possible, and optimizing neural networks with complex weights seems to be much more involved than real-valued weights
"""

weight1 = 0.2 * np.random.randn(2, 5) * 1j + 0.2 * np.random.randn(2, 5)
weight2 = 0.35 * np.random.randn(5, 5) * 1j + 0.35 * np.random.randn(5, 5)
weight3 = 0.45 * np.random.randn(5, 1) * 1j + 0.45 * np.random.randn(5, 1)

def model(inp):
    #fc1 = np.tan(inp @ weight1) 
    #fc2 = fc1 @ weight2
    #fc3 = fc2 @ weight3
    
    """
    MODEL 1:
    
    fc1 = (inp @ weight1) ** np.sqrt(2)
    fc2 = np.tan((fc1 @ weight2))
    fc3 = fc2 @ weight3
    """

    """
    MODEL 2:
    This model is particularly interesting because of the piecewise nature of the non-linearity (max). This results in fragments or shards in the end fractal.
    """
    #fc1 = np.maximum((inp @ weight1) ** np.sqrt(2), 0)
    #fc2 = np.maximum(np.tan((fc1 @ weight2)), 0)
    #fc3 = fc2 @ weight3 

    
    #MODEL 3:
    fc1 = (np.minimum(inp, 0) @ weight1) ** np.sqrt(2)
    fc2 = np.tan(fc1 @ weight2)
    fc3 = fc2 @ weight3
    
    return fc3

def graph(params, graph = True):
    """
        graph(...) -> None

        Given window parameters, graphs the fractal with 30 iterations and a threshold of 0.8
    """

    # Yikes
    s = time.time()
    x = np.arange(params['x_min'], params['x_max'], params['step'])
    y = np.arange(params['y_min'], params['y_max'], params['step'])

    params['num_x'] = len(x)
    params['num_y'] = len(y)

    xg, yg = np.meshgrid(x, y)

    params['c'] = (xg * 1j + yg).flatten()

    params['z'] = np.zeros(len(params['c']))
    params['heatmap'] = np.zeros_like(params['z'])

    for iteration in tqdm(range (30)):
        # plot it
        plt.cla()
        plt.imshow(params['heatmap'].reshape([params['num_x'], params['num_y']]), cmap = "jet")
        plt.pause(0.001)
        
        # housecleaning for neural network input
        c_ = np.expand_dims(params['c'], -1)
        z_ = np.expand_dims(params['z'], -1)

        inp = np.concatenate([c_, z_], axis = -1)

        # remove the second axis
        params['z'] = model(inp)[:, 0]#

        # is z above the threshold?
        params['heatmap'] += np.absolute(params['z']) > 0.8

    # plot it
    plt.cla()
    plt.imshow(params['heatmap'].reshape([params['num_x'], params['num_y']]), cmap = "jet")
    plt.pause(0.001)

def iteration(params, graph = True, its = 1):
    """
        iteration(...) -> None

        Performs a single iteration of the recursive function and updates the heatmap
    """

    for i in tqdm(range (its)):
        # plot it
        plt.cla()
        plt.imshow(params['heatmap'].reshape([params['num_x'], params['num_y']]), cmap = "jet")
        plt.pause(0.001)

        c_ = np.expand_dims(params['c'], -1)
        z_ = np.expand_dims(params['z'], -1)

        #print(c_.shape, z_.shape)

        inp = np.concatenate([c_, z_], axis = -1)

        params['z'] = model(inp)[:, 0]

        params['heatmap'] += np.absolute(params['z']) > 0.8

    plt.cla()
    plt.imshow(params['heatmap'].reshape([params['num_x'], params['num_y']]), cmap = "jet")
    plt.pause(0.001)


def onclick(event, params, ax):
    # zoom, first by centering around the mouse click
    posx = (event.xdata / params['num_x']) * (params['x_max'] - params['x_min']) + params['x_min']
    posy = (event.ydata / params['num_y']) * (params['y_max'] - params['y_min']) + params['y_min']

    # then decrease the window size (zoom)
   
    params['window_temp'] = params['window'] / params['zoom']

    # center the window around posx and posy
    params['x_min_temp'] = posx - params['window_temp']
    params['x_max_temp'] = posx + params['window_temp']

    params['y_min_temp'] = posy - params['window_temp']
    params['y_max_temp'] = posy + params['window_temp']

    # increase resolution by factor of two (computation time is maintained because you also scaled the size of the window)
    params['step_temp'] = params['step'] / params['zoom']
        
    # re-graph
    #graph(params)
    width = params['num_x'] / params['zoom']
    height = params['num_y'] / params['zoom']

    plt.cla()
    plt.imshow(params['heatmap'].reshape([params['num_x'], params['num_y']]), cmap = "jet")

    rect = patches.Rectangle((event.xdata - width/2, event.ydata - height/2), width, height, linewidth=1, edgecolor = 'b', facecolor='none')
    ax.add_patch(rect)

    plt.pause(0.001)

def main():
    # initial window parameters, adjust as needed
    x_min = -4
    x_max = 4
    y_min = -4
    y_max = 4

    step = 0.025

    window = 4

    num_x = (x_max - x_min)/step
    num_y = (y_max - y_min)/step

    z = None
    c = None
    heatmap = None

    params = {
        "z" : z, 
        "c" : c,
        "heatmap" : heatmap,
        "x_min" : x_min,
        "x_max" : x_max,
        "y_min" : y_min, 
        "y_max" : y_max,
        "step" : step,
        "window" : window,
        "num_x" : num_x,
        "num_y" : num_y,
        "zoom" : 2
    }
    
    pp = pprint.PrettyPrinter(indent = 4)

    # cool seeds:
    # seed, description - network model
    # 919333390 - 1
    # 650867666 - 1

    # 74214537, overlapping orange blobs - 1

    # 112940321 - 2

    # 525058265 - 3

    seed = input("Choose a seed, press enter for a random one: ")

    if seed == "":
        seed = np.random.randint(0, 1000000000)

    print("This is the network seed: " + str(seed))
    np.random.seed(int(seed))

    fig, ax = plt.subplots()
    # event listener for clicks
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, params, ax))

    graph(params)

    # really bad command line for refining or adding extra iterations to the graph
    while True:
        cmd = input("\n\n> ")

        s = cmd.split(' ')
        if (s[0].lower() == 'q'):
            break;
        elif (s[0] == "iterate"):
            its = int(s[1])
            iteration(params, its = its)

        elif (s[0] == "refine"):
            factor = float(s[1])
            params['step'] /= factor
            graph(params)
        
        elif (s[0] == 'setzoom'):
            factor = float(s[1])
            params['zoom'] = factor
        
        elif (s[0] == 'zoom'):
            param_list = ['window', 'x_min', 'x_max', 'y_min', 'y_max', 'step']

            for p in param_list:
                if params.get(p + '_temp') == None:
                    print("Click on the fractal to choose a location to zoom")

                    break;

                params[p] = params[p + '_temp']
                params[p + '_temp'] = None
            else:
                graph(params)

if __name__ == "__main__":
    main()