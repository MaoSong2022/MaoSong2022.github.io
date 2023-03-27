
# Part 1 Learning With Every Mile Driven

Motional uses Continuous Learning Framework (CLF) to train our vehicles at an unprecedented scale quickly.
![](assets/images/motional_continuous_learning_framework.png)


# Nuplan
With an open-loop system, the input is independent of the system's response, regardless of the system's behavior. Open-loop is sometimes called imitation learning, since the system simply checks that the planned route is similar to the one the driver took.

In closed-loop evaluation, the planned route is used to control the vehicle. The vehicle may deviate from the original route that the driver took. Other drivers will then react accordingly.


# Part 2 Closing the Loop: Traveling Back in Time to Help AVs Plan Better

Historically, learning-based planner architectures have used rasterized images to represent the ego vehicle and all objects in its environment. The problem is that rasterized images are not differentiable. One recent solution has been to use vectorized raster images, the elements of which are connected mathematically and can be manipulated by changing inputs and outputs. In a vectorized raster image, we can move around the ego vehicle through simple translational and rotational operators.
![](assets/images/motional_planning_loop.png)


![](assets/images/motional_closed_loop_training.png)


# Part 3 Predicting the future in real time for safer autonomous driving
Motional models all the agents and the map elements with a graph attention network that is processed through the vehicle’s onboard compute, which is running through models at a high update rate.
![](assets/images/motional_graph_attention_network.png)
Since there is always a degree of unpredictability with human drivers and other objects we encounter on the road.
So, to account for the uncertainties, Motional’s prediction model represents possible future trajectories with a mixture of Gaussian distributions.
![](assets/images/motional_2d_gaussian_model.png)


# References
https://motional.com/news
