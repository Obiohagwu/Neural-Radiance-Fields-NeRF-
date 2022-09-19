# NEURAL RADIANCE FIELDS
---
An introduction to Nerual Radiance Fields (NeRFs) by implementing the NeRF architecture outlined in Srinivasan et al's *"NeRF: Representing Scenes as
Neural Radiance Fields for View Synthesis"*

### INTRODUCTION
---
NeRFs are a new class of neural network architecture primarily suited for synthesis of novel volumetric scenes from sparse input scenes.
The algorithm works by representing a scene using a fully-connected (non-convolutional) DNN, where the input is a single continuous 5D coordinate (spatial(x,y,z) and view orientation( \theta, \phi)), with output being the volume density and view-dependant emitted radiadnce at that spatial location.
Generally speaking, we synthesize views, by querying 5D coordinates along camera 

### ARCHITECTURE
---
From pg




