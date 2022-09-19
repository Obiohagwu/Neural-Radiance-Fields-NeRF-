# NEURAL RADIANCE FIELDS
---
An introduction to Nerual Radiance Fields (NeRFs) by implementing the NeRF architecture outlined in Srinivasan et al's *"NeRF: Representing Scenes as
Neural Radiance Fields for View Synthesis"*

### INTRODUCTION
---
NeRFs are a new class of neural network architecture primarily suited for synthesis of novel volumetric scenes from sparse input scenes.
The algorithm works by representing a scene using a fully-connected (non-convolutional) DNN, where the input is a single continuous 5D coordinate (spatial(x,y,z) and view orientation( $\theta$, $\phi$ )), with output being the volume density and view-dependant emitted radiadnce at that spatial location.
Generally speaking, we synthesize views, by querying 5D coordinates along camera 

### ARCHITECTURE
---
From pg



### TECHNICAL CONTRIBUTIONS
---
- An approach for representing continous scenes with complex geometry and materials as 5D neural radiance fields, parameterized as basic MLP networks
- A differentiable rendering procedure based on classical volume rendering techniques, which we use to optimize these representation from standard RGB images. This includes hierarchical sampling strategy to allocate the MLPs capacity towards space with visible scene content
- A positional encoding to map each input 5D coordiante into a higher-dimensional space, which enables us to sucessfully optimize neural radiance fields to represent high-frequency scene content

We see that NeRFs achieve SOTA on multiple 3D volumetric benchmarks, beating tradinitonal convolutional based methods and other view synthesis methods.

