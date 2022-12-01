# Macrosystis Dynamics Notes

General dynamics from [Rosman2013](@cite) with details (e.g. drag formulation) from [Utter1996](@cite).

## Implimentation overview
- Each plant is split into nodes
- When the particle dynamics are called each node interpolates the velocity at the *node* (does not average over the segment so the characteristic length of the flow field needs to be >> segment length)
- Then calculates the tension from the adjacent segments, and the drag from the velocity on a projected area
- TendencyCallback then itterates over the particles and nodes to apply the drag
- For each node we transform every grid point to a frame of reference with the node at the centre, and the line connecting the adjacent nodes parallel to the z axis
- We then rotate the frame to have the line connecting the node to adjacent nodes (i.e. if z>0 in the previous frame the line connecting the node to the next node, and z<0 the previous node)
- For grid points less than a particular distance (rₑ) from the z axis their weight is calculated from mask, currently using a guassian with 3σ inside rₑ
- Once every grid points weight is found the sum of weights is taken to normalise the stencil
- For each grid point we then add to the tendency the drag force from the node multiplied by the normalised weight