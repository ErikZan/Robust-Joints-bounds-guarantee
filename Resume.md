# Joint position and velocity bounds 
Possible approaches:
- Add bounded known error in the model : addind this directly to the initial `MAX_ACC` leads to a different space state region plot and different value of viability etc but its quite **trivial**, namely is the same problem with a different `MAX_ACC` value.

- Add bounded know error only in the computation of future step: more representative but computation of feasible-viable region do not take into account of this error (constant and applied every timestep leading to a certain error) so not useful nor representative: our viability-feasibility computation don't have any tools to compensate to this error

- Add random error : is possible to  

  

