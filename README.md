# Ballistic_Missile_Simulation
This code can do some native numerical analysis on ballistic trajectory

## Basic physics
This code considers (1) gravity, (2) air drag and (3) rocket equation. The rotation of the earth is omitted becase only in 3D can we fully explore the effects of earth rotation. And since I'm too lazy to change the whole code from 2D to 3D, I decide to ignore it. 

### Gravity
This is simple, we simply have:
$$
\vec{a}_g = -\frac{GM_E}{r^3}\vec{r}
$$

### Air drag
This is much more tricky. In the acent phase, air drag is relatively relevant. But when the war head is returning to the Earth, it will hit the dense atmosphere with hypersonic velocity. However, hypersonic air drag is very complicated. In this work I simply use:
$$
\vec{a}_A = \frac{1}{2} C_D A v^2 
$$

### Rocket Equation
The acceleration of the rocket depends on it's mass, while thrust is basically the same. That is:
$$
\vec{a}_{R} = \frac{F}{m(t)}\vec{n}_{\rm pointing},\quad \frac{dm}{dt} = -F/v_e
$$

### Add them all together
Combining them, we can derive the total acceleration:
$$
\vec{a} = \vec{a}_g + \vec{a}_A + \vec{a}_{R}
$$

and use ```scipy.integrate.solve_ivp``` to do the integration.
