# Ballistic_Missile_Simulation
This code can do some native numerical analysis on ballistic trajectory

## Basic physics
This code considers (1) gravity, (2) air drag and (3) rocket equation. The rotation of the earth is omitted becase only in 3D can we fully explore the effects of earth rotation. And since I'm too lazy to change the whole code from 2D to 3D, I decide to ignore it. 

### Gravity
This is simple, we have:
$$\vec{a}_g = -\frac{GM_E}{r^3}\vec{r}$$

### Air drag
This is much more tricky. In the acent phase, air drag is relatively irrelevant. But when the war head is returning to the Earth, it will hit the dense atmosphere with hypersonic velocity. However, hypersonic/supersonic drag is very complicated. In this work I simply use:
$$\vec{a}_A = \frac{1}{2} \rho C_D A v^2$$
Note that here we are assuming a constant geometry. that is $\vec{v}\parallel \vec{a}_A$. $\rho$ is the air density, here we simply apply [the model for troposphere](https://en.wikipedia.org/wiki/Density_of_air).
$$\rho = \frac{p_0 M}{RT_0}\left(1-\frac{Lh}{T_0}\right)^{\frac{gM}{RL}-1}, \quad h = r - R_E$$

### [Rocket Equation](https://en.wikipedia.org/wiki/Tsiolkovsky_rocket_equation)
The acceleration of the rocket depends on it's mass, while thrust is basically the same. That is:
$$\vec{a}_R = \frac{F}{m(t)} \vec{n}, \quad \frac{dm}{dt} = -F/v_e$$
where $v_e={\rm Isp}\times g$ is the exhaust velocity. 
$\vec{n}$ is the pointing of the rocket, which is controllable. I used the following strategy:
* if $h < 10$ km, where the rocket is still in the dense air, then $\vec{n} = \mathrm{rotate}(\hat{r}, \theta)$, where $\theta$ is basically the $\mathrm{launch~angle} - 90^\circ$.
* if $h > 10$ km, $\vec{n} = \hat{v}$ so that the acceleration is maximized.


### Add them all together
Combining them, we can derive the total acceleration:
$$\vec{a} = \vec{a}_g+\vec{a}_A+\vec{a}_R$$

and use ```scipy.integrate.solve_ivp``` to do the integration.
