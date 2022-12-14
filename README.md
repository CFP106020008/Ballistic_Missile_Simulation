# Ballistic_Missile_Simulation
This code can do some native numerical analysis on ballistic trajectory

## Basic physics
This code considers (1) gravity, (2) air drag (3) rocket equation and (4) rotational fictitious force. 

### Gravity
This is simple, we have:
$$\vec{a}_g = -\frac{GM_E}{r^3}\vec{r}$$

### Air drag
This is much more tricky. In the acent phase, air drag is relatively irrelevant. But when the war head is returning to the Earth, it will hit the dense atmosphere with hypersonic velocity. However, hypersonic/supersonic drag is very complicated. In this work I simply use:
$$\vec{a}_A = -\frac{1}{2} \rho C_D A v \vec{v}$$
Note that here we are assuming a constant geometry. that is $\vec{v}\parallel \vec{a}_A$. $\rho$ is the air density, here we simply apply [the model for troposphere](https://en.wikipedia.org/wiki/Density_of_air).
$$\rho = \frac{p_0 M}{RT_0}\left(1-\frac{Lh}{T_0}\right)^{\frac{gM}{RL}-1}, \quad h = r - R_E$$

### [Rocket Equation](https://en.wikipedia.org/wiki/Tsiolkovsky_rocket_equation)
The acceleration of the rocket depends on it's mass, while thrust is basically the same. That is:
$$\vec{a}_R = \frac{F}{m(t)} \vec{n}, \quad \frac{dm}{dt} = -F/v_e$$
where $v_e={\rm Isp}\times g$ is the exhaust velocity. 
$\vec{n}$ is the pointing of the rocket, which is controllable. I used the following strategy:
* if $h < 10$ km, where the rocket is still in the dense air, then $\vec{n} = \mathrm{rotate}(\hat{r}, \theta)$, where $\theta$ is basically the $\mathrm{launch~angle} - 90^\circ$.
* if $h > 10$ km, $\vec{n} = \hat{v}$ so that the acceleration is maximized.

### [Rotational fictitious force](https://en.wikipedia.org/wiki/Coriolis_force)
There are 3 fictitious force in a rotating frame.
* Euler force: $-m\dot{\vec{\omega}}\times \vec{r}$, which is 0 since the angular velocity of the Earth is constant.
* Coriolis force: $-2m(\vec{\omega}\times \vec{v})$, need to be considered.
* Centrifugal force: $-m \vec{\omega}\times(\vec{\omega}\times \vec{r})$, which is effectively $\omega^2\vec{r}$.

### Add them all together
Combining them all, we obtain the total acceleration. Then we can use ```scipy.integrate.solve_ivp``` to do the integration.

## Result
https://user-images.githubusercontent.com/48315222/183479789-446ba6ba-58b0-4a22-8dcc-d449f36f8ce7.mp4



