# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 01:58:45 2022

@author: juliu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import os

# Fundamental Constants
G = 6.67e-11     # Gravitational Constant
M_E = 5.972e24   # kg
R_E = 6371000    # m
g   = 9.81       # m/s^2

# Missile
m_0 = 6200 # kg, launch mass
m_f = 1600 # kg, mass of war head + empty rocket 
m_wh = 800 # kg, mass of war head
isp = 237 # s
v_e = isp*g # m/s
TTW = 1.5 # thrust to weight
F = m_0*g*TTW # thrust, in N
C_D = 5e-4 # drag coefficient, dimensionless
A = np.pi*0.5**2 # cross section, m^2
EA = 80 # angle of elevation, in deg

# Atmospheric constants
T0 = 273.15 + 15 # K, surface temperature
p0 = 1013e2 # Pa, surface pressure
R  = 8.314 # J/K/mol, ideal gas constant
L = 0.0065 # K/m, temperature lapse rate
MW = 0.029 # kg/mol, molecular weight of the air

#%%

# Simulation properties
tmax = 3.6e3 # Simulation time in second, set to 60 min for maximum

# Visualization properties
Box_size = 1.2e7 # Size of the plot
frames = int(1800) # Output frames
Tracing = True # Viewing the sail with tracing mode.
SAVE_VIDEO = False  # Whether you want to save the video

#%%
def initial_condition():
    def launch_parameter():
        return [0, R_E, 0, 100, m_0] # x, y, vx, vy, m
    return launch_parameter()
y_0 = initial_condition()

def rotate(vector, theta): # in degree
    theta = theta/180*np.pi
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return np.matmul(rotation_matrix, vector)
    
def air_density(h):
    # Here we are modeling air density by extrapolating Troposphere condition
    if h < 4.4e4: # 44 km
        rho = p0*MW/R/T0*(1-L*h/T0)**(g*MW/R/L-1)
    else:
        rho = 0
    return rho

#%% Alltitude control
# This is the tricky part, lots of freedom included.
# Here we use: 
#    1. When the missile is still in a dense atmosphere, we let it accelerate
#       in a direction similar to rhat (slight deviation from going straight up)
#    2. When above 10 km, we let the acceleration align with the velocity vector. 
def Decide_Pointing(t, x, y, vx, vy):
    r = np.array([x, y])
    v = np.array([vx, vy])
    rhat = r/np.linalg.norm(r)
    if vx == 0 and vy == 0:
        vhat = rhat
    else:
        vhat = v/np.linalg.norm(v)
    Acc = False
    if np.linalg.norm(r) > R_E + 1e4: # above 10 km, align with velocity
        phat = vhat
    else:
        phat = rotate(rhat, EA-90)
    return phat

#%% Here are the function for ivp solver
def impact(t, y): return np.sqrt(y[0]**2 + y[1]**2) - R_E
impact.terminal = True
impact.direction = -1

def function(t, y):
    separation = False
    r_vec = y[:2]
    r = np.linalg.norm(r_vec)
    v_vec = y[2:4]
    v = np.linalg.norm(v_vec)
    phat = Decide_Pointing(t, y[0], y[1], y[2], y[3])
    dxdt = v_vec[0]
    dydt = v_vec[1]
    
    # This is for thrust, only exist when there are still enough fuel
    if y[4] > m_f:
        a_rocket = F/y[4]*phat
        dmdt = -F/v_e
    else:
        separation = True
        a_rocket = 0
        dmdt = 0
    # Gravity
    a_g = -G*M_E/r**3*r_vec
    # Air drag
    if separation: # decent
        a_A = -0.5*air_density(r-R_E)*v**2*C_D*(0.5*A)/m_wh*v_vec
    else: # acent
        a_A = -0.5*air_density(r-R_E)*v**2*C_D*A/y[4]*v_vec
    # Total acceleration
    a = a_rocket + a_g + a_A
    dvxdt = a[0]
    dvydt = a[1]
    return np.array([dxdt, dydt, dvxdt, dvydt, dmdt])
#%%
# Solving the orbit
sol = solve_ivp(fun=function,
                t_span=[0, tmax],
                y0=y_0,
                t_eval=np.linspace(0,tmax,frames),
                events=impact,
                method='DOP853')

t = sol.t
Data = sol.y
r = np.sqrt(Data[0,:]**2 + Data[1,:]**2)
x = Data[0,:]
y = Data[1,:]
vx = Data[2,:]
vy = Data[3,:]
m  = Data[4,:]

#%%
# Visualization Setup
COLOR = '#303030'
LineColor = 'silver'
fig = plt.figure(figsize = (8, 4.5), facecolor=COLOR)
gs = GridSpec(2, 4, figure=fig)

# Picture
ax = fig.add_subplot(gs[:, :2])
ax.set_facecolor('#202020')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.spines['bottom'].set_color(COLOR)
ax.spines['top'].set_color(COLOR) 
ax.spines['right'].set_color(COLOR)
ax.spines['left'].set_color(COLOR)

earth = plt.Circle((0, 0), R_E, color='cyan')
ax.add_patch(earth)
ax.set_aspect('equal', 'box')

line, = ax.plot(x[0], y[0], color='silver', linestyle='-', linewidth=1)
dot, = ax.plot([], [], color='silver', marker='o', markersize=1, markeredgecolor='w', linestyle='')

ax.set_xlim([-Box_size,Box_size])
ax.set_ylim([-Box_size,Box_size])
#%%
# Velocity Plot
ax1 = fig.add_subplot(gs[0, 2:])
ax1.set_facecolor(COLOR)
velline, = ax1.plot(t[0], np.sqrt(vx[0]**2+vy[0]**2), color='silver')
ax1.spines['bottom'].set_color(LineColor)
ax1.spines['top'].set_color(LineColor) 
ax1.spines['right'].set_color(LineColor)
ax1.spines['left'].set_color(LineColor)
ax1.set_xlim([0,t[-1]])
ax1.set_ylim([0,np.max(np.sqrt(vx**2+vy**2))*1.2])
ax1.tick_params(labelcolor=LineColor, labelsize='medium', width=3, colors=LineColor)
ax1.ticklabel_format(axis='y', style='sci', useMathText=True, scilimits=(4,5))
ax1.set_xlabel('Time (yr)')
ax1.set_ylabel('Velocity (m/s)')
ax1.xaxis.label.set_color(LineColor)
ax1.yaxis.label.set_color(LineColor)

# height Plot
ax2 = fig.add_subplot(gs[1, 2:])
ax2.set_facecolor(COLOR)
r = np.sqrt(x**2 + y**2)
h = r-R_E
heightline, = ax2.plot(t[0], h[0], color='silver')
ax2.spines['bottom'].set_color(LineColor)
ax2.spines['top'].set_color(LineColor) 
ax2.spines['right'].set_color(LineColor)
ax2.spines['left'].set_color(LineColor)
ax2.set_xlim([0, t[-1]])
ax2.set_ylim([0, np.max(h)*1.2])
ax2.tick_params(labelcolor=LineColor, labelsize='medium', width=3, colors=LineColor)
ax2.ticklabel_format(style='sci', useMathText=True)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Height (m)')
ax2.xaxis.label.set_color(LineColor)
ax2.yaxis.label.set_color(LineColor)

#%% Suptitle

Range = R_E*np.arccos(np.dot([x[0], y[0]], [x[-1],y[-1]])/R_E**2)/1000 # in km
v_f = np.sqrt(vx[-2]**2+vy[-2]**2)/340 # in Mach
fig.suptitle('Launch angle {:.0f} degree, Range: {:.0f} km, impact velocity M{:.1f}'.format(EA, Range, v_f), color='silver')

plt.tight_layout()
#%%
ms2AUyr = 86400*365/1.5e11
def update(i):
    dot.set_data(x[i], y[i])
    line.set_data(x[:i], y[:i])
    velline.set_data(t[:i], np.sqrt(vx[:i]**2+vy[:i]**2))
    heightline.set_data(t[:i], h[:i])
    if Tracing:
        #ax.set_xlim([-1.5*r,1.5*r])
        #ax.set_ylim([-1.5*r,1.5*r])
        ax.set_xlim([x[i]-1e5,x[i]+1e5])
        ax.set_ylim([y[i]-1e5,y[i]+1e5])
        #ax.set_xlim([np.min(x)-1e5,np.max(x)+1e5])
        #ax.set_ylim([np.min(y)-1e5,np.max(y)+1e5])
    O1 = ax.add_patch(earth)
    if SAVE_VIDEO:
        print(i)
        fig.savefig('./images/{:04d}.jpg'.format(i), dpi=300)
    return [dot, line, velline, heightline, O1]

if SAVE_VIDEO:
    for i in range(frames):
        update(i)
else:
    ani = FuncAnimation(fig=fig, 
                        func=update,
                        frames=frames, 
                        interval=10000/frames, 
                        blit=True, 
                        repeat=False)
    plt.show()
