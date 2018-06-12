#!/usr/bin/env python

################## Authors
## Brian Armstrong,		github.com/molomono
## Edwin Van Emmerik, 	github.com/edwinvanemmerik

import pandas as pd

from numpy import *             # Grab all of the NumPy functions
from matplotlib.pyplot import * # Grab MATLAB plotting functions

num_of_simulations  = 10

run_once            = True #Turn this off if step_start_states is True
show_simulation     = True
show_phase_space	= True
save_output_files   = False
random_start_states = False
step_start_states 	= False #used for sweeping through start states, Is used to plot nicer phase space graphs

control             = True #PID controllers
energy_control      = True #Swingup controllers
impulse_on          = False #Impulse at t=0 to analyize system characteristics (turn off controllers if you use this)
max_u               = 0.3 #max controller action (saturation estimated using a bump test but more thorough testing is neded)

disturbance         = False

toggle_angle        = 0.3

#list for generating Phase Space plots
Y_list_0 = []
Y_list_1 = []
# System dynamics
# Gravity and viscous friction
g  = 9.81;                #Gravity
dw = 0.002;               #friction reaction wheel
dp = 0.0275;              #friction pendulum

# Mass properties
mw = 0.34;                #mass reaction wheel
mp = 0.1;                 #mass pendulum
me = 0.42;                #mass electronics
ma = 0.070;               #mass of the actuator

#mt is the mass_total however one of the masses is located on the mirroring side of the pendulum
#me results in a torque in the oposite direction of the rest, and therefore:
# T_gravity = (mw+mp+ma)*g*sin(theta_p) - me*g*sin(theta_p)
# T_gravity = (mw+mp+ma-me)*g*sin(theta_p)
mt = mw + mp + ma - me

# Length dimensions
r   = 0.06;             #radius of reaction wheel
ra  = 0.012;            #radius of the actuator
lw  = 0.2;              #distance between wheel and pendulum axles
lpc = 0.02;             #distance to center of mass pendulum
le  = -0.10;            #distance to center of mass electronics

#system Inertias
Iw      = mw*r**2;                                        #MOI of reaction wheel
Ie      = me*le**2;                                       #MOI of the electronics
Ip_na   = mp*lpc**2 + mw*lw**2 + ma*lw**2 ;           #MOI of pendulum minus actuator MOI
Ia      = ma*ra**2;                            #MOI of actuator

Ip      = Ip_na + Ia + Ie;                                      #MOI of pendulum plus actuator and wheel MOI
It      = Iw + Ip;                                   #MOI total system

#State space model non-linear
x = zeros([4,1])

fx = matrix([   [x[1]],
                [(mt * g * sin(x.item(0)) - dp * x.item(1) + dw * x.item(3))/(It - Iw)],
                [x[3]],
                [(-mt * g * sin(x.item(0)) - dp * x.item(1) - (dw * It * x.item(3))/Iw)/(It + Iw)] ])

gx = matrix([   [0],
                [-1/(It - Iw)],
                [0],
                [(It/Iw)/(It + Iw)] ])

C = eye(4)

# Simulation part
# This for loop is to generate files to be used as datasets for the neural network backwards propogation
iteration_sim = 0
for file_num in range(num_of_simulations):
    #
    # Construct inputs and outputs corresponding to steps in xy position
    #
    if impulse_on: #if impulse is turned on an impulse is applied when the simulation starts
        u = 1.0
    else:
        u = 0.0

    max_time = 15    # run the simulation 'max_time' seconds
    dt = 0.001 # 1000hz updates

    x   = transpose(matrix([3.11,0,0,0]))
    x_d = zeros([4,1])
    y   = zeros([4,1])

    if step_start_states: #this is to sweep through the start states to generate multiple responses for the same plot (mainly used for phase space)
      	x[0] = linspace(3.1415/2, 3.1415+3.1415/2, num_of_simulations)[file_num]

    if random_start_states: #Random start states, used to evaluate reliability of controller depending on realistic starting conditions
        u = random.random_sample()*3

        if random.randint(2):
            u = u * -1
            ### Random system state start (random.randint(2, size=10)) creates array of size 10 with random integers 0 and 1
            x    = transpose(divide(matrix(random.randint(100*3.1415*2, size=4), dtype = float), 100))
            #make polarity positive and negative (shift the phase 180 degrees)
            x = x - 3.1415

        #Construct lists used for ploting data
    U = []
    E = []
    X = []
    Y = empty([4,int(max_time/dt)])

    #Controller parameters
    p_s  = 1;
    kp_p = 50;
    ki_p = 75;
    kd_p = 2;

    ki_w = 0.01;
    kd_w = 0.00;
    kp_w = 0.01;

    #setpoint used for both wheel and pendulum
    setpoint  = 0.0;
    y_dot = zeros(shape=[4,1]);

    ##Start simulation using discrete state space model
    iteration = 0
    u_old = 0
    for T in linspace(0,max_time,max_time/(dt)):

        #Logic statement described in the gain scheduler chapter, if the conditions are met the PID controllers are run
        if control and not (sin(toggle_angle)**2 < sin(x.item(0))**2 or cos(x.item(0)) <= 0):
            #pendulum control
            u_p = kp_p*sin(setpoint - x.item(0))
            u_p = u_p + ki_p*sin(setpoint -(x.item(0) + x.item(1)*dt) )
            u_p = u_p + kd_p*sin(setpoint - x.item(1))

            #wheel control
            u_w = ki_w*(setpoint -(x.item(3) + y_dot.item(3)*dt))
            u_w = u_w + kp_w*(setpoint - x.item(3))
            u_w = u_w + kd_w*(setpoint - y_dot.item(3))

            u = p_s*u_w + u_p

        u = u*-1;

        #Energy control turns off in the region close to the equilibirium states, an extra condition that must be met is that the imaginary parameters
        #of the angle must be positive for it to turn off. This keeps the energy controller on if the pendulum is near it's equilibirium in negative imaginary plane.
        if energy_control and (sin(toggle_angle)**2 < sin(x.item(0))**2 or cos(x.item(0)) <= 0):
            lma = 0.23
            ma = 0.12
            K = 10
            theta     = x.item(0)
            theta_dot = x.item(1)
            u = -K*theta_dot*(0.5*ma*lma**2*theta_dot**2 + ma*g*lma*(1+np.cos(theta)) + 2*ma*g*lma)


        lma = 0.23
        ma_es = 0.1
        #Record the energy error used for controller action of the Energy-Shaping controller
        E.append( (0.5*ma_es*lma**2*x.item(1)**2) + ma_es*g*lma*(1+np.cos(x.item(0))) - 2*ma_es*g*lma)


        #controller saturation
        if u < -max_u:
            u = -max_u
        if u > max_u:
            u = max_u

        #Discretize the controller to update control action at 10 Hz
        if int(T*1000)%10 != 0:
            u = u_old
        #remember the old controller action, this maintains conroller action between controller actions coming in at 10 Hz
        u_old = u

        #Nonlinear discretized state-space simulation
        x_d = matrix([ [x.item(1)],
                    [(mt * g * sin(x.item(0)) - dp * x.item(1) + dw * x.item(3))/(It - Iw)],
                    [x.item(3)],
                    [(-mt * g * sin(x.item(0)) - dp * x.item(1) - (dw * It * x.item(3))/Iw)/(It + Iw)] ]) + gx*u
        x   = x_d*dt + x
        y   = C*x
        y_dot = C*x_d #ouptuts the x_dot states for the controller (used for I control of the wheel velocity)

        #If disturbance is turned on it gets applied using this function
        if iteration == int(4/dt) and disturbance:
            x[1] = 3

        #Append data to be used for plotting
        U.append(u)
        X.append(T)
        Y[:, iteration] = transpose(y)

        iteration = iteration + 1

    iteration_sim = iteration_sim + 1
    ### Plot the simulations
    Y_list_0.append(Y[0,:])
    Y_list_1.append(Y[1,:])
    if show_simulation:
        #plotting ALL system states
        figure(1)
        subplot(221)
        plot( X, Y[2,:])
        grid(True)
        title("Position Wheel")
        xlabel("Time, Seconds")
        ylabel("Radians")
        subplot(222)
        plot( X, Y[3,:])
        grid(True)
        title("Velocity Wheel")
        xlabel("Time, Seconds")
        ylabel("Radians/second")
        subplot(223)
        plot( X, Y[0,:])
        grid(True)
        title("Position Pendulum")
        xlabel("Time, Seconds")
        ylabel("Radians")
        subplot(224)
        plot( X, Y[1,:])
        grid(True)
        title("Velocity Pendulum")
        xlabel("Time, Seconds")
        ylabel("Radians/second")

        #energy plot
        figure(2)
        plot(X, E)
        grid(True)
        title("Pendulum Energy Error")
        xlabel("Time, Seconds")
        ylabel("Energy error, Joules")
        #show()

    #Save plots to CSV files to be used for Neural Network training
    if save_output_files:
        # for file_num in range(number_of_simulations):
        #Save impulse --- Export simulation results
        file_name       = 'impulse_response_output_' + str(file_num) + '.csv'
        file_name_input = 'control_input_' + str(file_num) + '.csv'
        #print(file_name)
        df = pd.DataFrame(transpose(Y), columns=["position_p", "velocity_p", "position_w", "velocity_w"])
        df.to_csv(file_name, index=False)

        df = pd.DataFrame(transpose(U), columns=["u"])
        df.to_csv(file_name_input, index=False)

        #df = pd.DataFrame(Y2, columns=["column"])
        #df.to_csv('impulse_response.csv', index=False)


    if run_once: # If run_once is true the simulation loop exits here
        break

if show_phase_space: #If true plot the phase space, the number of trajectories plotted is equal to the number of simulations run
    figure(9)
    grid(True)
    title("Pendulum Joint, Phase Space plot")
    xlabel("Radians, pendulum position")
    ylabel("Radians/Second, pendulum Velocity")
    hold(True)
    for i in range(len(Y_list_0)):
            plot(Y_list_0[i], Y_list_1[i])

show()
