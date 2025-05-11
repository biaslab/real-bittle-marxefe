#!/usr/bin/python3
#  -*- coding: UTF-8 -*-

# MindPlus
# Python
from OpenCatPythonAPI.PetoiRobot import *
from MARXAgents import MARXAgent
import datetime as dtime
import numpy as np
import numpy.random as rnd
import pandas as pd
import timeit
from scipy.stats import multivariate_normal
from scipy.linalg import inv,det


if __name__ == '__main__':

    # Time
    len_trial = 10
    len_horizon = 2;

    # Namestamps
    now = dtime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    fname_params = "results/params/agentparams-$now"
    fname_agents = "results/agents/agent-$now"
    fname_trials = "results/trials/agenttrial-$now"

    # Dimensionalities
    Mu = 2 # includes current control uk
    My = 1
    Dy = 6 
    Du = 8
    Dx = My*Dy + Mu*Du

    # Prior parameters
    Nu0 = 100.
    Omega0 = 1e0*np.diag(np.ones(Dy))
    Lambda0 = 1e-3*np.diag(np.ones(Dx))
    Mean0 = 1e-8*rnd.randn(Dx,Dy)
    Upsilon0  = 1e-2*np.diag(np.ones(Du))

    # Setpoint (desired observation)
    m_star = np.zeros(Dy) # [roll, pitch, yaw, a_x, a_y, a_z]
    v_star = np.ones(Dy)
    goal   = multivariate_normal(m_star, np.diag(v_star))

    # Boot up agent
    agent = MARXAgent(Mean0, Lambda0, Omega0, Nu0, Upsilon0, goal, Dy=Dy, Du=Du, delay_inp=Mu, delay_out=My, time_horizon=len_horizon)

    # Preallocate
    times       = np.zeros((len_trial))
    preds_m     = np.zeros((Dy,len_trial))
    preds_S     = np.zeros((Dy,Dy,len_trial))
    preds_v     = np.zeros((Dy,len_trial))
    y_          = np.zeros((Dy,len_trial))
    u_          = np.zeros((Du,len_trial))
    F_          = np.zeros((len_trial))
    Means       = np.zeros((Dx,Dy,len_trial))
    Lambdas     = np.zeros((Dx,Dx,len_trial))
    Omegas      = np.zeros((Dy,Dy,len_trial))
    Nus         = np.zeros((len_trial))

    goodPorts = {}
    try:

        connectPort(goodPorts)    

        # Start the stopwatch
        start_time = timeit.default_timer()
        for k in range(2,len_trial):
            times[k] = timeit.default_timer()

            "Predict observation"
        
            # Calculate posterior predictive
            x_k = np.concatenate([agent.ubuffer.flatten(), agent.ybuffer.flatten()])
            eta_k, mu_k, Psi_k = agent.posterior_predictive(x_k)
            preds_m[:,k] = mu_k
            preds_S[:,:,k] = inv(Psi_k)*eta_k/(eta_k-2)
            preds_v[:,k] = np.diag( inv(Psi_k)*eta_k/(eta_k-2) )
            
            "Interact with environment"

            # Update system with selected control
            actions = [ 8, u_[0,k-1].astype(int), 
                       12, u_[1,k-1].astype(int),
                        9, u_[2,k-1].astype(int),
                       13, u_[3,k-1].astype(int),
                       11, u_[4,k-1].astype(int),
                       15, u_[5,k-1].astype(int),
                       10, u_[6,k-1].astype(int),
                       14, u_[7,k-1].astype(int)]
            send(goodPorts, ['I', actions, 1])
        
            # Read IMU
            serial_IMU = send(goodPorts, ['v', 0])
            ypr_acc = serial_IMU[1].split()
            y_[:,k] = np.array(ypr_acc)
                    
            "Parameter estimation"

            # Update parameters
            agent.update(y_[:,k], u_[:,k])

            # Track parameter beliefs
            Means[:,:,k]    = agent.M
            Lambdas[:,:,k]  = agent.Λ
            Omegas[:,:,k]   = agent.Ω
            Nus[k]          = agent.ν 

            # Track free energy
            F_[k] = agent.free_energy

            "Action selection"
            
            # Random actions
            u_[:,k] = np.clip(10*rnd.randn(8), a_min=-60, a_max=60).astype(int)

        closeAllSerial(goodPorts)

        print("Saving results..")
        pd.DataFrame(times).to_csv("results/trials/agent-random-" + now + "-times.csv"); time.sleep(3.0)
        pd.DataFrame(y_).to_csv("results/trials/agent-random-" + now + "-outputs.csv"); time.sleep(3.0)
        pd.DataFrame(u_).to_csv("results/trials/agent-random-" + now + "-inputs.csv"); time.sleep(3.0)
        pd.DataFrame(F_).to_csv("results/trials/agent-random-" + now + "-free_energy.csv"); time.sleep(3.0)
        pd.DataFrame(preds_m).to_csv("results/trials/agent-random-" + now + "-preds_m.csv"); time.sleep(3.0)
        pd.DataFrame(preds_v).to_csv("results/trials/agent-random-" + now + "-preds_v.csv"); time.sleep(3.0)
        # pd.DataFrame(Means).to_csv("results/params/agent-random-" + now + "-Means.csv"); time.sleep(3.0)
        # pd.DataFrame(Lambdas).to_csv("results/params/agent-random-" + now + "-Lambdas.csv"); time.sleep(3.0)
        # pd.DataFrame(Omegas).to_csv("results/params/agent-random-" + now + "-Omegas.csv"); time.sleep(3.0)
        # pd.DataFrame(Nus).to_csv("results/params/agent-random-" + now + "-Nus.csv"); time.sleep(3.0)
        print("Done")
        print("Experiment completed.")

    except:
        closeAllSerial(goodPorts)

