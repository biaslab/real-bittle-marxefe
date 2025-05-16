#!/usr/bin/python3
#  -*- coding: UTF-8 -*-

# MindPlus
# Python
from OpenCatPythonAPI.PetoiRobot import *
from MARXAgents import MARXAgent
import datetime as dtime
import numpy as np
import numpy.random as rnd
import timeit
from scipy.stats import multivariate_normal
from scipy.linalg import inv,det
from scipy.optimize import minimize


if __name__ == '__main__':

    # Time
    len_trial = 100
    len_horizon = 2;
    now = dtime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Dimensionalities
    Mu = 2 # includes current control uk
    My = 2
    Dy = 6 
    Du = 8
    Dx = My*Dy + (Mu+1)*Du

    # Prior parameters
    Nu0 = 20.
    Omega0 = 1e0*np.diag(np.ones(Dy))
    Lambda0 = 1e-3*np.diag(np.ones(Dx))
    Mean0 = 1e-8*rnd.randn(Dx,Dy)
    Upsilon0  = 1e-4*np.diag(np.ones(Du))

    # Setpoint (desired observation)
    m_star = [0.0, -10., 0.0, 0.0, 0.0, 0.0] # [yaw, pitch, roll, a_x, a_y, a_z]
    v_star = [1e0, 1e-5, 1e0, 1e3, 1e3, 1e3]
    goal   = multivariate_normal(m_star, np.diag(v_star))

    # Control limits
    u_lims = (-20,20)
    policy = 10*rnd.randn((Du*len_horizon))

    # Boot up agent
    agent = MARXAgent(Mean0, Lambda0, Omega0, Nu0, Upsilon0, goal, Dy=Dy, Du=Du, delay_inp=Mu, delay_out=My, time_horizon=len_horizon)

    # Preallocate
    times       = np.zeros((len_trial))
    preds_m     = np.zeros((Dy,len_trial))
    preds_P     = np.repeat(np.expand_dims(np.eye(Dy), axis=2), len_trial, axis=2)
    preds_v     = np.zeros((Dy,len_trial))
    y_          = np.zeros((Dy,len_trial))
    u_          = np.zeros((Du,len_trial))
    FE_         = np.zeros((len_trial))
    goals_m     = np.zeros((Dy,len_trial))
    Means       = np.zeros((Dx,Dy,len_trial))
    Lambdas     = np.repeat(np.expand_dims(np.eye(Dx), axis=2), len_trial, axis=2)
    Omegas      = np.repeat(np.expand_dims(np.eye(Dy), axis=2), len_trial, axis=2)
    Nus         = np.zeros((len_trial))

    goodPorts = {}
    try:

        connectPort(goodPorts)    
        send(goodPorts, ['B', 0.0])
        # send(goodPorts, ['I', [8, 0, 12, 0, 9, 0, 13, 0, 11, 0, 15, 0, 10, 0, 14, 0], 0.0])

        # Start the stopwatch
        start_time = timeit.default_timer()
        for k in range(1,len_trial):
            times[k] = timeit.default_timer()
            logger.info(f"tstep = {k}/{len_trial}")

            # "Change goal prior"
            # if k == 100:
            #     m_star = [0.0, 10., 0.0, 0.0, 0.0, 0.0] # [yaw, pitch, roll, a_x, a_y, a_z]
            #     v_star = [1e0, 1e-5, 1e0, 1e3, 1e3, 1e3]
            #     goal   = multivariate_normal(m_star, np.diag(v_star))
            #     agent.goal_prior = goal
            # goals_m[:,k] = agent.goal_prior.mean

            "Predict observation"
        
            # Calculate posterior predictive
            x_k = np.concatenate([agent.ubuffer.flatten(), agent.ybuffer.flatten()])
            eta_k, mu_k, Psi_k = agent.posterior_predictive(x_k)
            preds_m[:,k] = mu_k
            preds_P[:,:,k] = Psi_k
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
            send(goodPorts, ['I', actions, 0.0])
        
            # Read IMU
            serial_IMU = send(goodPorts, ['v', 0.0])
            ypr_acc = serial_IMU[1].split()
            y_[:,k] = np.array(ypr_acc)

            # Normalize accelerations
            y_[3:,k] /= 1e6
                    
            "Parameter estimation"

            # Update parameters
            agent.update(y_[:,k], u_[:,k-1])

            # Track parameter beliefs
            Means[:,:,k]    = agent.M
            Lambdas[:,:,k]  = agent.Λ
            Omegas[:,:,k]   = agent.Ω
            Nus[k]          = agent.ν 

            # Track free energy
            FE_[k] = agent.free_energy
            logger.debug(FE_[k])

            "Action selection"
            
            # Call minimizer using constrained L-BFGS procedure
            bounds = [u_lims] * (Du*len_horizon)
            results = minimize(agent.EFE, policy, method='L-BFGS-B', bounds=bounds, options={'disp': True, 'maxiter': 10})
            
            # Extract minimizing control
            if k < 10:
                policy = 10*rnd.random((Du*len_horizon))
            else:
                policy = results.x
            u_[:,k] = np.clip(policy[:Du], a_min=u_lims[0], a_max=u_lims[1]).astype(int)
            # u_[:,k] = np.clip(10*rnd.randn(8), a_min=-30, a_max=30).astype(int)
            logger.info(u_[:,k])
            logger.debug(policy)


        closeAllSerial(goodPorts)
        logger.info("Ports closed.")

        print("Saving results..")
        np.save("results/trials/agent-MARXEFE-" + now + "-times.npy", times)
        np.save("results/trials/agent-MARXEFE-" + now + "-outputs.npy", y_)
        np.save("results/trials/agent-MARXEFE-" + now + "-inputs.npy", u_)
        np.save("results/trials/agent-MARXEFE-" + now + "-preds_m.npy", preds_m)
        np.save("results/trials/agent-MARXEFE-" + now + "-preds_v.npy", preds_v)
        np.save("results/trials/agent-MARXEFE-" + now + "-preds_P.npy", preds_P)
        np.save("results/trials/agent-MARXEFE-" + now + "-goals_m.npy", goals_m)
        np.save("results/params/agent-MARXEFE-" + now + "-Means.npy", Means)
        np.save("results/params/agent-MARXEFE-" + now + "-Lambdas.npy", Lambdas)
        np.save("results/params/agent-MARXEFE-" + now + "-Omegas.npy", Omegas)
        np.save("results/params/agent-MARXEFE-" + now + "-Nus.npy", Nus)
        print("Experiment completed.")

    except Exception as e:
        logger.info("Exception")
        logger.info(e)
        closeAllSerial(goodPorts)
        os._exit(0)

