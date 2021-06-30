# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 21:28:29 2021

@author: u0139894
"""

import streamlit as stl


import numpy as np
import scipy.integrate as solver
import pandas as pd

def monod_g(s, mu_max, ks):
    
    return mu_max*(s/(s + ks))


def dynamics(t, vars_t, D, dr, s_I, mu_max, ks, y):
    s = vars_t[0]
    
    x = vars_t[1]
    
    
    xd = vars_t[2]
    
    mu = monod_g(s, mu_max, ks)
    
    dxddt = dr*x - D*xd
    
    dsdt = D*(s_I-s) -(mu/y)*x
    
    dxdt = (mu*x)-(D*x)-(dr*x)
    
    
    
    return np.array([dsdt, dxdt, dxddt])

def integrate(vars_init, t_start, t_end, t_interval, params, method = 'bdf' ):
    ode = solver.ode(dynamics)
    
    # BDF method suited to stiff systems of ODEs
    n_steps = (t_end - t_start)/t_interval
    
    ode.set_integrator('vode', nsteps=n_steps, method= method)
    
    # Time
    
    t_step = (t_end - t_start)/n_steps
    
    ode.set_f_params(params['D'], params['dr'], params['s_I'], params['mu_max'], params['ks'], params['y'])
    
    ode.set_initial_value(vars_init, t_start)
    
    t_s = []
    var_s = []
        
    while ode.successful() and ode.t < t_end:
            ode.integrate(ode.t + t_step)
            t_s.append(ode.t)
            var_s.append(ode.y)
    
    time = np.array(t_s)
    vars_total = np.vstack(var_s).T
    
    return time, vars_total


stl.sidebar.write('### Parameters')


stl.sidebar.write('**Potentially controlable**')

stl.sidebar.write('*-----------------------------------------*\n\n\n')
cd = stl.sidebar.slider('culture days', 1, 50, 5, step=1, format='%.3f')

stl.sidebar.write('*-----------------------------------------*\n\n\n')


s_0 = stl.sidebar.slider('Initial resource concentration (mM)', 0.001, 100.0, 5.55, step=0.001, format='%.3f')

stl.sidebar.write('The starting concentration of the limiting resourse in the chemostat vessel (Conc. of glucose in WC media = 5.5 mM)')

stl.sidebar.write('*-----------------------------------------*\n\n\n')

x_0 = stl.sidebar.slider('Initial bacterial concentration (k cells/ml)', 0.0001, 5.0, 0.001, step=0.001, format='%.3f')

stl.sidebar.write('k = $10^8$; I hope to soon have a conversion factor between OD and cell counts for each strain')

stl.sidebar.write('*-----------------------------------------*\n\n\n')

xd_0 = 0

D = stl.sidebar.slider('Dilution parameter (1/h)', 0.00001, 1.0, 0.07, step=0.0001, format='%.3f')

stl.sidebar.write('the maximum dilution allowed by the AMBR is approx. 0.07 (considering a vol of 10 mL and inflow of 0.7 ml/h)')

stl.sidebar.write('*-----------------------------------------*\n\n\n')

s_I = stl.sidebar.slider('Feed resource concentration (mM)', 0.001, 100.0, 5.55, step=0.001, format='%.3f')

stl.sidebar.write('*-----------------------------------------*\n\n\n')


stl.sidebar.write('**Hypothetical strain**')

stl.sidebar.write('*-----------------------------------------*\n\n\n')

mu_max = stl.sidebar.slider('Max growth rate (1/h)', 0.00001, 3.0, 1.0, step=0.0001, format='%.3f')

stl.sidebar.write('*-----------------------------------------*\n\n\n')

y = stl.sidebar.slider('yield (unitless)', 0.00001, 3.0, 0.5, step=0.0001, format='%.3f')

stl.sidebar.write('Although one can attribute units (e.g. (k cell/ml)/mM), they can ultimately be cancelled out')

stl.sidebar.write('*-----------------------------------------*\n\n\n')

ks = stl.sidebar.slider('Monod constant (mM)', 1.0, 100.0, 50.0, step=0.5, format='%.3f')

stl.sidebar.write('*-----------------------------------------*\n\n\n')

dr = stl.sidebar.slider('death rate (1/h)', 0.000001, 1.0, 0.01, step=0.00001, format='%.3f')

stl.sidebar.write('Assumed constant, but it is possibly higher at very low dilution rates')

vars_init = np.array([s_0, x_0, xd_0])
params = {'D':D, 'dr':dr, 's_I':s_I, 'mu_max':mu_max, 'ks':ks, 'y':y}

t, v = integrate(vars_init, 0, cd*24, 0.1, params)

data = pd.DataFrame(v.T, columns = ['S (mM)', 'LiveCells k cells/ml', 'DeadCells k cells/ml'], index=t)

stl.line_chart(data)
