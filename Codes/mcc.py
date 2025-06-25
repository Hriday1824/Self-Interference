# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:58:26 2022

@author: admin
"""

# joao.paulo.f.guimaraes@gmail.com
# Fixed point using the Maximum Correntropy Criteria
# 10/04.2018
import random
import math
import numpy as np
from scipy.stats import levy_stable
alpha, beta = 1.5, 1
r = levy_stable.rvs(alpha, beta, size=18412)
#r = np.load("sample.npy")
def mcc(x,d,sigma,w0):

    N = d.size
    print(np.shape(N))
    w = w0    
    wl = 13#w.shape

    # record the estimated w at each step
    aux = np.zeros((wl,N)) + 1j*np.zeros((wl,N));
    h_w = np.matrix(aux)
    h_w[:,0] = w0


    # auxiliar matrix
    R_ = 1e-4*np.matrix(np.eye(wl))
    P_ = 1e-4*np.matrix(np.ones((wl,1)))

    n=0
    while(n<N):
        
        # input data at iteration 'n'
        xn = x[:,n]; xn = xn.reshape(13,1)
        xn_H = np.conjugate(np.transpose(xn))#np.conjugate(np.transpose(xn))

        # desired signal at iteration 'n'
        dn = d[:,n]; 
        dn_c = np.conjugate(dn)
        
        # transposing w
        w_t = np.transpose(w)
        outlier=(1)*np.zeros([1,18412]);
        outlier=(1)*np.zeros([1,18412]);
        outlier[0,18000]=2000;

        # calculating error and the error conjugate
        e = dn - w_t.dot(xn)+r[n]+outlier[0,n];#+outlier[0,n];
        e_c = np.conjugate(e)
        
        # exponential part
        B = -0.5*(e*e_c)/(math.pow(sigma,2))
        mu=.01
        # bug: exp of complex = error, but e*e_c always real, i.e. [1+0j]
        B = np.real(B);
        expB = math.exp(B)       
        CC = expB/math.pow(sigma,2)
        w_t=w_t+1*mu*CC*((xn.T))  #np.conjugate    
        P = P_ + np.multiply(expB*dn_c,xn)
        R = R_ + expB*xn*xn_H; R_i = np.linalg.inv(R)
        
        w = R_i*P
        
        # old R and P updated with the new values
        R_ = R; P_ = P

        # updating the log variable
        if(n!=0):
            h_w[:,n] = w;

        n = n+1


       # hw=h_w[:,18412]
    
    return h_w
        
