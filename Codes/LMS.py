# # -*- coding: utf-8 -*-
# """
# Created on Wed Jun 22 16:58:26 2022

# @author: admin
# """

# # joao.paulo.f.guimaraes@gmail.com
# # Fixed point using the Maximum Correntropy Criteria
# # 10/04.2018
# import random
# import math
# import numpy as np
# from scipy.stats import levy_stable
# alpha, beta = 1, 0
# r = levy_stable.rvs(alpha, beta, size=18412)
# #r = np.load("sample.npy")
# def LMS(x,d):

#     N = d.size
#     print(np.shape(N))
#     w =  np.zeros([13,1])  #0.001*np.random.randn(13,1) +1j*0.001*np.random.randn(13,1) #np.zeros([13,1])  
#     wl = 13#w.shape
#     R_ = 1e-4*np.matrix(np.eye(wl))
#     P_ = 1e-4*np.matrix(np.ones((wl,1)))
#     # record the estimated w at each step
#     aux = np.zeros((wl,N)) + 1j*np.zeros((wl,N));
#     h_w = np.matrix(aux)
#     h_w[:,0] = w
#     mu=0.01
#     n=0
#     while(n<N):
        
#         # input data at iteration 'n'
#         xn = x[:,n]; xn = xn.reshape(13,1)
#         xn_H = np.conjugate(np.transpose(xn))

#         # desired signal at iteration 'n'
#         dn = d[:,n]; 
#         dn_c = np.conjugate(dn)
        
#         # transposing w
#         w_t=np.transpose(w)
#         #w_t = np.transpose(w)
#         outlier=(1)*np.zeros([1,18412])
#         outlier[0,18000]=2000
#         # calculating error and the error conjugate
#         e = dn - (w_t).dot((xn))+r[n]+outlier[0,n]
#         e_c = np.conjugate(e)
#         #e = np.conjugate(e)
#         w_t=w_t+2*mu*(e_c)*((xn.T))  #np.conjugate
#         # updating the log variable
#         B= e*e_c; B=np.real(B)
#         B=B.item(0)
#         P = P_ + np.multiply(B*dn_c,xn)
#         R = R_ + B*xn*xn_H
#         R_i = np.linalg.inv(R)
        
#         w = R_i*P
        
#         # old R and P updated with the new values
#         R_ = R; P_ = P
#         if(n!=0):
#             h_w[:,n] = w_t.T

#         n = n+1


#        # hw=h_w[:,18412]
    
#     return h_w
        
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

alpha, beta = 1, 0
r = levy_stable.rvs(alpha, beta, size=18412)

def LMS(x, d):
    N = d.size
    w = np.zeros([13, 1])
    wl = 13
    R_ = 1e-4 * np.matrix(np.eye(wl))
    P_ = 1e-4 * np.matrix(np.ones((wl, 1)))
    aux = np.zeros((wl, N)) + 1j * np.zeros((wl, N))
    h_w = np.matrix(aux)
    h_w[:, 0] = w
    mu = 0.01
    n = 0

    while n < N:
        xn = x[:, n].reshape(13, 1)
        xn_H = np.conjugate(np.transpose(xn))
        dn = d[:, n]
        dn_c = np.conjugate(dn)
        w_t = np.transpose(w)
        outlier = (1) * np.zeros([1, 18412])
        outlier[0, 18000] = 2000
        e = dn - (w_t).dot((xn)) + r[n] + outlier[0, n]
        e_c = np.conjugate(e)
        w_t = w_t + 2 * mu * (e_c) * ((xn.T))
        B = e * e_c
        B = np.real(B)
        B = B.item(0)
        P = P_ + np.multiply(B * dn_c, xn)
        R = R_ + B * xn * xn_H
        R_i = np.linalg.inv(R)
        w = R_i * P
        R_ = R
        P_ = P
        if n != 0:
            h_w[:, n] = w_t.T
        n = n + 1

    return h_w