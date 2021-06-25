# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 12:56:56 2021

@author: JuanBarrios
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def p(xc):
    p = np.sqrt(abs(1.-(xc+3.)**2))*(xc>-4.)*(xc<-2.)
    # p = np.sin(xc)
    # p = 1.*(xc<=-0.) + 0.*(xc>-0.)
    return p


def pexac(xm,T):
    R = 0.5*(p(xm + T) + p(xm - T)) - 1/2.*(p(xm + T) - p(xm - T))
    return R

def uexac(xm,T):
    R = -1/2. * (p(xm + T) - p(xm - T)) + 1/2. *(p(xm + T) + p(xm - T))
    return R

def rho(xm, TL, NV):
    rhoa = 1.
    rhob = 3.
    rho = np.zeros(NV)
    a=0
    b=TL
    while b<=NV-TL:
        rho[a:b]=rhoa
        rho[a+TL:b+TL] = rhob
        a+=2*TL
        b+=2*TL
    return rho

def k(xm, TL, NV):
    rhoa = 1.
    rhob = 3.
    rho = np.zeros(NV)
    a=0
    b=TL
    while b<=NV-TL:
        rho[a:b]=rhoa
        rho[a+TL:b+TL] = rhob
        a+=2*TL
        b+=2*TL
    return rho

def RiemmanAcusticLineal(q,k,rho,n):
    c = np.sqrt(k/rho)
    z = c*rho
    r1 = np.array([-z[0:-1],np.ones(n-1)])
    r2 = np.array([z[1:],np.ones(n-1)])
    dq = q[:,1:]-q[:,0:-1]
    alpha1 = np.array(-dq[0,:]+z[1:]*dq[1,:])/(z[0:-1] + z[1:])
    alpha2 = np.array((dq[0,:]+z[0:-1]*dq[1,:])/(z[0:-1] + z[1:]))
    W1 = alpha1 * r1
    W2 = alpha2 * r2
    Am = -c[:-1]*W1
    Ap = c[1:]*W2
    return W1, W2, c, Am, Ap

def Fimm(s, W1, W2, L, dt, dx):
    F = np.copy(W1)
    norm = np.sum(W1[:,1:-1]**2,0)
    norm += (norm == 0. ) * 1.
    theta = np.sum(W1[:,2:]*W1[:,1:-1],0)/norm
    W1L = Limitador(theta, L)*W1[:,1:-1]
    norm = np.sum(W2[:,1:-1]**2,0)
    norm += (norm == 0 ) * 1.
    theta = np.sum(W2[:,0:-2]*W2[:,1:-1],0)/norm
    W2L = Limitador(theta, L)*W2[:,1:-1]
    F = ((np.abs(s[1:-2])*(1.-dt/dx*np.abs(s[1:-2]))*W1L) +
        (np.abs(s[2:-1])*(1.-dt/dx*np.abs(s[2:-1]))*W2L))
    F = F * 0.5
    return F, W1L, W2L, theta

def Limitador(theta, o):
    shape = np.shape(theta)
    if o == 0:
        phy = 0.
        return phy
    if o == 1:
        phy = 1.
        return phy
    if o == 2:
        phy = theta
        return phy
    if o == 3:
        phy = 0.5*(1.+theta)
        return phy
    if o == 4:
        theta1 = np.array([np.ones(shape), theta])
        phy = MinMod(theta1)
        return phy
    if o == 5:
        a = np.zeros(shape)
        b = np.ones(shape)
        c = np.min([b , 2.*theta],0)
        d = np.min([2.*b, theta],0)
        phy = np.max([a,c,d],0)
        return phy
    if o == 6:
        a = np.zeros(shape)
        b = np.ones(shape)
        c = np.min([(b + theta)/2. , 2.*theta, 2*b],0)
        phy = np.max([a,c],0)
        return phy
    if o == 7:
        phy = (theta + np.abs(theta))/(1 + np.abs(theta))
        return phy

def MaxMod(a):
    k1 = a > 0.
    k2 = a < 0.
    return (k1.all(0))*np.max(a,0) + (k2.all(0))*np.min(a,0)

def MinMod(a):
    k1 = a > 0.
    k2 = a < 0.
    return (k1.all(0))*np.min(a,0) + (k2.all(0))*np.max(a,0)

def xxm(LimInf, LimSup, NumVol):
    dx = (LimSup - LimInf)/NumVol
    x = np.zeros(NumVol+1)
    xm = np.zeros(NumVol)
    for i in range(NumVol+1):
        x[i] = LimInf + i*dx
    for i in range(NumVol):
        xm[i] = LimInf + (i + 0.5)*dx
    return x, xm, dx

def BC(qb, t, Rho):
    qb[0,0] = qb[0,1]
    # qb[:,-1] = qb[:,1]
    u0 = 0.2*(1. + np.cos(np.pi*(t-30.)/30.))
    if t<=60:
        qb[1,0] = Rho[0]*u0
    else:
        qb[1,0] = 0.
    return qb



NV = 1200
NL = 300
TL = NV//NL
t = 0.
T = 240.
MaxDer = 1.
x, xm, dx = xxm(0.,300.,NV)
# x, xm, dx = xxm(0.,300.,NV)
# x, xm, dx = xxm(-5.,5.,NV)
CFL = 0.1
# CFL = 1.
dt = (dx * CFL)/MaxDer
q = np.zeros((2,NV))
q[0,:] = 0.
q[1,:] = 0.
q[1,0] = 0.2*(1. + np.cos(np.pi*(t-30.)/30.))
qb = np.copy(q)
qb1 = np.copy(q)
Rho = rho(xm, TL, NV)
K = k(xm, TL, NV)
Nt = int(round(T/dt))
# plt.plot(xm,np.sin(xm-T))
# ini = time.time()
for i in range(Nt):
    t += dt
    R = RiemmanAcusticLineal(q, K, Rho, NV)
    qb[:,1:-1] = q[:,1:-1] - dt/dx * (R[4][:,:-1] + R[3][:,1:])
    F = Fimm(R[2],R[0],R[1],6, dt, dx)
    qb[:,2:-2] = qb[:,2:-2] - dt/dx * (F[0][:,1:] - F[0][:,0:-1])
    qb = BC(qb, t, Rho)
    q = np.copy(qb)
    
HRMPlot1, = plt.plot(xm,q[0,:],'-',label='Ref',color = 'black')  

