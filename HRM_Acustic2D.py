# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 19:50:25 2021

@author: JuanBarrios
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import HRM_Acustic1D as RA

def dominio(xmin, xmax, numvolumsx, ymin, ymax, numvolumsy):
    dx = (xmax - xmin)/numvolumsx
    dy = (ymax - ymin)/numvolumsy
    xm = np.arange(xmin + dx, xmax, dx)
    ym = np.arange(ymin + dy, ymax, dy)
    Xm, Ym = np.meshgrid(xm, ym)
    return xm, dx

def BC(qb, t):
    qb[0,0] = qb[0,1]
    # qb[:,-1] = qb[:,1]
    u0 = 0.2*(1. + np.cos(np.pi*(t-30.)/30.))
    if t<=60:
        qb[1,0] = Rho[0]*u0
    else:
        qb[1,0] = 0.
    return qb


NV = 600
NL = 2
TL = NV//NL
t = 0.
T = 170.
MaxDer = 1.
# xm, dx = dominio(0.,300.,NV)
x, xm, dx = RA.xxm(0.,300.,NV)
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
Rho = RA.rho(xm, TL, NV)
K = RA.k(xm, TL, NV)
Nt = int(round(T/dt))

for i in range(Nt):
    t += dt
    R = RA.RiemmanAcusticLineal(q, K, Rho, NV)
    qb[:,1:-1] = q[:,1:-1] - dt/dx * (R[4][:,:-1] + R[3][:,1:])
    F = RA.Fimm(R[2],R[0],R[1],6, dt, dx)
    qb[:,2:-2] = qb[:,2:-2] - dt/dx * (F[0][:,1:] - F[0][:,0:-1])
    qb = RA.BC(qb, t, Rho)
    q = np.copy(qb)
    

HRMPlot1, = plt.plot(xm,q[0,:],'-',label='Ref',color = 'black')  
# print(dominio(0,1,10,0,1,10))
# fig = plt.figure()
# ax = Axes3D(fig)
# X = np.arange(-4, 4, 0.25)
# Y = np.arange(-4, 4, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X ** 2 + Y ** 2)
# Z = np.sin(R)

# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
# ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
# ax.set_zlim(-2, 2)

# plt.show()

# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')