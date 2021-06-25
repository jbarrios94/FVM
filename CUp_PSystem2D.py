# -*- coding: utf-8 -*-
"""
Created on Sat May  8 10:52:47 2021

@author: JuanBarrios
"""

import numpy as np
import matplotlib.pyplot as plt
from visclaw import colormaps as cm

""""
all_white 
all_light_red 
all_light_blue 
all_light_green 
all_light_yellow 

red_white_blue 
blue_white_red 
red_yellow_blue 
blue_yellow_red 
yellow_red_blue 
white_red 
white_blue 

schlieren_grays 
schlieren_reds 
schlieren_blues
schlieren_greens
"""

def Dominio(LimInfx, LimSupx, LimInfy, LimSupy, NumVolx, NumVoly):
    #Funcion para generar Dominio computacional
    #LimInf es el limite inferior
    #LimSup es el limite superior
    #NumVol es el numero de volumenes (celdas)
    dx = (LimSupx - LimInfx)/NumVolx
    dy = (LimSupy - LimInfy)/NumVoly
    xn = np.linspace(LimInfx, LimSupx, NumVolx + 1)
    yn = np.linspace(LimInfy, LimSupy, NumVoly + 1)
    xc = np.zeros(NumVolx + 4)
    yc = np.zeros(NumVoly + 4)
    xc[2:-2] = xn[:-1]
    yc[2:-2] = yn[:-1]
    Xc, Yc = np.meshgrid(xc,yc)
    Xn, Yn = np.meshgrid(xn, yn)
    return Xc, Yc, Xn, Yn, dx, dy

def Condicion_Inicial(Xc, Yc, Xn, Yn, NumVolx, NumVoly):
    q = np.zeros((3,NumVoly + 4,NumVolx + 4))
    #Teste1-------------------------------------------------------------------
    # x0 = -0.5; y0 = 0.
    # width = 0.1; rad = 0.25
    # r = np.sqrt((Xc - x0)**2 + (Yc - y0)**2)
    # q[0] = (np.abs(r - rad)<=width)* (1. + np.cos(np.pi*(r-rad)/width))
    #-------------------------------------------------------------------------
    #teste3-------------------------------------------------------------------
    r = np.sqrt(Xc**2 + Yc**2)
    q[0] = (np.abs(r-0.5)<=0.2)*(1.+np.cos(np.pi*(r-0.5)/0.2))
    #Teste2-------------------------------------------------------------------
    # zz = np.array([Xc>-0.35, Xc<-0.2])
    # zz = zz.all(0)
    # q[0] = -1*zz
    # q[1] = 1*zz
    # q[2] = 0*zz
    return q

def velocidades(Rho, K):
    c = np.sqrt(K/Rho)
    return c

def Parametros_Dominio(Xc, Yc,rho_a, rho_b, k_a, k_b, NumVolx, NumVoly):
    #Teste1-------------------------------------------------------------------
    #Medio Heterogeneo--------------------------------------------------------
    # rho = rho_a * (Xc<0) + rho_b * (Xc>=0)
    # k = k_a * (Xc<0) + k_b * (Xc>=0)
    #Medio Homogeneo----------------------------------------------------------
    rho = rho_a * (Xc<0) + rho_a * (Xc>=0)
    k = k_a * (Xc<0) + k_a * (Xc>=0)
    #Teste2-------------------------------------------------------------------
    # rho = np.zeros(np.shape(Xc))
    # k = np.zeros(np.shape(Xc))
    # for j in range(NumVoly + 4):
    #     for i in range(NumVolx + 4):
    #         if i < np.max([(NumVolx + 4 )//2,j]) :   
    #             rho[j][i] = rho_a
    #             k[j][i] = k_b
    #         if i >= np.max([(NumVolx + 4)//2,j]) :   
    #             rho[j][i] = rho_b
    #             k[j][i] = k_b
    #         if j >= (NumVoly + 4)//2 and i==j :   
    #             rho[j][i] = (rho_a + rho_b)/2.
    #             k[j][i] = 1/(2*(k_a + k_b))
    return rho, k
    
def fluxo(u,i,axe):
    f = np.copy(u)
    if axe == 1:
        f[0,:,:] = -u[1,:,:]/Rho[:,i[0]:i[1]]
        f[1,:,:] = -u[0,:,:]*K[:,i[0]:i[1]]
        f[2,:,:] = 0.
    elif axe == 2:
        f[0,:,:] = -u[2,:,:]/Rho[i[0]:i[1],:]
        f[1,:,:] = 0.
        f[2,:,:] = -u[0,:,:]*K[i[0]:i[1],:]
    return f
    
def Limitador(cp,cm,cpm,lim, axe):
    if lim == 0:
        return 0.
    if lim == 1:
        return MinMod(np.array([cp/dx,cm/dx]),axe)
    if lim == 2:
        sig1 = MinMod(np.array([2.*cm/dx,cp/dx]), axe)
        sig2 = MinMod(np.array([cm/dx,2.*cp/dx]), axe)
        return MaxMod(np.array([sig1,sig2]), axe)
    if lim == 3:
        return MinMod(np.array([2.*cm/dx, cpm/(2.*dx), 2.*cp/dx]),axe)

def MaxMod(a, axe):
    k1 = a > 0.
    k2 = a < 0.
    return (k1.all(0))*np.max(a,0) + (k2.all(0))*np.min(a,0)

 	
def MinMod(a, axe):
    k1 = a > 0.
    k2 = a < 0.
    return (k1.all(0))*np.min(a,0) + (k2.all(0))*np.max(a,0)

def CUp(q, c, lim):
    #reconstruccion en x
    cpx = q[:,:,2:] - q[:,:,1:-1] 
    cmx = q[:,:,1:-1] - q[:,:,:-2]
    cpmx = q[:,:,2:] - q[:,:,:-2]
    qW = q[:,:,1:-1] - dx/2.*  Limitador(cpx, cmx, cpmx, lim, 1)
    qE = q[:,:,1:-1] + dx/2. *  Limitador(cpx, cmx, cpmx, lim, 1)  

    #reconstuccion y
    cpy = q[:,2:,:] - q[:,1:-1,:] 
    cmy = q[:,1:-1,:] - q[:,:-2,:]
    cpmy = q[:,2:,:] - q[:,:-2,:]
    qS = q[:,2:-1,:] - dy/2.*  Limitador(cpy[:,1:,:],cmy[:,1:,:],cpmy[:,1:,:],lim, 2)
    qN = q[:,1:-2,:] + dy/2.*  Limitador(cpy[:,:-1,:],cmy[:,:-1,:],cpmy[:,:-1,:],lim, 2)
    
    
    qNE = qE[:,1:-1,:] + dy/2.*  Limitador(cpy[:,:,1:-1],cmy[:,:,1:-1],cpmy[:,:,1:-1],lim, 2)
    qNW = qW[:,1:-1,1:] + dy/2.*  Limitador(cpy[:,:,2:-1],cmy[:,:,2:-1],cpmy[:,:,2:-1],lim, 2)
    qSE = qE[:,2:-1,:] - dy/2.*  Limitador(cpy[:,1:,1:-1],cmy[:,1:,1:-1],cpmy[:,1:,1:-1],lim, 2)
    qSW = qW[:,2:-1,1:] - dy/2.*  Limitador(cpy[:,1:,2:-1],cmy[:,1:,2:-1],cpmy[:,1:,2:-1],lim, 2)

    ap = np.maximum(c[:,1:-2], np.maximum(c[:,2:-1], 0.))
    am = np.minimum(-c[:,1:-2], np.minimum(-c[:,2:-1], 0.))
    
    bp = np.maximum(c[1:-2,:], np.maximum(c[2:-1,:], 0.))
    bm = np.minimum(-c[1:-2,:], np.minimum(-c[2:-1,:], 0.))
    
    apmam = ap - am
    bpmbm = bp - bm
    
    Wintx = (ap*qW[:,:,1:] - am*qE[:,:,:-1] - (fluxo(qW[:,:,1:],[2,-1], 1) - fluxo(qE[:,:,:-1],[1,-2], 1)))
    Wintx /= apmam
    T1 = (qNW[:,2:,:] - Wintx[:,2:-2,:])/apmam[2:-2,:]
    T2 = (Wintx[:,2:-2,:] - qNE[:,2:,:-1])/apmam[2:-2,:]
    T3 = (qSW[:,1:,:] - Wintx[:,2:-2,:])/apmam[2:-2,:]
    T4 = (Wintx[:,2:-2,:] - qSE[:,1:,:-1])/apmam[2:-2,:]
    qchiux = MinMod(np.array([T1,T2,T3,T4]), 1)
    
    Winty = (bp*qS - bm*qN - (fluxo(qS,[2,-1], 2) - fluxo(qN,[1,-2], 2)))
    Winty /= bpmbm
    T1 = (qSW[:,:,1:] - Winty[:,:,2:-2])/bpmbm[:,2:-2]
    T2 = (Winty[:,:,2:-2] - qNW[:,:-1,1:])/bpmbm[:,2:-2]
    T3 = (qSE[:,:,1:-1] - Winty[:,:,2:-2])/bpmbm[:,2:-2]
    T4 = (Winty[:,:,2:-2] - qNE[:,:-1,2:])/bpmbm[:,2:-2]
    qchiuy = MinMod(np.array([T1,T2,T3,T4]), 2)

    apxam = ap * am
    bpxbm = bp * bm
    Hx = (ap*fluxo(qE[:,:,:-1],[1,-2], 1) - am*fluxo(qW[:,:,1:],[2,-1], 1))[:,2:-2,:]
    Tcx = ((qW[:,:,1:] - qE[:,:,:-1])/apmam)[:,2:-2,:] - qchiux
    Hx = Hx/apmam[2:-2,:] + (apxam[2:-2,:] * Tcx)
    Hx = - (Hx[:,:,1:] - Hx[:,:,:-1])/dx
    
    Hy = (bp*fluxo(qN,[1,-2], 2) - bm*fluxo(qS,[2,-1], 2))[:,:,2:-2]
    Tcy = ((qS - qN)/bpmbm)[:,:,2:-2] - qchiuy
    Hy = Hy/bpmbm[:,2:-2] + (bpxbm[:,2:-2] * Tcy)
    Hy = - (Hy[:,1:,:] - Hy[:,:-1,:])/dx
    
    H = Hx + Hy
    return H

def BCx(q, op, op1):
    if op1 == 0:
        if op == 1:
            q[:,:,0] = q[:,:,-4]
            q[:,:,1] = q[:,:,-3]
        if op == 2:
            q[:,:,0] = q[:,2,:]
            q[:,:,1] = q[:,:,2]
            return q
        if op == 3:
            q[:,:,1] = q[:,:,2]
            q[:,:,0] = q[:,:,3]
            q[1,:,0] = -q[1,:,0] 
            q[1,:,1] = -q[1,:,1]
            return q
    if op1 == 1:
        if op == 1:
            q[:,:,-1] = q[:,:,3]
            q[:,-2,:] = q[:,:,2]
        if op == 2:
            q[:,:,-1] = q[:,:,-3]
            q[:,:,-2] = q[:,:,-3]
            return q
        if op == 3:
            q[:,:,-2] = q[:,:,-3]
            q[:,:,-1] = q[:,:,-4]
            q[2,:,-1] = -q[1,:,-1] 
            q[2,:,-2] = -q[1,:,-2]
            return q

def BCy(q, op, op1):
    if op1 == 0:
        if op == 1:
            q[:,0,:] = q[:,-4,:]
            q[:,1,:] = q[:,-3,:]
        if op == 2:
            q[:,0,:] = q[:,2,:]
            q[:,1,:] = q[:,2,:]
            return q
        if op == 3:
            q[:,1,:] = q[:,2,:]
            q[:,0,:] = q[:,3,:]
            q[2,0,:] = -q[2,0,:] 
            q[2,1,:] = -q[2,1,:]
            return q
    if op1 == 1:
        if op == 1:
            q[:,-1,:] = q[:,3,:]
            q[:,-2,:] = q[:,2,:]
        if op == 2:
            q[:,-1,:] = q[:,-3,:]
            q[:,-2,:] = q[:,-3,:]
            return q
        if op == 3:
            q[:,-2,:] = q[:,-3,:]
            q[:,-1,:] = q[:,-4,:]
            q[2,-1,:] = -q[1,-1,:] 
            q[2,-2,:] = -q[1,-2,:]
            return q

Ibc = 3 #tipo de condicion de frontera a la izquierda
Dbc = 2 #tipo de condicion de frontera a la derecha
Arbc = 2 #tipo de condicion de frontera a la derecha
Abbc = 2 #tipo de condicion de frontera a la derecha
"""enumeracion:
  1 condicion de frontera periodica
  2 condicion de frontera absorvente (zero-extrapolacion)
  3 condicion de rontera de pared solida (reflectivas)
"""
Lim = 3 #Limitador que sera usado para la reconstruccion
""" enumeracion
    0 primera orden (reconstruccion constante por partes)
    1 Limitador de fluxo MinMod
    2 Limitador de fluxo SuperBee
    3 Limitador de fluxo MC
"""
#-----------------------------------------------------------------------------
#Parametros Iniciales---------------------------------------------------------
#-----------------------------------------------------------------------------
rho_a = 1; rho_b = 2 #Densidad
k_a = 1; k_b = 0.5  #Bulk modulus
Lim_Infx = -1; Lim_Supx = 1; NumVolx = 400
Lim_Infy = -1; Lim_Supy = 1; NumVoly = 400
t_inicial = 0; t_final = 0.6
CFL = 0.5 #Condicion CFL para el calculo del paso de tiempo
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
Xc, Yc, Xn, Yn, dx, dy = Dominio(Lim_Infx, Lim_Supx, Lim_Infy, Lim_Supy, NumVolx, NumVoly) #Generar dominio computacional
Rho, K = Parametros_Dominio(Xc, Yc, rho_a, rho_b, k_a, k_b, NumVolx, NumVoly)
c = velocidades(K, Rho) #Velocidades, son dadas al resolver el problema de riemann
VelMax = np.max(c) #Velocidad maxima para el calculo del CFL
dt = (dx * CFL)/VelMax #Tamaño del paso del tiempo
q = Condicion_Inicial(Xc, Yc, Xn, Yn, NumVolx, NumVoly) #Calculo de la condicion inicial
q = BCx(q, Ibc, 0)
q = BCx(q, Dbc, 1)
q = BCy(q, Arbc, 0)
q = BCy(q, Abbc, 1)
qb = np.copy(q) #Copia de q, para evolucion temporal
qb1 = np.copy(q)
qini = np.copy(q)
Nt = int(round(t_final/dt))
for i in range(Nt):
    t_inicial += dt
    qb1 = q[:,2:-2,2:-2] + dt*(CUp(q, c, Lim))
    qb[:,2:-2,2:-2] = qb1
    qb1 = 3./4. * q[:,2:-2,2:-2] + (1. - 3./4.)*(qb1 + dt*(CUp(qb, c, Lim)))
    qb[:,2:-2,2:-2] = qb1
    qb1 = 1./3. * q[:,2:-2,2:-2] + (1. - 1./3.)*(qb1 + dt*(CUp(qb, c, Lim)))
    qb[:,2:-2,2:-2] = qb1
    qb = BCx(qb, Ibc, 0)
    qb = BCx(qb, Dbc, 1)
    qb = BCy(qb, Arbc, 0)
    qb = BCy(qb, Abbc, 1)
    q = np.copy(qb)

StressCUp = (q[0]*K)[2:-2,2:-2]
(vx, vy) = np.gradient(StressCUp, dx, dx)
vs = np.sqrt(vx**2 + vy**2)
fig, ax = plt.subplots()
# cnt = ax.contour(X[2:-2,2:-2], Y[2:-2,2:-2], StressCUp, np.arange(-0.5,1.03,0.03), cmap = cm.schlieren_grays)
cnt = ax.pcolormesh(Xn, Yn, vs, cmap = cm.yellow_red_blue)

# ax.plot([.0, .0],[-1, 0],'k', linewidth = 2)
# ax.plot([.0, 1],[0, 1],'k', linewidth = 2)