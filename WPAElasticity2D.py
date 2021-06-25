# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:09:35 2021

@author: JuanBarrios
"""

import numpy as np
import matplotlib.pyplot as plt
from visclaw import colormaps as cm

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
    x0 = -0.5; y0 = 0.
    width = 0.1; rad = 0.25
    r = np.sqrt((Xc - x0)**2 + (Yc - y0)**2)
    q[0] = (np.abs(r - rad)<=width)* (1. + np.cos(np.pi*(r-rad)/width))
    #-------------------------------------------------------------------------
    #teste3-------------------------------------------------------------------
    # r = np.sqrt(Xc**2 + Yc**2)
    # q[0] = (np.abs(r-0.5)<=0.2)*(1.+np.cos(np.pi*(r-0.5)/0.2))
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
    rho = rho_a * (Xc<0) + rho_b * (Xc>=0)
    k = k_a * (Xc<0) + k_b * (Xc>=0)
    #Medio Homogeneo----------------------------------------------------------
    # rho = rho_a * (Xc<0) + rho_a * (Xc>=0)
    # k = k_a * (Xc<0) + k_a * (Xc>=0)
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

def Riemann_Elasticity_2Dx(q, rho, k, c):
    r11 = 1/np.sqrt(k*rho)[:,:-1]
    r13 = -1/np.sqrt(k*rho)[:,1:]
    dfq1 = -((q[1]/rho)[:,1:] - (q[1]/rho)[:,:-1])
    dfq2 = -((q[0]*k)[:,1:] - (q[0]*k)[:,:-1])
    betha1 = (dfq1 - r13*dfq2)/(r11 - r13)
    betha3 = (-dfq1 + r11*dfq2)/(r11 - r13)
    dim = np.shape(q)
    dim = (dim[0], dim[1],dim[2]-1)
    W1 = np.zeros(dim)
    W3 = np.zeros(dim)
    W1[0] = betha1*r11
    W1[1] = betha1*1.
    W3[0] = betha3*r13
    W3[1] = betha3*1.
    Am = np.zeros(dim)
    Ap = np.zeros(dim)
    Am[:] = W1[:]
    Ap[:] = W3[:]
    return W1, W3, Am, Ap


def Riemann_Elasticity_2Dy(q, rho, k, c):
    r11 = 1/np.sqrt(k*rho)[:-1,:]
    r13 = -1/np.sqrt(k*rho)[1:,:]
    dgq1 = -((q[2]/rho)[1:,:] - (q[2]/rho)[:-1,:])
    dgq3= -((q[0]*k)[1:,:] - (q[0]*k)[:-1,:])
    betha1 = (dgq1 - r13*dgq3)/(r11 - r13)
    betha3 = (-dgq1 + r11*dgq3)/(r11 - r13)
    dim = np.shape(q)
    dim = (dim[0], dim[1]-1,dim[2])
    W1 = np.zeros(dim)
    W3 = np.zeros(dim)
    W1[0] = betha1*r11
    W1[2] = betha1*1.
    W3[0] = betha3*r13
    W3[2] = betha3*1.
    Bm = np.zeros(dim)
    Bp = np.zeros(dim)
    Bm[:] = W1[:]
    Bp[:] = W3[:]
    return W1, W3, Bm, Bp

def Transvese_Riemann_Elasticity_2Dx(q, rho, k, Ap, Am, c):
    r11 = (1/np.sqrt(k*rho))[:-2,1:]
    r13 = (-1/np.sqrt(k*rho))[1:-1:,1:]
    gamma1 = (Ap[0,1:-1,:] - r13*Ap[2,1:-1,:])/(r11 - r13)
    dim = np.shape(r11)
    BmAp = np.zeros((3,dim[0],dim[1]))
    BmAp[0] = -c[:-2,1:]*gamma1*r11
    BmAp[1] = -c[:-2,1:]*gamma1*0.
    BmAp[2] = -c[:-2,1:]*gamma1*1.
    
    r11 = (1/np.sqrt(k*rho))[1:-1,1:]
    r13 = (-1/np.sqrt(k*rho))[2:,1:]
    gamma3 = (-Ap[0,1:-1,:] + r11*Ap[2,1:-1,:])/(r11 - r13)
    dim = np.shape(r11)
    BpAp = np.zeros((3,dim[0],dim[1]))
    BpAp[0] = c[2:,1:]*gamma3*r13
    BpAp[1] = c[2:,1:]*gamma3*0.
    BpAp[2] = c[2:,1:]*gamma3*1.
    
    r11 = (1/np.sqrt(k*rho))[:-2,:-1]
    r13 = (-1/np.sqrt(k*rho))[1:-1,:-1]
    gamma1 = (Am[0,1:-1,:] - r13*Am[2,1:-1,:])/(r11 - r13)
    dim = np.shape(r11)
    BmAm = np.zeros((3,dim[0],dim[1]))
    BmAm[0] = -c[:-2,:-1]*gamma1*r11
    BmAm[1] = -c[:-2,:-1]*gamma1*0.
    BmAm[2] = -c[:-2,:-1]*gamma1*1.
    
    r11 = (1/np.sqrt(k*rho))[1:-1,:-1]
    r13 = (-1/np.sqrt(k*rho))[2:,:-1]
    gamma3 = (-Am[0,1:-1,:] + r11*Am[2,1:-1,:])/(r11 - r13)
    dim = np.shape(r11)
    BpAm = np.zeros((3,dim[0],dim[1]))
    BpAm[0] = c[2:,:-1]*gamma3*r13
    BpAm[1] = c[2:,:-1]*gamma3*0.
    BpAm[2] = c[2:,:-1]*gamma3*1.
    return BpAp, BmAp, BpAm, BmAm


def Transvese_Riemann_Elasticity_2Dy(q, rho, k, Bp, Bm, c):
    r11 = (1/np.sqrt(k*rho))[1:,:-2]
    r13 = (-1/np.sqrt(k*rho))[1:,1:-1]
    gamma1 = (Bp[0,:,1:-1] - r13*Bp[1,:,1:-1])/(r11 - r13)
    dim = np.shape(r11)
    AmBp = np.zeros((3,dim[0],dim[1]))
    AmBp[0] = -c[1:,:-2]*gamma1*r11
    AmBp[1] = -c[1:,:-2]*gamma1*0.
    AmBp[2] = -c[1:,:-2]*gamma1*1.
    
    r11 = (1/np.sqrt(k*rho))[1:,1:-1]
    r13 = (-1/np.sqrt(k*rho))[1:,2:]
    gamma3 = (-Bp[0,:,1:-1] + r11*Bp[1,:,1:-1])/(r11 - r13)
    dim = np.shape(r11)
    ApBp = np.zeros((3,dim[0],dim[1]))
    ApBp[0] = c[1:,2:]*gamma3*r13
    ApBp[1] = c[1:,2:]*gamma3*0.
    ApBp[2] = c[1:,2:]*gamma3*1.
    
    r11 = (1/np.sqrt(k*rho))[:-1,:-2]
    r13 = (-1/np.sqrt(k*rho))[:-1,1:-1]
    gamma1 = (Bm[0,:,1:-1] - r13*Bm[1,:,1:-1])/(r11 - r13)
    dim = np.shape(r11)
    AmBm = np.zeros((3,dim[0],dim[1]))
    AmBm[0] = -c[:-1,:-2]*gamma1*r11
    AmBm[1] = -c[:-1,:-2]*gamma1*0.
    AmBm[2] = -c[:-1,:-2]*gamma1*1.
    
    r11 = (1/np.sqrt(k*rho))[:-1,1:-1]
    r13 = (-1/np.sqrt(k*rho))[:-1,2:]
    gamma3 = (-Bm[0,:,1:-1] + r11*Bm[1,:,1:-1])/(r11 - r13)
    dim = np.shape(r11)
    ApBm = np.zeros((3,dim[0],dim[1]))
    ApBm[0] = c[:-1,2:]*gamma3*r13
    ApBm[1] = c[:-1,2:]*gamma3*0.
    ApBm[2] = c[:-1,2:]*gamma3*1.
    return ApBp, AmBp, ApBm, AmBm

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

def Fimm1D(s, W1, W2, L):
    F = np.zeros(np.shape(W1))
    norm = np.sum(W1[:,1:-1]**2,0)
    norm += (norm == 0. ) * 1.
    theta = np.sum(W1[:,2:]*W1[:,1:-1],0)/norm
    W1L = Limitador1D(theta, L)*W1[:,1:-1]
    norm = np.sum(W2[:,1:-1]**2,0)
    norm += (norm == 0 ) * 1.
    theta = np.sum(W2[:,0:-2]*W2[:,1:-1],0)/norm
    W2L = Limitador1D(theta, L)*W2[:,1:-1]
    F = ((np.sign(-s[1:-2])*(1.-dt/dx*np.abs(s[1:-2]))*W1L) +
        (np.sign(s[2:-1])*(1.-dt/dx*np.abs(s[2:-1]))*W2L))
    F = F * 0.5
    # print(np.shape(F))
    return F

def Fimm2D(s, W1, W2, L, axe):
    dim = np.shape(W1)
    if axe == 1:
        F = np.zeros((dim[0], dim[1],dim[2] - 2))
        for j in range(np.shape(s)[1]):
            F[:,j,:] = Fimm1D(s[:,j], W1[:,j,:], W2[:,j,:], L)
    elif axe == 2:
        F = np.zeros((dim[0], dim[1] - 2,dim[2]))
        for j in range(np.shape(s)[1]):
            F[:,:,j] = Fimm1D(s[j,:], W1[:,:,j], W2[:,:,j], L)
    return F

def Limitador1D(theta, o):
    shape = np.shape(theta)
    if o == 0:
        phy = 0.
        return phy
    if o == 1:
        phy = 1.
        return phy
    if o == 2:
        theta1 = np.array([np.ones(shape), theta])
        phy = MinMod(theta1)
        return phy
    if o == 3:
        a = np.zeros(shape)
        b = np.ones(shape)
        c = np.min([b , 2.*theta],0)
        d = np.min([2.*b, theta],0)
        phy = np.max([a,c,d],0)
        return phy
    if o == 4:
        a = np.zeros(shape)
        b = np.ones(shape)
        c = np.min([(b + theta)/2. , 2.*theta, 2*b],0)
        phy = np.max([a,c],0)
        return phy

def MaxMod(a):
    k1 = a > 0.
    k2 = a < 0.
    return (k1.all(0))*np.max(a,0) + (k2.all(0))*np.min(a,0)

def MinMod(a):
    k1 = a > 0.
    k2 = a < 0.
    return (k1.all(0))*np.min(a,0) + (k2.all(0))*np.max(a,0)

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
Lim = 4 #Limitador que sera usado para la reconstruccion
""" enumeracion
    0 primera orden (hace el termino de correccion F = 0)
    1 Metodo Lax-Wendroff
    2 Limitador de fluxo MinMod
    3 Limitador de fluxo SuperBee
    4 Limitador de fluxo MC-theta = 2
"""
#-----------------------------------------------------------------------------
#Parametros Iniciales---------------------------------------------------------
#-----------------------------------------------------------------------------
rho_a = 1; rho_b = 2 #Densidad
k_a = 1; k_b = 0.5  #Bulk modulus
Lim_Infx = -1; Lim_Supx = 1; NumVolx = 200
Lim_Infy = -1; Lim_Supy = 1; NumVoly = 200
t_inicial = 0; t_final = 0.6
CFL = 0.5 #Condicion CFL para el calculo del paso de tiempo
Dimensional_Splitting = False; DS_ordem = 2
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
qini = np.copy(q)
Nt = int(round(t_final/dt))
if Dimensional_Splitting:
    if DS_ordem == 1:
        for i in range(Nt):
            t_inicial += dt
            F = np.zeros((3,NumVoly+4,NumVolx+3))
            G = np.zeros((3,NumVoly+3,NumVolx+4))
            W1, W3, Am, Ap = Riemann_Elasticity_2Dx(q, Rho, K, c)
            qb[:,:,2:-2] = qb[:,:,2:-2] - dt/dx * (Ap[:,:,1:-2] + Am[:,:,2:-1])
            F[:,:,1:-1] = Fimm2D(c, W1, W3, Lim, 1)
            qb[:,:,2:-2] = qb[:,:,2:-2] - dt/dx * (F[:,:,2:-1] - F[:,:,1:-2])
            qb = BCx(qb, Ibc, 0)
            qb = BCx(qb, Dbc, 1)
            W1, W3, Bm, Bp = Riemann_Elasticity_2Dy(qb, Rho, K, c)
            qb[:,2:-2,:] = qb[:,2:-2,:] - dt/dy * (Bp[:,1:-2,:] + Bm[:,2:-1,:])
            G[:,1:-1,:] = Fimm2D(c, W1, W3, Lim, 2)
            qb[:,2:-2,:] = qb[:,2:-2,:] - dt/dy * (G[:,2:-1,:] - G[:,1:-2,:])
            qb = BCy(qb, Arbc, 0)
            qb = BCy(qb, Abbc, 1)
            q = np.copy(qb)
    else:
        for i in range(Nt):
            t_inicial += dt
            F = np.zeros((3,NumVoly+4,NumVolx+3))
            G = np.zeros((3,NumVoly+3,NumVolx+4))
            W1, W3, Am, Ap = Riemann_Elasticity_2Dx(q, Rho, K, c)
            qb[:,:,2:-2] = qb[:,:,2:-2] - dt/(2*dx) * (Ap[:,:,1:-2] + Am[:,:,2:-1])
            F[:,:,1:-1] = Fimm2D(c, W1, W3, Lim, 1)
            qb[:,:,2:-2] = qb[:,:,2:-2] - dt/dx * (F[:,:,2:-1] - F[:,:,1:-2])
            qb = BCx(qb, Ibc, 0)
            qb = BCx(qb, Dbc, 1)
            W1, W3, Bm, Bp = Riemann_Elasticity_2Dy(qb, Rho, K, c)
            qb[:,2:-2,:] = qb[:,2:-2,:] - dt/dy * (Bp[:,1:-2,:] + Bm[:,2:-1,:])
            G[:,1:-1,:] = Fimm2D(c, W1, W3, Lim, 2)
            qb[:,2:-2,:] = qb[:,2:-2,:] - dt/dy * (G[:,2:-1,:] - G[:,1:-2,:])
            qb = BCy(qb, Arbc, 0)
            qb = BCy(qb, Abbc, 1)
            W1, W3, Am, Ap = Riemann_Elasticity_2Dx(qb, Rho, K, c)
            qb[:,:,2:-2] = qb[:,:,2:-2] - dt/(2*dx) * (Ap[:,:,1:-2] + Am[:,:,2:-1])
            F[:,:,1:-1] = Fimm2D(c, W1, W3, Lim, 1)
            qb[:,:,2:-2] = qb[:,:,2:-2] - dt/dx * (F[:,:,2:-1] - F[:,:,1:-2])
            qb = BCx(qb, Ibc, 0)
            qb = BCx(qb, Dbc, 1)
            q = np.copy(qb)
else:
    for i in range(Nt):
        t_inicial += dt
        F = np.zeros((3,NumVoly+4,NumVolx+3))
        G = np.zeros((3,NumVoly+3,NumVolx+4))
        W1x, W3x, Am, Ap = Riemann_Elasticity_2Dx(q, Rho, K, c)
        F[:,:,1:-1] = F[:,:,1:-1] + Fimm2D(c, W1x, W3x, Lim, 1)
        BpAp, BmAp, BpAm, BmAm = Transvese_Riemann_Elasticity_2Dx(q, Rho, K, Ap, Am, c)
        G[:,:-1,1:] = G[:,:-1,1:] - dt/(2.*dx)*BmAp
        G[:,1:,1:] = G[:,1:,1:] - dt/(2.*dx)*BpAp
        G[:,:-1,:-1] = G[:,:-1,:-1] - dt/(2.*dx)*BmAm
        G[:,1:,:-1] = G[:,1:,:-1] - dt/(2.*dx)*BpAm
        
        W1y, W3y, Bm, Bp = Riemann_Elasticity_2Dy(q, Rho, K, c)
        G[:,1:-1,:] = G[:,1:-1,:] + Fimm2D(c, W1y, W3y, Lim, 2)
        ApBp, AmBp, ApBm, AmBm = Transvese_Riemann_Elasticity_2Dy(q, Rho, K, Bp, Bm, c)
        F[:,1:,:-1] = F[:,1:,:-1] - dt/(2.*dy)*AmBp
        F[:,1:,1:] = F[:,1:,1:] - dt/(2.*dy)*ApBp
        F[:,:-1,:-1] = F[:,:-1,:-1] - dt/(2.*dy)*AmBm
        F[:,:-1,1:] = F[:,:-1,1:] - dt/(2.*dy)*ApBm
        
        
        qb[:,2:-2:,2:-2] = qb[:,2:-2,2:-2] - dt/dx * (Ap[:,2:-2,1:-2] + Am[:,2:-2,2:-1])\
            - dt/dy * (Bp[:,1:-2,2:-2] + Bm[:,2:-1,2:-2])\
                - dt/dx * (F[:,2:-2,2:-1] - F[:,2:-2,1:-2])\
                    - dt/dy * (G[:,2:-1,2:-2] - G[:,1:-2,2:-2])\
                        
                        
                    
        qb = BCx(qb, Ibc, 0)
        qb = BCx(qb, Dbc, 1)
        qb = BCy(qb, Arbc, 0)
        qb = BCy(qb, Abbc, 1)
        #-------------------------------------------------------------------------
        
        q = np.copy(qb)

StressWPA = (q[0]*K)[2:-2,2:-2]
(vx, vy) = np.gradient(StressWPA, dx, dy)
vs = np.sqrt(vx**2 + vy**2)
fig, ax = plt.subplots()
cnt = ax.pcolormesh(Xn, Yn, StressWPA, cmap = cm.yellow_red_blue)
# cnt = ax.contour(X[2:-2,2:-2], Y[2:-2,2:-2], sig11, np.arange(-0.5,1.03,0.03), cmap = cm.schlieren_grays)
# cnt = ax.pcolormesh(Xn, Yn, vs, cmap = cm.schlieren_grays)
# cnt = ax.contour(X[2:-2,2:-2], Y[2:-2,2:-2], vs, cmap = cm.schlieren_grays)
# cnt = ax.pcolor(X[2:-2,2:-2], Y[2:-2,2:-2], vs, cmap = 'binary')
# ax.axvline(x=0, color = 'black')
# fig.colorbar(cnt)
# ax.plot([.0, .0],[-1, 0],'k', linewidth = 2)
# ax.plot([.0, 1],[0, 1],'k', linewidth = 2)