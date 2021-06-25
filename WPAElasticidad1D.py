# -*- coding: utf-8 -*-
"""
@author: JuanBarrios
"""
import numpy as np
import matplotlib.pyplot as plt

def Dominio(LimInf, LimSup, NumVol):
    #Funcion para generar Dominio computacional
    #LimInf es el limite inferior
    #LimSup es el limite superior
    #NumVol es el numero de volumenes (celdas)
    dx = (LimSup - LimInf)/NumVol
    # xn = np.linspace(LimInf-2.*dx,LimSup+2.*dx,NumVol + 5)
    xn = np.zeros(NumVol + 5)
    xn[2:-2] = np.linspace(LimInf,LimSup,NumVol + 1)
    # xc = (xn[:-1] + xn[1:])/2
    # xc = xn[:-1]
    xn[0] = xn[2] - 2*dx
    xn[1] = xn[2] - dx
    xn[-1] = xn[-3] + 2*dx
    xn[-2] = xn[-3] + dx
    xc = xn[:-1]
    return xc, xn, dx

def Condicion_Inicial(xc, NumVol):
    q = np.zeros((2,NumVol + 4))
    # q[0,:] = np.sin(np.pi * xc)
    # for i in range(NumVol + 4):
    #     if 0.4 < xc[i] < 0.6:
    #         q[0,i] = (7/4. - 3/4*np.cos(10*np.pi*xc[i] - 4*np.pi))/K[i]
    #     else:
    #         q[0,i] = 1/K[i]
    return q

def velocidades(k, rho):
    c = np.sqrt(k/rho)
    return c

def k(xc, dl, NumVol):
    ka = 1. #módulo de volumen (bulk modulus) 1
    kb = 3. #módulo de volumen (bulk modulus) 2
    
    #Layered Medium-----------------------------------------------------------
    k = np.zeros(NumVol+4)
    a=2
    b=2+dl
    while a<=NumVol-dl:
        k[a:b]=ka
        if b<NumVol:
            k[a+dl:b+dl] = kb
        a+=2*dl
        b+=2*dl
    k[:2] = k[2]
    k[-2:] = k[-3]
    #-------------------------------------------------------------------------
    
    #Medio Heterogeneo com unica discontinuidade em x = 150
    # k = ka*(xc<=150.) + kb*(xc>150.)
    #-------------------------------------------------------------------------
    
    #Medio Homogeneo
    # k = np.ones(NumVol + 4)*ka
    # -------------------------------------------------------------------------
    return k


def rho(xc, dl, NumVol):
    rhoa = 1. #Densidad 1
    rhob = 3. #Densidad 2
    
    #Layered Medium-----------------------------------------------------------
    rho = np.zeros(NumVol + 4)
    a=2
    b=2+dl
    while a<=NumVol - dl:
        rho[a:b]=rhoa
        if b<NumVol:
            rho[a+dl:b+dl] = rhob
        a+=2*dl
        b+=2*dl
    rho[:2] = rho[2]
    rho[-2:] = rho[-3]
    #-------------------------------------------------------------------------
    
    #Medio Heterogeneo com unica discontinuidade em x = 0.5
    # rho = rhoa*(xc<=0.5) + rhob*(xc>0.5)
    #-------------------------------------------------------------------------
    
    #Medio Homogeneo
    # rho = np.ones(NumVol + 4)*rhoa
    #-------------------------------------------------------------------------
    return rho

def RiemmanElasticity(q, rho, k, c):
    Stress = k*q[0,:]
    z = c*rho
    dfq1 = q[1,1:]/rho[1:] - q[1,:-1]/rho[:-1]
    dfq2 = Stress[1:] - Stress[:-1]
    betha1 = -(z[1:]*dfq1 + dfq2)/(z[1:] + z[:-1])
    betha2 = -(z[:-1]*dfq1 - dfq2)/(z[1:] + z[:-1])
    W1 = np.zeros((2,np.shape(betha1)[0]))
    W2 = np.copy(W1)
    W1[0] = betha1
    W1[1] = betha1*z[:-1]
    W2[0] = betha2
    W2[1] = -betha2*z[1:]
    Am = W1
    Ap = W2
    return W1, W2, Am, Ap

def Fimm(s, W1, W2, L):
    F = np.zeros(np.shape(W1))
    norm = np.sum(W1[:,1:-1]**2,0)
    norm += (norm == 0. ) * 1.
    theta = np.sum(W1[:,2:]*W1[:,1:-1],0)/norm
    W1L = Limitador(theta, L)*W1[:,1:-1]
    norm = np.sum(W2[:,1:-1]**2,0)
    norm += (norm == 0. ) * 1.
    theta = np.sum(W2[:,:-2]*W2[:,1:-1],0)/norm
    W2L = Limitador(theta, L)*W2[:,1:-1]
    F = ((np.sign(-s[1:-2])*(1.-dt/dx*np.abs(s[1:-2]))*W1L) +
        (np.sign(s[2:-1])*(1.-dt/dx*np.abs(s[2:-1]))*W2L))
    F = F * 0.5
    return F

def Limitador(theta, o):
    shape = np.shape(theta)
    if o == 0:
        #No usar limitador (reduce el metodo a primera orden)------------------
        phy = 0.
        return phy
    if o == 1:
        #Metodo Lax-Wendroff--------------------------------------------------
        phy = 1.
        return phy
    if o == 2:
        #Limitador de Fluxo MinMod--------------------------------------------
        theta1 = np.array([np.ones(shape), theta])
        phy = MinMod(theta1)
        return phy
    if o == 3:
        #Limitador de Fluxo SuperBee------------------------------------------
        a = np.zeros(shape)
        b = np.ones(shape)
        c = np.min([b , 2.*theta],0)
        d = np.min([2.*b, theta],0)
        phy = np.max([a,c,d],0)
        return phy
    if o == 4:
        #Limitador de Fluxo MC------------------------------------------------
        t = 2.
        a = np.zeros(shape)
        b = np.ones(shape)
        c = np.min([(b + theta)/2. , t*theta, t*b],0)
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

def BC(q, t, tipo, Lado):
    #Condicion de frontera izquierda------------------------------------------
    if Lado == 0:
        if tipo == 1:
        #Condicoes de fronteira periodicas
            q[:,0] = q[:,-4]
            q[:,1] = q[:,-3]
            return q
        if tipo == 2:
        #Condicoes de fronteira absorvente
            q[:,0] = q[:,2]
            q[:,1] = q[:,2]
            return q
        if tipo == 3:
        #Condicoes de fronteira de pared solida
            q[:,1] = q[:,2]
            q[:,0] = q[:,3]
            q[1,0] = -q[1,0]
            q[1,1] = q[1,1]
            return q
        if tipo == 4:
        #Condicion de frontera modificada Parede Oscilante
            if t<=60:
                w1 = (-q[0,2] + q[1,2])/2.
                u0 = -0.2*(1. + np.cos(np.pi*((t + dx/2)-30.)/30.))
                u1 = -0.2*(1. + np.cos(np.pi*((t + 3*dx/2.)-30.)/30.))
                q[:,1] = w1*np.array([1,1]) - u0*np.array([1,-1])
                q[:,0] = w1*np.array([1,1]) - u1*np.array([1,-1])
                q[1,2] = Rho[0]*(-0.2*(1. + np.cos(np.pi*((t-30.)/30.))))
            else:
                q[1,2] = 0.
            return q
    if Lado == 1:
            if tipo == 1:
            #Condicoes de fronteira periodicas
                q[:,-1] = q[:,3]
                q[:,-2] = q[:,2]
                return q
            if tipo == 2:
            #Condicoes de fronteira absorvente
                q[:,-1] = q[:,-3]
                q[:,-2] = q[:,-3]
                return q
            if tipo == 3:
            #Condicoes de fronteira de pared solida
                q[:,-2] = q[:,-3]
                q[:,-1] = q[:,-4]
                q[1,-1] = -q[1,-1]
                q[1,-2] = -q[1,-2]
                return q

Ibc = 4 #tipo de condicion de frontera a la izquierda
Dbc = 2 #tipo de condicion de frontera a la derecha
"""enumeracion:
  1 condicion de frontera periodica
  2 condicion de frontera absorvente (zero-extrapolacion)
  3 condicion de rontera de pared solida (reflectivas)
"""
Lim = 4 #Limitador que sera usado para limitar las ondas 
""" enumeracion
    0 primera orden (hace el termino de correccion F = 0)
    1 reduce el metodo al metodo de lax-wendroff
    2 Limitador de fluxo MinMod
    3 Limitador de fluxo SuperBee
    4 Limitador de fluxo MC
"""
#-----------------------------------------------------------------------------
#Parametros Iniciales---------------------------------------------------------
#-----------------------------------------------------------------------------
rho_a = 1; rho_b = 3 #Densidad
k_a = 1; k_b = 3  #Bulk modulus
Lim_Inf = 0; Lim_Sup = 300; NumVol = 300*2**8; NumLayers = 300
t_inicial = 0; t_final = 240
CFL = 0.5 #Condicion CFL para el calculo del paso de tiempo
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
dl = NumVol//NumLayers #tamaño de cada camada
xc, xn, dx = Dominio(Lim_Inf, Lim_Sup, NumVol) #Generar dominio computacional
Rho = rho(xc, dl, NumVol) #Densidad, Parametro del medio
K = k(xc, dl, NumVol) #Bulk Modulus, Parametro del medio
c = velocidades(K, Rho) #Velocidades, son dadas al resolver el problema de riemann
#Test 1 ---------------------------------------------------------------------
# c = 1. + 0.5*np.sin(10.*np.pi*xc); Z = np.ones(np.shape(c))
#-----------------------------------------------------------------------------
#Test 2 ----------------------------------------------------------------------
# c = 1. + 0.5*np.sin(10.*np.pi*xc); Z = 1 + 0.25*np.cos(10.*np.pi*xc)
# -----------------------------------------------------------------------------
#Teste 3 ---------------------------------------------------------------------
# a = xc>0.35
# b = xc<0.65
# c = np.where(np.array([a,b]).all(0), 2., 0.6); Z = np.where(np.array([a,b]).all(0), 2., 6.)
#-----------------------------------------------------------------------------
# Rho = Z/c
# K = c * Z
#-----------------------------------------------------------------------------
VelMax = np.max(c) #Velocidad maxima para el calculo del CFL
dt = (dx * CFL)/VelMax #Tamaño del paso del tiempo
q = Condicion_Inicial(xc, NumVol) #Calculo de la condicion inicial
q = BC(q, t_inicial, Ibc, 0) #aplicamos las condiciones de frontera al lado izquierdo
q = BC(q, t_inicial, Dbc, 1) #Aplicamos las condiciones de frontera al lado derecho
qb = np.copy(q) #Copia de q, para evolucion temporal
# qini = np.copy(q)
# plt.plot(xc[2:-2], qini[0,2:-2]*K[2:-2], 'r--')
# plt.plot(xc[2:-2], c[2:-2], 'r--')
# plt.plot(xc[2:-2], Z[2:-2], 'b--')
Nt = int(round(t_final/dt))
for i in range(Nt):
    t_inicial += dt #incrementamos el tiempo
    W1, W2, Am, Ap = RiemmanElasticity(q, Rho, K, c) #solucionamos el problema de Riemann
    #aplicamos el metodo 
    qb[:,2:-2] = q[:,2:-2] - dt/dx * (Ap[:,1:-2] + Am[:,2:-1])
    F = Fimm(c, W1, W2, Lim)
    qb[:,2:-2] = qb[:,2:-2] - dt/dx * (F[:,1:] - F[:,:-1])
    qb = BC(qb, t_inicial, Ibc, 0)
    qb = BC(qb, t_inicial, Dbc, 1)
    q = np.copy(qb)

Stress = (K * q[0,:])[2:-2] #Calculamos la tension \sigma = k * \epsilon
# print(np.sum(np.abs(Stress))/NumVol)
plt.plot(xc[2:-2], Stress) #Graficamos la solucion \sigma
# Erro = np.abs(1.1367898328013288 - np.sum(np.abs(Stress))/NumVol)
# Erro = np.sum(np.abs(qini[0,2:-2] - Stress))/NumVol
# print('N = ', NumVol)
# print('L1 = ',format(Erro, '.3E'))
# Erro1 = np.abs(1.750000908615183 - np.max(Stress))
# print('L-inf = ',format(Erro1, '.3E'))