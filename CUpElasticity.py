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
    #Funcion en la que se imponen las condiciones iniciales
    q = np.zeros((2,NumVol + 4))
    #Test solucion exacta conocida
    # q[0,:] = np.sin(np.pi * xc)
    #Test1 - 2 - 3   
    # for i in range(NumVol + 4):
    #     if 0.4 < xc[i] < 0.6:
    #         q[0,i] = (7/4. - 3/4*np.cos(10*np.pi*xc[i] - 4*np.pi))/K[i]
    #     else:
    #         q[0,i] = 1./K[i]
    return q

def velocidades(k, rho):
    #Velocidades (autovalores de la matriz jacobiana del flujo)
    c = np.sqrt(k/rho)
    return c

def k(xc, dl, NumVol, k_a, k_b):
    #Layered Medium-----------------------------------------------------------
    k = np.zeros(NumVol+4)
    a=2
    b=2+dl
    while a<=NumVol-dl:
        k[a:b]=k_a
        if b<=NumVol:
            k[a+dl:b+dl] = k_b
        a+=2*dl
        b+=2*dl
    k[:2] = k[2]
    k[-2:] = k[-3]
    #-------------------------------------------------------------------------
    
    #Medio Heterogeneo com unica discontinuidade em x = 150
    # k = k_a*(xc<=0.5) + k_b*(xc>0.5)
    #-------------------------------------------------------------------------
    
    #Medio Homogeneo
    # k = np.ones(NumVol + 4)*k_a
    #-------------------------------------------------------------------------
    
    #Test 1-------------------------------------------------------------------
    
    return k


def rho(xc, dl, NumVol, rho_a, rho_b):
    #Layered Medium-----------------------------------------------------------
    rho = np.zeros(NumVol + 4)
    a=2
    b=2+dl
    while a<=NumVol - dl:
        rho[a:b]=rho_a
        if b<=NumVol:
            rho[a+dl:b+dl] = rho_b
        a+=2*dl
        b+=2*dl
    rho[:2] = rho[2]
    rho[-2:] = rho[-3]
    #-------------------------------------------------------------------------
    
    #Medio Heterogeneo com unica discontinuidade em x = 0.5
    # rho = rho_a*(xc<=0.5) + rho_b*(xc>0.5)
    #-------------------------------------------------------------------------
    
    #Medio Homogeneo
    # rho = np.ones(NumVol + 4)*rho_a
    #-------------------------------------------------------------------------
    return rho

def Fluxo(q,i):
    #Funcion de flujo de las equaciones de elasticidad 1D
    f = np.copy(q)
    f[0,:] = -q[1,:]/Rho[i[0]:i[1]]
    f[1,:] = -q[0,:]*K[i[0]:i[1]]
    return f  
    
def Limitador(cp,cm,cpm,lim):
    if lim == 0:
        #Reconstruccion constante por partes----------------------------------
        return 0.
    if lim == 1:
        #Limitador MinMod-----------------------------------------------------
        return MinMod(np.array([cp/dx,cm/dx]))
    if lim == 2:
        #Limitador SuperBee
        sig1 = MinMod(np.array([2.*cm/dx,cp/dx]))
        sig2 = MinMod(np.array([cm/dx,2.*cp/dx]))
        return MaxMod(np.array([sig1,sig2]))
    if lim == 3:
        #Limitador MC---------------------------------------------------------
        return MinMod(np.array([2.*cm/dx,cpm/(2.*dx),2.*cp/dx]))

def MaxMod(a):
    k1 = a > 0.
    k2 = a < 0.
    return (k1.all(0))*np.max(a,0) + (k2.all(0))*np.min(a,0)
	
def MinMod(a):
    k1 = a > 0.
    k2 = a < 0.
    return (k1.all(0))*np.min(a,0) + (k2.all(0))*np.max(a,0)

def CUp(q, c):
    #Reconstruccion
    cp = q[:,2:] - q[:,1:-1] 
    cm = q[:,1:-1] - q[:,:-2]
    cpm = q[:,2:] - q[:,:-2]
    
    qp = q[:,2:-1] - dx/2. *  Limitador(cp[:,1:], cm[:,1:], cpm[:,1:], Lim) 
    qm = q[:,1:-2] + dx/2. *  Limitador(cp[:,:-1],cm[:,:-1],cpm[:,:-1], Lim)
    
    ap = np.maximum(c[1:-2], np.maximum(c[2:-1], 0.))
    am = np.minimum(-c[1:-2], np.minimum(-c[2:-1], 0.))
    
    apmam = ap - am
    
    Wint = ap*qp - am*qm - (Fluxo(qp,[2,-1]) - Fluxo(qm,[1,-2]))
    Wint /= apmam
    T1 = (qp - Wint)/apmam
    T2 = (Wint - qm)/apmam
    qchiu = MinMod(np.array([T1,T2]))
    # qchiu = 0.
    apxam = ap * am
    H = ap*Fluxo(qm,[1,-2]) - am*Fluxo(qp,[2,-1])
    H = H/apmam + (apxam * ((qp - qm)/apmam - qchiu))
    return  - (H[:,1:] - H[:,:-1])/dx

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
Lim = 3 #Limitador que sera usado para limitar las ondas 
""" enumeracion
    0 reconstruccion constante por partes
    1 Limitador de flujo MinMod
    2 Limitador de flujo SuperBee
    3 Limitador de flujo MC
"""
#-----------------------------------------------------------------------------
#Parametros Iniciales---------------------------------------------------------
#-----------------------------------------------------------------------------
rho_a = 1; rho_b = 3 #Densidad
k_a = 1; k_b = 3  #Bulk modulus
Lim_Inf = 0; Lim_Sup = 300; NumVol = 100; NumLayers = 300
t_inicial = 0; t_final = 80
CFL = 0.5 #Condicion CFL para el calculo del paso de tiempo
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
dl = NumVol//NumLayers #tamaño de cada camada
xc, xn, dx = Dominio(Lim_Inf, Lim_Sup, NumVol) #Generar dominio computacional
Rho = rho(xc, dl, NumVol, rho_a, rho_b) #Densidad, Parametro del medio
K = k(xc, dl, NumVol, k_a, k_b) #Bulk Modulus, Parametro del medio
c = velocidades(K, Rho) #Velocidades, son dadas al resolver el problema de riemann
# Test 1 ---------------------------------------------------------------------
# c = 1. + 0.5*np.sin(10.*np.pi*xc); Z = np.ones(np.shape(c))
#-----------------------------------------------------------------------------
#Test 2 ----------------------------------------------------------------------
# c = 1. + 0.5*np.sin(10.*np.pi*xc); Z = 1 + 0.25*np.cos(10.*np.pi*xc)
#-----------------------------------------------------------------------------
# Teste 3 ---------------------------------------------------------------------
# a = xc>0.35
# b = xc<0.65
# c = np.where(np.array([a,b]).all(0), 2., 0.6); Z = np.where(np.array([a,b]).all(0), 2., 6.)
# -----------------------------------------------------------------------------
# Rho = Z/c
# K = c*Z
VelMax = np.max(c) #Velocidad maxima para el calculo del CFL
dt = (dx * CFL)/VelMax #Tamaño del paso del tiempo
q = Condicion_Inicial(xc, NumVol) #Calculo de la condicion inicial
q = BC(q, t_inicial, Ibc, 0) #aplicamos las condiciones de frontera al lado izquierdo
q = BC(q, t_inicial, Dbc, 1) #Aplicamos las condiciones de frontera al lado derecho
qb = np.copy(q) #Copia de q, para evolucion temporal
qb1 = np.copy(q)
qini = np.copy(q)
# plt.plot(xc[2:-2], qini[0,2:-2]*K[2:-2], 'r--')
# plt.plot(xc[2:-2],c[2:-2])
# plt.plot(xc[2:-2],np.ones(np.shape(xc))[2:-2])
Nt = int(round(t_final/dt))
for i in range(Nt):
    t_inicial += dt
    qb = q[:,2:-2] + dt*CUp(q, c)
    qb1[:,2:-2] = qb
    qb = 3./4. * q[:,2:-2] + (1. - 3./4.)*(qb + dt*CUp(qb1, c))
    qb1[:,2:-2] = qb
    qb = 1./3. * q[:,2:-2] + (1. - 1./3.)*(qb + dt*CUp(qb1, c))
    qb1[:,2:-2] = qb
    qb1 = BC(qb1, t_inicial, Ibc, 0)
    qb1 = BC(qb1, t_inicial, Dbc, 1)
    q = np.copy(qb1)
     
StressCUp = (q[0]*K)[2:-2]
plt.plot(xc[2:-2], StressCUp)
# print(np.abs(1.1367898328370631 - np.sum(np.abs(StressCUp))/NumVol))
# plt.plot(xc[2:-2], qini[0,2:-2])
# Erro = np.abs(1.1367898328013288 - np.sum(np.abs(StressCUp))/NumVol)
# print('N = ', NumVol)
# print('L1 = ',format(Erro, '.3E'))
# Erro1 = np.sum(np.abs(qini[0,2:-2] - StressCUp))/NumVol
# Erro1 = np.abs(1.750000908615183 - np.max(np.abs(StressCUp)))
# print('L-inf = ',format(Erro1, '.3E'))