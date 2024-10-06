import numpy as np

def pot_doublewell(x, f=0.0367493, a0=0.0, a1=0.429, a2=-1.126, a3=-0.143, a4=0.563):
    # A-T pair double-well potential in Hartrees (x is in Bohr)
    xi = x/1.9592
    return f*(a0 + a1*xi + a2*xi**2 + a3*xi**3 + a4*xi**4)

#some parameter to convert the units
au2fs   = 2.418884254E-2
au2cm   = 219474.63068
au2joule = 4.35974381E-18
bolz   = 1.3806503E-23
au2ev   = 27.2114
hbar = 1.0

mass0 = 1836.15
beta = au2joule/(bolz*300)
omega = 0.00436 #the frequency associate with the right well
kappa = 1/(10/au2fs)
nth = 1/(np.exp(beta*omega)-1) #0.010264683592287289

#set up the grid point
xmin = -4.0
xmax = 4.0
ndvr = 1024
xgrid = np.linspace(xmin,xmax,ndvr)

pot_arr = pot_doublewell(xgrid)

import scipy.linalg as LA
#output the eigen state for potential in x-space
#input:kinetic energy hamiltonian, potential in x-space
def eig_state(hamk,pot,xgrid,Nstate):

  Mata = hamk.copy()
  for i in range(ndvr):
    Mata[i,i]+=pot[i]

  val,arr = LA.eigh(Mata)
  dx = xgrid[1]-xgrid[0]
  return val[:Nstate],arr[:,:Nstate]/dx**0.5

import scipy.fft as sfft
kgrid = np.zeros(ndvr,dtype=np.float64)
#ak2: kinetic energy array in k-space
ak2   = np.zeros(ndvr,dtype=np.float64)

dx = xgrid[1]-xgrid[0]
dk = 2.0*np.pi/((ndvr)*dx)
coef_k = hbar**2/(2.0*mass0)

for i in range(ndvr):
  if(i<ndvr//2):
    kgrid[i] = i*dk
  else:
    kgrid[i] = -(ndvr-i) * dk

  ak2[i] = coef_k*kgrid[i]**2

akx0 = sfft.ifft(ak2)
#hamk: kinetic hamiltonian Matrix in position x grid space
hamk = np.zeros((ndvr,ndvr),dtype=np.complex128)

for i in range(ndvr):
  for j in range(ndvr):
    if(i<j):
      hamk[i,j] = akx0[i-j].conj()
    else:
      hamk[i,j] = akx0[i-j]

Neig = 50
eneg_DW,psi_DW = eig_state(hamk,pot_arr,xgrid,Neig)

#the eigenstate in the k-space representation
#(by Fourier transform of the original eigenstate in x-space)
psik_DW = np.zeros((ndvr,Neig),dtype=np.complex_)
pre_fac = dx/(2*np.pi)**0.5
for i in range(Neig):
  psik_DW[:,i] = sfft.fft(psi_DW[:,i])*pre_fac

#initial density matrix
ini_occu = np.zeros(Neig,dtype=np.complex_)
ini_occu[5] = 1.0
rho0 = np.outer(ini_occu,ini_occu.conj())

#The operator in the eigenstate
xmat_eig = np.zeros((Neig,Neig),dtype=np.complex_)
pmat_eig = np.zeros((Neig,Neig),dtype=np.complex_)
for i in range(Neig):
  for j in range(Neig):
    xmat_eig[i,j] = np.dot(np.multiply(psi_DW[:,i].conj(),xgrid),psi_DW[:,j])*dx
    pmat_eig[i,j] = np.dot(np.multiply(psik_DW[:,i].conj(),kgrid),psik_DW[:,j])*dk

#hamiltonian
H_dw = np.diag(eneg_DW)
#creation/annihilation operator
amat_eig = xmat_eig.copy()*np.sqrt(mass0*omega/2)+1j*pmat_eig.copy()/np.sqrt(mass0*omega*2)
adegmat_eig = xmat_eig.copy()*np.sqrt(mass0*omega/2)-1j*pmat_eig.copy()/np.sqrt(mass0*omega*2)

#define the population on the left/right well
x_barrier = 0.37321768
P_R = np.heaviside(xgrid-x_barrier,1)
P_L = 1 - np.heaviside(xgrid-x_barrier,1)

P_R_eig = np.zeros((Neig,Neig),dtype=np.complex_)
P_L_eig = np.zeros((Neig,Neig),dtype=np.complex_)
for i in range(Neig):
  for j in range(Neig):
    P_R_eig[i,j] = np.dot(np.multiply(psi_DW[:,i].conj(),P_R),psi_DW[:,j])*dx
    P_L_eig[i,j] = np.dot(np.multiply(psi_DW[:,i].conj(),P_L),psi_DW[:,j])*dx