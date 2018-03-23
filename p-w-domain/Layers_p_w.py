#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 13:27:28 2017

@author: christianreini
"""
from Wavefield_p_w import Wavefield_p_w
import numpy as np
import matplotlib.pylab as plt


# Responses of layered media are wavefields
# Hence they inherit all properties of wavefields
class Layers_p_w(Wavefield_p_w):  
    """
    Layers
    
    Compute responses of layered media for multiple frequencies and a single ray-parameter.
    
    Variables:
        nt:      Number of time/frequency samples
        dt:      Duration per time sample in seconds
        nr:      Number of space/wavenumber samples
        dx:      Distance per space sample in metres
        nf:      Number of time samples divided by 2 plus 1.
        nk:      Number of space samples divided by 2 plus 1.
        verbose: Set verbose=1 to gt some feedback about processes.
        dzvec:   List or array with the thickness of each layer
        cpvec:   List or array with the P-wave veclocity in each layer
        csvec:   List or array with the S-wave veclocity in each layer
        rovec:   List or array with the density of each layer
        
    Data sorting: 
        nt (x nr) x 4
        
    Vectorised computation
    """
    def __init__(self,nt,dt,nr,dx,dzvec,cpvec,csvec,rovec):
        Wavefield_p_w.__init__(self,nt,dt,nr,dx)
        self.dzvec = dzvec
        self.cpvec = cpvec
        self.csvec = csvec
        self.rovec = rovec
        self.zvec  = np.cumsum(self.dzvec).tolist()
        
    def Plot_model(self):
        """
        Plot 1D Model.
        """
    
        zvec = np.cumsum(self.dzvec)
        Z    = np.sum(self.dzvec)
        N    = len(self.dzvec)
        z    = np.arange(0,Z+1)
        
        q = [self.cpvec,self.csvec,self.rovec]
        Q = [z]
        Xlab = []
        Vmin = []
        Vmax = []
        
        for m in range(0,3):
        
            mod = q[m]
            Mod = np.zeros_like(z)
        
            start = 0
            for i in range(0,N):
                stop = zvec[i]
                Mod[start:stop] = mod[i]
                start = stop
                
            Mod[-1] = mod[-1]
            Q.append(Mod)
 
            if m == 0:
                vmin = np.array(self.cpvec).min() - 100
                vmax = np.array(self.cpvec).max() + 100
                title = 'P-Velocity'
            elif m == 1:
                vmin = np.array(self.csvec).min() - 100
                vmax = np.array(self.csvec).max() + 100
                title = 'S-Velocity'
            else:
                vmin = np.array(self.rovec).min() - 100
                vmax = np.array(self.rovec).max() + 100
                title = 'Density'
                
            Xlab.append(title)
            Vmin.append(vmin)
            Vmax.append(vmax)
            
        plt.figure()
        ax0 = plt.subplot2grid((1, 3), (0, 0),colspan=1)
        ax1 = plt.subplot2grid((1, 3), (0, 1),colspan=1)
        ax2 = plt.subplot2grid((1, 3), (0, 2),colspan=1)
        
        ax0.plot(Q[1],z)
        ax1.plot(Q[2],z)
        ax2.plot(Q[3],z)
        
        ax0.set_xlabel(Xlab[0])
        ax1.set_xlabel(Xlab[1])
        ax2.set_xlabel(Xlab[2])
        
        ax0.set_xlim(Vmin[0],Vmax[0])
        ax1.set_xlim(Vmin[1],Vmax[1])
        ax2.set_xlim(Vmin[2],Vmax[2])
        
        ax0.invert_yaxis()
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        
        plt.suptitle('1D Model')
        
        return Q
        
    def L1P_p_w(self,cp,cs,ro,p):
        """
        L1P = L1P_p_w(self,cp,cs,ro,p):
            
        Decomposition matrix L-one-plus for power-flux decomposition for all frequencies and one ray-parameter.
        """
        qp = (1/cp**2 - p**2)**(1/4)
        qs = (1/cs**2 - p**2)**(1/4)
        fac = cs**2*np.sqrt(ro/2)
        L1P = np.zeros((1,4),dtype=complex)
        
        L1P[0,0] = 2*p*qp
        L1P[0,1] = -(1/cs**2 - 2*p**2)/qs
        L1P[0,2] = (1/cs**2 - 2*p**2)/qp
        L1P[0,3] = 2*p*qs
        L1P      = fac*L1P
#        L1P = np.tile(L1P,(self.nf,1))
        return L1P
    
    def L1M_p_w(self,cp,cs,ro,p):
        """
        L1M = L1M_p_w(self,cp,cs,ro,p):
            
        Decomposition matrix L-one-plus for power-flux decomposition for all frequencies and one ray-parameter.
        """
        qp = (1/cp**2 - p**2)**(1/4)
        qs = (1/cs**2 - p**2)**(1/4)
        fac = cs**2*np.sqrt(ro/2)
        L1M = np.zeros((1,4),dtype=complex)
        
        L1M[0,0] = -2*p*qp
        L1M[0,1] = -(1/cs**2 - 2*p**2)/qs
        L1M[0,2] = (1/cs**2 - 2*p**2)/qp
        L1M[0,3] = -2*p*qs
        L1M      = fac*L1M
        return L1M
    
    def L2P_p_w(self,cp,cs,ro,p):
        """
        L2P = L2P_p_w(self,cp,cs,ro,p):
            
        Decomposition matrix L-two-plus for power-flux decomposition for all frequencies and one ray-parameter.
        """
        qp = (1/cp**2 - p**2)**(1/4)
        qs = (1/cs**2 - p**2)**(1/4)
        fac = 1/np.sqrt(2*ro)
        L2P = np.zeros((1,4),dtype=complex)
        
        L2P[0,0] = p/qp
        L2P[0,1] = -qs
        L2P[0,2] = qp
        L2P[0,3] = p/qs
        L2P      = fac*L2P
        return L2P
    
    def L2M_p_w(self,cp,cs,ro,p):
        """
        L2M = L2M_p_w(self,cp,cs,ro,p):
            
        Decomposition matrix L-two-minus for power-flux decomposition for all frequencies and one ray-parameter.
        """
        qp = (1/cp**2 - p**2)**(1/4)
        qs = (1/cs**2 - p**2)**(1/4)
        fac = 1/np.sqrt(2*ro)
        L2M = np.zeros((1,4),dtype=complex)
        
        L2M[0,0] = p/qp
        L2M[0,1] = qs
        L2M[0,2] = -qp
        L2M[0,3] = p/qs
        L2M      = fac*L2M
        return L2M
    
    def RT_p_w(self,cp1,cs1,ro1,cp2,cs2,ro2,p,conv):
        """
        Rplus,Tplus,Rmin,Tmin,Tplusinv = RT_kx_w(self,cp1,cs1,ro1,cp2,cs2,ro2,p)
        
        Compute scattering matrices from above and below for a horizontal interface, and all frequencies and one ray-parameter.
        
        Index 1: Layer avove the interface
        Index 2: Layer below the interface
        """
        
        if cp1==cp2 and cs1==cs2 and ro1==ro2:
            Rplus = np.zeros((1,4))
            Rmin = np.zeros((1,4))
            # The transmission matrices are an identity matrix for all w-kx
            Tplus = np.zeros((1,4))
            Tplus[0,0] = 1
            Tplus[0,3] = 1
            Tmin = Tplus.copy()
            return (Rplus,Tplus,Rmin,Tmin,Tplus)
        
#        if cs1 == 0 or cs2 == 0:
#            Rplus,Tplus,Rminus,Tminus,Tplusinv = self.RTac_kx_w(cp1,cs1,ro1,cp2,cs2,ro2,Kx,Om,conv,nf,nk)
#            return Rplus,Tplus,Rminus,Tminus,Tplusinv
        
        # Compute L1 matrices
        L1Pl = self.L1P_p_w(cp2,cs2,ro2,p)
        L1Mu = self.L1M_p_w(cp1,cs1,ro1,p)
        
        # Compute L2 matrices
        L2Pl = self.L2P_p_w(cp2,cs2,ro2,p)
        L2Mu = self.L2M_p_w(cp1,cs1,ro1,p)
        
        # Compute N1 matrices
        # N1Pu(+kx) = - L2Mu(-kx).T = - { -J L2Mu(+kx) J }.T = { J L2Mu(+kx) J }.T
        N1Pu = L2Mu.copy()
        N1Pu[0,1] = -L2Mu[0,2]   # Transpose and minus
        N1Pu[0,2] = -L2Mu[0,1]   # Transpose and minus
        
        # N1Pl(+kx) = - L2Ml(-kx).T
        N1Pl = - self.L2M_p_w(cp2,cs2,ro2,-p)
        tmp = N1Pl.copy()       # Transpose
        N1Pl[0,1] = tmp[0,2]    # Transpose
        N1Pl[0,2] = tmp[0,1]    # Transpose
        
        # N1Mu(+kx) = L2Pu(-kx).T
        N1Mu =   self.L2P_p_w(cp1,cs1,ro1,-p)
        tmp = N1Mu.copy()       # Transpose
        N1Mu[0,1] = tmp[0,2]    # Transpose
        N1Mu[0,2] = tmp[0,1]    # Transpose
        
        #N1Ml =   L2P_kx_w(cp2,cs2,ro2,-kx,om).T
        
        # Compute N2 matrices
        # N2Pu(+kx) = L1Mu(-kx).T = { -J L1Mu(+kx) J }.T = -{ J L1Mu(+kx) J }.T
        N2Pu = -L1Mu.copy()
        N2Pu[0,1] =  L1Mu[0,2]     # Transpose and minus
        N2Pu[0,2] =  L1Mu[0,1]     # Transpose and minus
        
        # N2Pl(+kx) = L1Ml(-kx).T
        N2Pl =   self.L1M_p_w(cp2,cs2,ro2,-p)
        tmp = N2Pl.copy()       # Transpose
        N2Pl[0,1] = tmp[0,2]    # Transpose
        N2Pl[0,2] = tmp[0,1]    # Transpose
        
        # N2Mu(+kx) = -L1Pu(-kx).T
        N2Mu = - self.L1P_p_w(cp1,cs1,ro1,-p)
        tmp = N2Mu.copy()       # Transpose
        N2Mu[0,1] = tmp[0,2]    # Transpose
        N2Mu[0,2] = tmp[0,1]    # Transpose
        
        #N2Ml = - L1P_kx_w(cp2,cs2,ro2,-kx,om).T   
        
        # Compute scattering matrices
        
        # Inverse of Tplus
        Tplusinv = self.My_dot(N1Pu,L1Pl) + self.My_dot(N2Pu,L2Pl)
        Tplus = self.My_inv(Tplusinv)
        
        # Tmin = J Tplus.T J
        Tmin = Tplus.copy()
        Tmin[0,1] = - Tplus[0,2]    # Tranpose and minus
        Tmin[0,2] = - Tplus[0,1]    # Tranpose and minus
    
        # Rplus = left.Tplus = ( N1Mu.L1Pl + N2Mu.L2Pl ).Tplus
        tmp = self.My_dot(N1Mu,L1Pl) + self.My_dot(N2Mu,L2Pl)
        Rplus = self.My_dot( tmp , Tplus )
        
        # Rmin = ( N1Pl.L1Mu + N2Pl.L2Mu ).Tmin
        tmp = self.My_dot(N1Pl,L1Mu) + self.My_dot(N2Pl,L2Mu)
        Rmin = self.My_dot( tmp , Tmin )
    
        
        if conv==0:
            Tplus[0,1] = 0
            Tplus[0,2] = 0
            Tmin[0,1]  = 0
            Tmin[0,2]  = 0
            Rplus[0,1] = 0
            Rplus[0,2] = 0
            Rmin[0,1]  = 0
            Rmin[0,2]  = 0
        
        return Rplus,Tplus,Rmin,Tmin,Tplusinv
    
#    def RTac_kx_w(self,cp1,cs1,ro1,cp2,cs2,ro2,Kx,Om,conv,nf,nk):
#        """
#        Rplus,Tplus,Rmin,Tmin,Tplusinv = RTac_kx_w(self,cp1,cs1,ro1,cp2,cs2,ro2,kx,om,conv,nf,nk)
#        
#        Compute scattering matrices from above and below for a horizontal interface, and all wavenumbers and frequencies. Here we consider the limit of cs1=0 and/or cs2=0. The below expressions were computed in Mathematica using the Limit function.
#        
#        Index 1: Layer avove the interface
#        Index 2: Layer below the interface
#        """
#        
#        if cs1 == 0 and cs2 == 0:
#            
#            Tplus    = np.zeros((nf,nk,4),dtype=complex)
#            Tminus   = np.zeros((nf,nk,4),dtype=complex)
#            Rplus    = np.zeros((nf,nk,4),dtype=complex)
#            Rminus   = np.zeros((nf,nk,4),dtype=complex)
#            Tplusinv = np.zeros((nf,nk,4),dtype=complex)
#            
#            kzp1 = (Om**2/cp1**2 - Kx**2)**0.5
#            kzp2 = (Om**2/cp2**2 - Kx**2)**0.5
#            
#            denom = ro2*kzp1 + ro1*kzp2
#        
#            Tplus[:,:,0] = 2*np.sqrt(ro1*ro2*kzp1*kzp2)/denom
#            Tminus[:,:,0] = Tplus[:,:,0]
#            Rplus[:,:,0] = (ro2*kzp1 - ro1*kzp2) / denom
#            Rplus[:,:,3] = 1
#            Rminus[:,:,0] = -Rplus[:,:,0]
#            Rminus[:,:,3] = -1
#            
#            # In testing, check theory of Tplusinv
#            Tplusinv[:,:,0] = (ro2*kzp1 + ro1*kzp2) / (2 * np.sqrt(ro1*ro2*kzp1*kzp2) )
#            
#        elif cs1 == 0:
#            
#            Tplus    = np.zeros((nf,nk,4),dtype=complex)
#            Tminus   = np.zeros((nf,nk,4),dtype=complex)
#            Rplus    = np.zeros((nf,nk,4),dtype=complex)
#            Rminus   = np.zeros((nf,nk,4),dtype=complex)
#            Tplusinv = np.zeros((nf,nk,4),dtype=complex)
#            
#            kzp1 = (Om**2/cp1**2 - Kx**2)**0.5
#            kzp2 = (Om**2/cp2**2 - Kx**2)**0.5
#            kzs2 = (Om**2/cs2**2 - Kx**2)**0.5
#
#            # Gives zero division
#            denom = -4*cs2**2*Kx**2*ro2*Om*kzp1 + (ro2*kzp1 + ro1*kzp2)*Om**3 + 4*cs2**4*Kx**2*ro2*kzp1/Om*(Kx**2 + kzp2*kzs2)
#            
#            Tplus[:,:,0] = 2 * (ro1*ro2*kzp1*kzp2)**0.5 * Om * (Om**2 - 2*cs2**2*Kx**2)  / denom # double-checked
#            Tplus[:,:,2] = 4*cs2**2*Kx* (ro1*ro2*kzp1*kzs2)**0.5 *kzp2*Om / denom   # double-checked
#            
#            Tminus[:,:,0] =  Tplus[:,:,0] # double-checked
#            Tminus[:,:,1] = -Tplus[:,:,2] # double-checked
#            
#            # Gives zero division: double-checked
#            Rplus[:,:,0] = (-4*cs2**2*Kx**2*ro2*Om*kzp1 + (ro2*kzp1 - ro1*kzp2)*Om**3 + 4*cs2**4*Kx**2*ro2*kzp1/Om*(Kx**2 + kzp2*kzs2)) / denom
#            Rplus[:,:,3] = 1
#            
#            # Gives zero division: double-checked
#            Rminus[:,:,0] = (4*cs2**2*Kx**2*ro2*Om*kzp1 + (-ro2*kzp1 + ro1*kzp2)*Om**3 + 4*cs2**4*Kx**2*ro2*kzp1/Om*(-Kx**2 + kzp2*kzs2)) / denom
#            Rminus[:,:,1] = 4 * cs2**2 * Kx * ro2 * kzp1 * (kzp2*kzs2)**0.5 / Om * (Om**2 - 2 * cs2**2 * Kx**2)  / denom
#            Rminus[:,:,2] = -Rminus[:,:,1]
#            Rminus[:,:,3] = (4*cs2**2*Kx**2*ro2*Om*kzp1 - (ro2*kzp1 + ro1*kzp2)*Om**3 + 4*cs2**4*Kx**2*ro2*kzp1/Om*(-Kx**2 + kzp2*kzs2)) / denom
#            
#            # In testing, check theory of Tplusinv
#            # Gives zero division
#            Tplusinv[:,:,0] = ( 2 * cs2**2 * Kx**2 * ro2 * (-kzp1 + kzp2) /  Om**2 + (ro2 * kzp1 + ro1 * kzp2) ) / (2 * np.sqrt(ro1*ro2*kzp1*kzp2) )
#            Tplusinv[:,:,1] = Kx * ( (ro1-ro2)*Om**2 + 2 * cs2**2 * ro2 * (Kx**2 + kzp1*kzs2) ) /  ( 2*np.sqrt(ro1*ro2*kzp1*kzs2) * Om**2 )
#            
#        elif cs2 == 0:
#            
#            Tplus    = np.zeros((nf,nk,4),dtype=complex)
#            Tminus   = np.zeros((nf,nk,4),dtype=complex)
#            Rplus    = np.zeros((nf,nk,4),dtype=complex)
#            Rminus   = np.zeros((nf,nk,4),dtype=complex)
#            Tplusinv = np.zeros((nf,nk,4),dtype=complex)
#            
#            kzp1 = (Om**2/cp1**2 - Kx**2)**0.5
#            kzp2 = (Om**2/cp2**2 - Kx**2)**0.5
#            kzs1 = (Om**2/cs1**2 - Kx**2)**0.5
#
#            # Gives zero division: double-checked
#            denom = - 4 * cs1**2 * Kx**2 * ro1 * Om * kzp2 + (ro2*kzp1 + ro1*kzp2)*Om**3 + 4 * cs1**4 * Kx**2 * ro1 * kzp2 / Om * (Kx**2 + kzp1*kzs1)
#            
#            Tplus[:,:,0] = 2 * (ro1*ro2*kzp1*kzp2)**0.5 * Om * (Om**2 - 2 * cs1**2 * Kx**2)  / denom
#            Tplus[:,:,1] = 4 * cs1**2 * Kx * (ro1*ro2*kzp2*kzs1)**0.5 * kzp1 * Om / denom  
#            
#            Tminus[:,:,0] =  Tplus[:,:,0] # double-checked
#            Tminus[:,:,2] = -Tplus[:,:,1] # double-checked
#            
#            # Give all zero division: double-checked
#            Rplus[:,:,0] = (4 * cs1**2 * Kx**2 * ro1 * Om * kzp2 + (ro2*kzp1 - ro1*kzp2) * Om**3 + 4 * cs1**4 * Kx**2 * ro1 * kzp2 / Om * (-Kx**2 + kzp1*kzs1)) / denom
#            Rplus[:,:,1] = -4 * cs1**2 * Kx * ro1 * kzp2 * (kzp1*kzs1)**0.5 / Om * (Om**2 - 2 * cs1**2 * Kx**2) / denom
#            Rplus[:,:,2] = -Rplus[:,:,1]
#            Rplus[:,:,3] = (4 * cs1**2 * Kx**2 * ro1 * Om * kzp2 - (ro2*kzp1 + ro1*kzp2) * Om**3 + 4 * cs1**4 * Kx**2 * ro1 * kzp2 / Om * (-Kx**2 + kzp1*kzs1)) / denom
#            
#            Rminus[:,:,0] = (-4 * cs1**2 * Kx**2 * ro1 * Om * kzp2 + (-ro2*kzp1 + ro1*kzp2) * Om**3 + 4 * cs1**4 * Kx**2 * ro1 * kzp2 / Om * (Kx**2 + kzp1*kzs1)) / denom
#            Rminus[:,:,3] = 1
#            
#            # In testing, check theory of Tplusinv
#            # Give all zero division
#            Tplusinv[:,:,0] = ( 2 * cs1**2 * Kx**2 * ro1 * (kzp1 - kzp2) /  Om**2 + (ro2 * kzp1 + ro1 * kzp2) ) / (2 * np.sqrt(ro1*ro2*kzp1*kzp2) )
#            Tplusinv[:,:,2] = Kx * ( (-ro1+ro2)*Om**2 + 2 * cs1**2 * ro1 * (Kx**2 + kzp2*kzs1) ) /  ( 2*np.sqrt(ro1*ro2*kzp2*kzs1) * Om**2 )
#            
#        # This inverse does not exist, think about it!
#        #Tplusinv = self.My_inv(Tplus)
#        Rplus  = np.nan_to_num(Rplus)
#        Rminus = np.nan_to_num(Rminus)
#        Tplus  = np.nan_to_num(Tplus)
#        Tminus = np.nan_to_num(Tminus)
#        return Rplus,Tplus,Rminus,Tminus,Tplusinv
    
    def Layercel_p_w(self,p,mul=1,conv=1,eps=None,sort=1):
        """
        R,T = Layercel_p_w(p,mul=1,conv=1,eps=None,sort=1)
        Compute reflection / transmission response for a single ray-parameter and all frequencies.
        
        Inputs:
            p:    Ray-parameter.
            mul:  Set mul=1 to model internal multiples.
            conv: Set conv=1 to model P/S conversions.
            eps:  Set eps to add an imaginary constant to the frequency: w -> w + 1j*eps. This reduces the temporal wrap-around but one has to correct for this by multiplying the data by exp(eps*t) in the time domain.
            sort: Set sort=1 (default) to get positive and negative frequencies and wavenumbers.
        
        Output:
            RP: Reflection response from above (nf x 4), 1st element corresponds to zero frequency.
            TP: Transmission response from above (nf x 4), 1st element corresponds to zero frequency.
            RM: Reflection response from below (nf x 4), 1st element corresponds to zero frequency.
            TM: Transmission response from below (nf x 4), 1st element corresponds to zero frequency.
        """
            
        print('Modelling reflection / transmission response for p = %.2f*1e-3 ...'%(p*1e3))
        
        # Number of layers
        N = np.size(self.cpvec)
        
        if eps is None:
            eps = 3/(self.nf*self.dt)
        
        # Frequency and wavenumber meshgrids
        Wfft = self.Wvec()[1]
    
        # Extract positive frequencies including most negative frequency sample
        Wpos  = Wfft[0:self.nf,0] + 1j*eps
        
        # Propagation and scattering matrices of an infinitesimal layer without any contrast
        
        W = np.zeros((self.nf,4),dtype=complex)
        
        RP = np.zeros((self.nf,4),dtype=complex)
        RM = np.zeros((self.nf,4),dtype=complex)
        
        I = np.zeros((self.nf,4),dtype=complex)
        I[:,0] = 1
        I[:,3] = 1
        M1 = I.copy()
        M2 = I.copy()
        
        # Here every frequency and every wavenumber component have an amplitude 
        # equal to one. Hence, the total wavefield has a strength of sqrt(nt*nr)
        # When an inverse fft (ifft2) is applied the wavefield is scaled by 
        # 1/sqrt(nt*nr) hence in the time domain the wavefield has the an
        # amplitude equal to one.
        TP = I.copy()
        TM = I.copy()
        
#        if self.csvec[0] == 0:
#            TP[:,3] = 0
#            TM[:,3] = 0
        
        # Loop over N-1 interfaces
        for n in range(0,N-1):
            
            dz1 = self.dzvec[n]
            
            # Parameters of top layer
            cp1 = self.cpvec[n]
            cs1 = self.csvec[n]
            ro1 = self.rovec[n]
        
            # Parameters of bottom layer
            cp2 = self.cpvec[n+1]
            cs2 = self.csvec[n+1]
            ro2 = self.rovec[n+1]
            
            qp = (1/cp1**2 - p**2)**0.5 # kz/w for p-waves
            qs = (1/cs1**2 - p**2)**0.5 # kz/w for s-waves
                
            W[:,0] = np.exp(1j*Wpos*qp*dz1)
#            if cs1 != 0:
            W[:,3] = np.exp(1j*Wpos*qs*dz1)   # for elastic layer
#            else: 
#                W[:,:,3] = 0                    # for purely acoustic layer
            
            # Outputs.shape = (1,4)
            rP,tP,rM,tM = self.RT_p_w(cp1,cs1,ro1,cp2,cs2,ro2,p,conv)[0:4]
    
            if mul == 1:
                tmp = I - self.Mul_My_dot(RM,W,rP,W)
                # Inverse of tmp
                M1 = self.My_inv(tmp)
                
                tmp = I - self.Mul_My_dot(rP,W,RM,W)
                # Inverse of tmp
                M2 = self.My_inv(tmp)
                
            # Update reflection / transmission responses
            RP = RP + self.Mul_My_dot(TM,W,rP,W,M1,TP)
            RM = rM + self.Mul_My_dot(tP,W,RM,W,M2,tM)
            TP = self.Mul_My_dot(tP,W,M1,TP)
            TM = self.Mul_My_dot(TM,W,M2,tM)      
        
        # Conjugate wavefields
        RP = RP.conj()
        TP = TP.conj()
        RM = RM.conj()
        TM = TM.conj()
        
        # Verbose: Remove NaNs and Infs
        if self.verbose == 1:
            
            if (np.isnan(RP).any() or np.isnan(TP).any() or np.isnan(RM).any() or np.isnan(TM).any()
                or np.isinf(RP).any() or np.isinf(TP).any() or np.isinf(RM).any() or np.isinf(TM).any()):
                print('\n')
                print('Layercel_p_w:')
                print('\n'+100*'-'+'\n')
                print('One of the modelled wavefields contains a NaN (Not a Number) or an Inf (infinite) element. '+
                      'In this step, NaN is replaced by zero, and infinity (-infinity) is replaced by the largest '+
                      '(smallest or most negative) floating point value that fits in the output dtype. Also see '+
                      'numpy.nan_to_num (in numpy or scipy documentation).')
                print('\n')
                
                if np.isnan(RP).any():
                    print('\t - RP contains '+np.count_nonzero(np.isnan(RP))+' NaN.')
                if np.isinf(RP).any():
                    print('\t - RP contains '+np.count_nonzero(np.isinf(RP))+' Inf.')
                if np.isnan(TP).any():
                    print('\t - TP contains '+np.count_nonzero(np.isnan(TP))+' NaN.')
                if np.isinf(TP).any():
                    print('\t - TP contains '+np.count_nonzero(np.isinf(TP))+' Inf.')
                if np.isnan(RM).any():
                    print('\t - RM contains '+np.count_nonzero(np.isnan(RM))+' NaN.')
                if np.isinf(RM).any():
                    print('\t - RM contains '+np.count_nonzero(np.isinf(RM))+' Inf.')
                if np.isnan(TM).any():
                    print('\t - TM contains '+np.count_nonzero(np.isnan(TM))+' NaN.')
                if np.isinf(TM).any():
                    print('\t - TM contains '+np.count_nonzero(np.isinf(TM))+' Inf.')
            
                print('\n')

        # Remove NaNs and Infs
        RP = np.nan_to_num(RP)
        TP = np.nan_to_num(TP)
        RM = np.nan_to_num(RM)
        TM = np.nan_to_num(TM)
        
        # Write full reflection and transmission matrices
        if sort == 1:
            RP,TP,RM,TM = self.Sort_w(RP,TP,RM,TM)
            return RP,TP,RM,TM
        
        return RP,TP,RM,TM
    
    def Sort_w(self,*args):
        """
        Out = Sort_w(*args)
        
        *args are wavefield matrices that only contain positive w elements.
        Sort_w exploits symmetry of wavefields in layered media to contruct the negative wavenumbers.
        Sort_w exploits causality to construct the negative frequency components of the reflection/transmission.
        
        Input: 
            *args: Wavefield matrix/matrices in w-p-domain for zero-positive w including most negative w element (real-valued).
        
        Output: 
            Out: Wavefield matrix/matrices in w-p-domain for positive and negative w and a single p.
        """

        Out = []

        for i in np.arange(0,len(args)):
            F = args[i]
            
            # Preallocate wavefield matrix
            Ffull = np.zeros((self.nt,4),dtype=complex)
            
            # Copy all positive wavenumbers including most negative wavenumber
            Ffull[0:self.nf,:] = F.copy()
            
            # Apply causality to get negative frequencies: F(-w,p) = F(w,p).conj()
            
            # Copy positive w
            Ffull[self.nf:,:]  = Ffull[self.nf-2:0:-1,:]   
            
            # Conjugate negative w
            Ffull[self.nf:,:]  = Ffull[self.nf:,:].conj()
            
            Out.append(Ffull)
        
        if len(args) == 1:
            Out = Out[0]
        
        return Out
    
    def Reverse_p(self,array):
        """
        Reverse the p-value of an array.
        
        Inputs:
            array:  An array of the shape (nt x 4).
            
        Outputs:
            array:  Input array with reversed p value.
            
        """
        # F(-p) = J F(p) J
        Prev = array.copy()
        Prev[:,1] = -array[:,1]
        Prev[:,2] = -array[:,2]
        return Prev
    
    # Green's function between depth z and surface
    def Gz2surf(self,p,z,mul=1,conv=1,eps=None):
        """
        GMP,GPP,GMP2,GMM = Gz2surf(p,z,mul=1,conv=1,eps=None)
        Compute one-way Green's matrices for a single ray-parameter and all frequencies. 
        
        Inputs:
            p:    Ray-parameter.
            mul:  Set mul=1 to model internal multiples.
            conv: Set conv=1 to model P/S conversions.
            eps:  Set eps to add an imaginary constant to the frequency: w -> w + 1j*eps. This reduces the temporal wrap-around but one has to correct for this by multiplying the data by exp(eps*t) in the time domain.
        
        Output:
            GMP:  Green's matrix for a downgping source at the top boundary and an upgoing receiver at z. (nf x 4), 1st element corresponds to zero frequency.
            GPP:  Green's matrix for a downgoing source at the top boundary and an downgoing receiver at z. (nf x 4), 1st element corresponds to zero frequency.
            GMP2: Green's matrix for a downgoing source at z and an upgoing receiver at the top boundary. (nf x 4), 1st element corresponds to zero frequency.
            GMM:  Green's matrix for an upgoing source at z and an upgoing receiver at the top boundary. (nf x 4), 1st element corresponds to zero frequency.
        """
        
        print('Modelling Greens functions to upper boundary for p = %.2f*1e-3 ...'%(p*1e3))
        
        # Insert layers
        self.Insert_layer(z)
        
        # Layer 
        N = np.cumsum(self.dzvec).tolist().index(z)+1
        
        # Number of layers
        NN = np.size(self.cpvec)
        
        if eps is None:
            eps = 3/(self.nf*self.dt)
        
        # Frequency and wavenumber meshgrids
        Wfft = self.Wvec()[1]
    
        # Extract positive frequencies including the highest negative frequency sample
        Wpos  = Wfft[0:self.nf,0] + 1j*eps
        
        # Propagation and scattering matrices of an infinitesimal layer without any contrast
        
        W = np.zeros((self.nf,4),dtype=complex)
        
        RP  = np.zeros((self.nf,4),dtype=complex)
        RPP = np.zeros((self.nf,4),dtype=complex)
        RM  = np.zeros((self.nf,4),dtype=complex)
        
        I = np.zeros((self.nf,4),dtype=complex)
        I[:,0] = 1
        I[:,3] = 1
        M1 = I.copy()
        M2 = I.copy()
        M3 = I.copy()
        
        # Here every frequency and every wavenumber component have an amplitude 
        # equal to one. Hence, the total wavefield has a strength of sqrt(nt*nr)
        # When an inverse fft (ifft2) is applied the wavefield is scaled by 
        # 1/sqrt(nt*nr) hence in the time domain the wavefield has the an
        # amplitude equal to one.
        TP = I.copy()
        TM = I.copy()
        
#        if self.csvec[0] == 0:
#            TP[:,:,3] = 0
#            TM[:,:,3] = 0
        
        # Start the first (downward) recursion loop over upper N-1 layers
        for n in range(0,N):
        
            dz1 = self.dzvec[n]
            
            # Parameters of top layer
            cp1 = self.cpvec[n]
            cs1 = self.csvec[n]
            ro1 = self.rovec[n]
        
            # Parameters of bottom layer
            cp2 = self.cpvec[n+1]
            cs2 = self.csvec[n+1]
            ro2 = self.rovec[n+1]
            
            qp = (1/cp1**2 - p**2)**0.5 # kz/w for p-waves
            qs = (1/cs1**2 - p**2)**0.5 # kz/w for s-waves
                
            W[:,0] = np.exp(1j*Wpos*qp*dz1)
#            if cs1 != 0:
            W[:,3] = np.exp(1j*Wpos*qs*dz1)   # for elastic layer
#            else: 
#                W[:,:,3] = 0                    # for purely acoustic layer
        
            rP,tP,rM,tM = self.RT_p_w(cp1,cs1,ro1,cp2,cs2,ro2,p,conv)[0:4]
    
            if mul == 1:
                tmp = I - self.Mul_My_dot(RM,W,rP,W)
                # Inverse of tmp
                M1 = self.My_inv(tmp)
                
                tmp = I - self.Mul_My_dot(rP,W,RM,W)
                # Inverse of tmp
                M2 = self.My_inv(tmp)
                
            # Update reflection / transmission responses
            RP = RP + self.Mul_My_dot(TM,W,rP,W,M1,TP)
            RM = rM + self.Mul_My_dot(tP,W,RM,W,M2,tM)
            TP = self.Mul_My_dot(tP,W,M1,TP)
            TM = self.Mul_My_dot(TM,W,M2,tM) 
            
        #Start the second (upward) recursion loop over the NN-N layers
        for m in range(2,NN-N+1):
        
            n	= NN-m
            
            dz1 = self.dzvec[n]
            
            # Parameters of top layer
            cp1 = self.cpvec[n]
            cs1 = self.csvec[n]
            ro1 = self.rovec[n]
        
            # Parameters of bottom layer
            cp2 = self.cpvec[n+1]
            cs2 = self.csvec[n+1]
            ro2 = self.rovec[n+1]
            
            qp = (1/cp1**2 - p**2)**0.5 # kz/w for p-waves
            qs = (1/cs1**2 - p**2)**0.5 # kz/w for s-waves
                
            W[:,0] = np.exp(1j*Wpos*qp*dz1)
#            if cs1 != 0:
            W[:,3] = np.exp(1j*Wpos*qs*dz1)   # for elastic layer
#            else: 
#                W[:,:,3] = 0                    # for purely acoustic layer
        
            rP,tP,rM,tM = self.RT_p_w(cp1,cs1,ro1,cp2,cs2,ro2,p,conv)[0:4]
       
            if mul==1:
                M3	 = self.My_inv(I - self.My_dot(rM,RPP))

            RPP =	self.Mul_My_dot( W,(rP + self.Mul_My_dot(tM,RPP,M3,tP)),W )
            
        # Compute the Green's functions
        if mul == 1:
            M = I - self.My_dot(RM,RPP)
            GPP =	self.My_dot(self.My_inv(M),TP)
            M = I - self.My_dot(RPP,RM)
            GMP =	self.Mul_My_dot(self.My_inv(M),RPP,TP)
        elif mul == 0:
            GPP = TP
            GMP = self.My_dot(RPP,TP)
            
        # Remove layers
        self.Remove_layer(z)
        
        # Conjugate wavefields
        GPP = GPP.conj()
        GMP = GMP.conj()
        
        # Verbose: Remove NaNs and Infs
        if self.verbose == 1:
            
            if np.isnan(GMP).any() or np.isnan(GPP).any() or np.isinf(GMP).any() or np.isinf(GPP).any():
                print('\n')
                print('Gz2surf:')
                print('\n'+100*'-'+'\n')
                print('One of the modelled wavefields contains a NaN (Not a Number) or an Inf (infinite) element. '+
                      'In this step, NaN is replaced by zero, and infinity (-infinity) is replaced by the largest '+
                      '(smallest or most negative) floating point value that fits in the output dtype. Also see '+
                      'numpy.nan_to_num (in numpy or scipy documentation).')
                print('\n')
                
                if np.isnan(GMP).any():
                    print('\t - GMP contains '+np.count_nonzero(np.isnan(GMP))+' NaN.')
                if np.isinf(GMP).any():
                    print('\t - GMP contains '+np.count_nonzero(np.isinf(GMP))+' Inf.')
                if np.isnan(GPP).any():
                    print('\t - GPP contains '+np.count_nonzero(np.isnan(GPP))+' NaN.')
                if np.isinf(GPP).any():
                    print('\t - GPP contains '+np.count_nonzero(np.isinf(GPP))+' Inf.')
            
                print('\n')
        
        # Delete NaN's and limit inf's
        GMP  = np.nan_to_num(GMP) 
        GPP  = np.nan_to_num(GPP)
        
        # Construct negative frequency samples for a real-valued time signal
        GMP,GPP = self.Sort_w(GMP,GPP)
    
        # Apply source - receiver reciprocity
        
        # GMM(p) = -GPP(-p).T (Paper Physical review E 2014 Eq. A.16) 
        GMM = -self.My_T(GPP)             # Transpose and sign invert GPP
        GMM = self.Reverse_p(GMM)         # Reverese p

        # GMP2(p) = GMP(-p).T (Paper Physical review E 2014 Eq. A.16) 
        GMP2 = self.My_T(GMP)               # Transpose GMP
        GMP2 = self.Reverse_p(GMP2)         # Reverese p
        
        return GMP,GPP,GMP2,GMM

    
    # Green's function between depth z and bottom
    def Gz2bot(self,p,z,mul=1,conv=1,eps=None):
        """
        GPM,GMM,GPM2,GPP = Gz2bot(p,z,mul=1,conv=1,eps=None)
        Compute one-way Green's matrices for a single ray-parameter and all frequencies. 
        
        Inputs:
            p:    Ray-parameter.
            mul:  Set mul=1 to model internal multiples.
            conv: Set conv=1 to model P/S conversions.
            eps:  Set eps to add an imaginary constant to the frequency: w -> w + 1j*eps. This reduces the temporal wrap-around but one has to correct for this by multiplying the data by exp(eps*t) in the time domain.
        
        Output:
            GPM:  Green's matrix for a upgping source at the bottom boundary and a downgoing receiver at z. (nf x 4), 1st element corresponds to zero frequency.
            GMM:  Green's matrix for a upgping source at the bottom boundary and an upgoing receiver at z. (nf x 4), 1st element corresponds to zero frequency.
            GPM2: Green's matrix for an upgoing source at z and a downgoing receiver at the bottom boundary. (nf x 4), 1st element corresponds to zero frequency.
            GPP:  Green's matrix for a downgoing source at z and a downgoing receiver at the bottom boundary. (nf x 4), 1st element corresponds to zero frequency.
        """
        
        print('Modelling Greens functions to lower boundary for p = %.2f*1e-3 ...'%(p*1e3))
        
        # Insert layers
        self.Insert_layer(z)
        
        # Index of the interface 
        N = np.cumsum(self.dzvec).tolist().index(z)
        
        # Number of layers
        NN = np.size(self.cpvec)
        
        if eps is None:
            eps = 3/(self.nf*self.dt)
        
        # Frequency vector
        Wfft = self.Wvec()[1]
    
        # Extract positive frequencies 
        Wpos  = Wfft[0:self.nf,0] + 1j*eps
        
        # Propagation and scattering matrices of an infinitesimal layer without any contrast
        
        W = np.zeros((self.nf,4),dtype=complex)
        
        RP  = np.zeros((self.nf,4),dtype=complex)
        RM  = np.zeros((self.nf,4),dtype=complex)
        
        I = np.zeros((self.nf,4),dtype=complex)
        I[:,0] = 1
        I[:,3] = 1
        
        # Here every frequency and every wavenumber component have an amplitude 
        # equal to one. Hence, the total wavefield has a strength of sqrt(nt*nr)
        # When an inverse fft (ifft2) is applied the wavefield is scaled by 
        # 1/sqrt(nt*nr) hence in the time domain the wavefield has the an
        # amplitude equal to one.
        TP  = I.copy()     # Upgoing sources have negative amplitude
        TM  = I.copy()     # Upgoing sources have negative amplitude
        
        if self.csvec[N+1] == 0:
            TP[:,3] = 0
            TM[:,3] = 0
        
        # This loop will compute RP of the lower bit of the medium
        # Start the first (downward) loop over lower layers from N+1 to len(dz)-1 (excluded)
        for n in range(N+1,NN-1):
            
            # Parameters of top layer
            dz1 = self.dzvec[n]
            cp1 = self.cpvec[n]
            cs1 = self.csvec[n]
            ro1 = self.rovec[n]
        
            # Parameters of bottom layer
            cp2 = self.cpvec[n+1]
            cs2 = self.csvec[n+1]
            ro2 = self.rovec[n+1]
            
            qp = (1/cp1**2 - p**2)**0.5 # kz/w for p-waves
            qs = (1/cs1**2 - p**2)**0.5 # kz/w for s-waves
                
            W[:,0] = np.exp(1j*Wpos*qp*dz1)
#            if cs1 != 0:
            W[:,3] = np.exp(1j*Wpos*qs*dz1)   # for elastic layer
#            else: 
#                W[:,:,3] = 0                    # for purely acoustic layer
        
            rP,tP,rM,tM = self.RT_p_w(cp1,cs1,ro1,cp2,cs2,ro2,p,conv)[0:4]
            
            if mul == 1:
                M1 = self.My_inv( I - self.Mul_My_dot(RM,W,rP,W) )
                M2 = self.My_inv( I - self.Mul_My_dot(rP,W,RM,W) )
            
            RP = RP + self.Mul_My_dot(TM,W,rP,W,M1,TP)
            RM = rM + self.Mul_My_dot(tP,W,RM,W,M2,tM) 
            TP = self.Mul_My_dot(tP,W,M1,TP)
            TM = self.Mul_My_dot(TM,W,M2,tM)
        
        # Parameters of bottom layer
        dz = self.dzvec[NN-1]
        cp = self.cpvec[NN-1]
        cs = self.csvec[NN-1]
        qp = (1/cp**2 - p**2)**0.5 # kz/w for p-waves
        qs = (1/cs**2 - p**2)**0.5 # kz/w for s-waves
            
        W[:,0] = np.exp(1j*Wpos*qp*dz)
#        if cs != 0:
        W[:,3] = np.exp(1j*Wpos*qs*dz)   # for elastic layer
#        else: 
#            W[:,:,3] = 0                    # for purely acoustic layer
        
        # Propagate responses from below through the bottom layer
        TM = self.My_dot(TM,W)
        RM = self.Mul_My_dot(W,RM,W)
        
        RMM  = np.zeros((self.nf,4),dtype=complex)
            
        # Start the second (downward) loop over the upper layers from zero to N (excluded)
        for n in range(0,N+1):
            
            # Parameters of top layer
            dz1 = self.dzvec[n]
            cp1 = self.cpvec[n]
            cs1 = self.csvec[n]
            ro1 = self.rovec[n]
        
            # Parameters of bottom layer
            cp2 = self.cpvec[n+1]
            cs2 = self.csvec[n+1]
            ro2 = self.rovec[n+1]
            
            qp = (1/cp1**2 - p**2)**0.5 # kz/w for p-waves
            qs = (1/cs1**2 - p**2)**0.5 # kz/w for s-waves
                
            W[:,0] = np.exp(1j*Wpos*qp*dz1)
#            if cs1 != 0:
            W[:,3] = np.exp(1j*Wpos*qs*dz1)   # for elastic layer
#            else: 
#                W[:,:,3] = 0                    # for purely acoustic layer
            
            rP,tP,rM,tM = self.RT_p_w(cp1,cs1,ro1,cp2,cs2,ro2,p,conv)[0:4]
       
            if mul==1:
                M3 = self.My_inv(I - self.Mul_My_dot(rP,W,RMM,W))

            RMM =	rM + self.Mul_My_dot( tP,W,RMM,W,M3,tM )
            
        # Compute the Green's functions
        if mul == 1:
            M = self.My_inv( I - self.My_dot(RP,RMM) )
            # Multiply by -1 because upgoing sources are defined with negative amplitude
            GMM =	-self.My_dot(M,TM)
            GPM =	-self.Mul_My_dot(RMM,M,TM)
        elif mul == 0:
            # Multiply by -1 because upgoing sources are defined with negative amplitude
            GMM = -TM
            GPM = -self.My_dot(RMM,TM)

        # Remove layer
        self.Remove_layer(z)

        # Conjugate wavefields
        GMM = GMM.conj()
        GPM = GPM.conj()
        
        # Verbose: Remove NaNs and Infs
        if self.verbose == 1:
            
            if np.isnan(GMM).any() or np.isnan(GPM).any() or np.isinf(GMM).any() or np.isinf(GPM).any():
                print('\n')
                print('Gz2surf:')
                print('\n'+100*'-'+'\n')
                print('One of the modelled wavefields contains a NaN (Not a Number) or an Inf (infinite) element. '+
                      'In this step, NaN is replaced by zero, and infinity (-infinity) is replaced by the largest '+
                      '(smallest or most negative) floating point value that fits in the output dtype. Also see '+
                      'numpy.nan_to_num (in numpy or scipy documentation).')
                print('\n')
                
                if np.isnan(GMM).any():
                    print('\t - GMM contains '+np.count_nonzero(np.isnan(GMM))+' NaN.')
                if np.isinf(GMM).any():
                    print('\t - GMM contains '+np.count_nonzero(np.isinf(GMM))+' Inf.')
                if np.isnan(GPM).any():
                    print('\t - GPM contains '+np.count_nonzero(np.isnan(GPM))+' NaN.')
                if np.isinf(GPM).any():
                    print('\t - GPM contains '+np.count_nonzero(np.isinf(GPM))+' Inf.')
            
                print('\n')
        
        # Delete NaN's and limit inf's
        GMM  = np.nan_to_num(GMM)
        GPM  = np.nan_to_num(GPM) 
        
        GPM,GMM = self.Sort_w(GPM,GMM)
        
        
        # Apply source - receiver reciprocity
        
        # GPP(p) = -GMM(-p).T
        GPP = -self.My_T(GMM)        # Transpose and sign invert GMM      
        GPP = self.Reverse_p(GPP)    # Reverese p

        # GPM2(p) = GPM(-p).T
        GPM2 = self.My_T(GPM)         # Transpose GMP
        GPM2 = self.Reverse_p(GPM2)   # Reverese p
        
        return GPM,GMM,GPM2,GPP
    
    
    # Green's function between depth z and surface
    def Gz2bound(self,p,z,mul=1,conv=1,eps=None,initials=[]):
        """
        Gs,Gb,initials = Gz2bound(self,p,z,mul=1,conv=1,eps=None,initials=[])
        
        Compute one-way Green's matrices between just below depth level z and the top and bottom boundaries, for a single ray-parameter and all frequencies. 
        
        Inputs:
            p:          Ray-parameter.
            z:          Depth level inside the model.
            mul:        Set mul=1 to model internal multiples.
            conv:       Set conv=1 to model P/S conversions.
            eps:        Set eps to add an imaginary constant to the frequency: w -> w + 1j*eps. This reduces the temporal wrap-around but one has to correct for this by multiplying the data by exp(eps*t) in the time domain.
            initials:   List of reflection and transmission responses of the overburden, i.e. between the top boundary and a depth level z from a prior computation. The list initials allows to reuse previously computed medium responses to avoid repeating the computation.
        
        Output:
            Gs:         Green's matrices for sources and receives at the top boundary and at z. (nf x 4), 1st element corresponds to zero frequency.
            Gb:         Green's matrices for sources and receives at the bottom boundary and at z. (nf x 4), 1st element corresponds to zero frequency.
            initials:   List of reflection and transmission responses of the overburden, i.e. between the top boundary and z. If in a next step z is set to a deeper level the responses of initials can be reused to avoid repeating the computation.
       
            Gs       = [GsMP,GsPP,GsMP2,GsMM]
            Gb       = [GbPM,GbMM,GbPM2,GbPP]
            initials = [RP,RM,TP,TM,z]
        """
        
        print('Modelling Greens functions to lower and upper boundary for p = %.2f*1e-3 ...'%(p*1e3))
        
        # Insert layers
        self.Insert_layer(z)
        if initials != []:
            self.Insert_layer(initials[-1])
        
        # To avoid indexing errors when z = model depth
        remove = 0
        if z == np.cumsum(self.dzvec)[-1]:
            self.Insert_layer(np.cumsum(self.dzvec)[-1] + 1)
            remove = 1
        
        # Layer 
        N = np.cumsum(self.dzvec).tolist().index(z)
        
        # Number of layers
        NN = np.size(self.cpvec)
        
        if eps is None:
            eps = 3/(self.nf*self.dt)
        
        # Frequency vector
        Wfft = self.Wvec()[1]
    
        # Extract positive frequencies including the highest negative frequency sample
        Wpos  = Wfft[0:self.nf,0] + 1j*eps
        
        # Propagation matrix of an infinitesimal layer without any contrast
        W = np.zeros((self.nf,4),dtype=complex)
        I = np.zeros((self.nf,4),dtype=complex)
        I[:,0] = 1
        I[:,3] = 1
        
        # Preallocate multiple generators in case of mul=0
        M1 = I.copy()
        M2 = I.copy()
        
        if initials == []:
            # Upper half of the medium
            # Initial scattering matrices of an infinitesimal layer without any contrast
            RP  = np.zeros((self.nf,4),dtype=complex)
            RM  = np.zeros((self.nf,4),dtype=complex)
            # Here every frequency and every wavenumber component have an amplitude 
            # equal to one. Hence, the total wavefield has a strength of sqrt(nt*nr)
            # When an inverse fft (ifft2) is applied the wavefield is scaled by 
            # 1/sqrt(nt*nr) hence in the time domain the wavefield has the an
            # amplitude equal to one.
            TP = I.copy()
            TM = I.copy()
            
#            if self.csvec[0] == 0:
#                TP[:,3] = 0
#                TM[:,3] = 0
            
            Nstart = 0
            
        else:
            RP,RM,TP,TM,zstart = initials 
            Nstart = np.cumsum(self.dzvec).tolist().index(zstart) + 1
        
        
        # Lower half of the medium
        RPb  = np.zeros((self.nf,4),dtype=complex)
        RMb  = np.zeros((self.nf,4),dtype=complex)
        TPb = I.copy()
        TMb = I.copy()
        

#        if self.csvec[N+1] == 0:
#            TPb[:,:,3] = 0
#            TMb[:,:,3] = 0
        
        
        # Start the first (downward) recursion loop over upper N-1 layers
        for n in range(Nstart,N+1):
        
            dz1 = self.dzvec[n]
            
            # Parameters of top layer
            cp1 = self.cpvec[n]
            cs1 = self.csvec[n]
            ro1 = self.rovec[n]
        
            # Parameters of bottom layer
            cp2 = self.cpvec[n+1]
            cs2 = self.csvec[n+1]
            ro2 = self.rovec[n+1]
            
            # Propagator through top layer
            qp = (1/cp1**2 - p**2)**0.5 # kz/w for p-waves
            qs = (1/cs1**2 - p**2)**0.5 # kz/w for s-waves
            
            W[:,0] = np.exp(1j*Wpos*qp*dz1)
#            if cs1 != 0:
            W[:,3] = np.exp(1j*Wpos*qs*dz1)   # for elastic layer
#            else: 
#                W[:,:,3] = 0                    # for purely acoustic layer
        
            rP,tP,rM,tM = self.RT_p_w(cp1,cs1,ro1,cp2,cs2,ro2,p,conv)[0:4]
    
            if mul == 1:
                tmp = I - self.Mul_My_dot(RM,W,rP,W)
                # Inverse of tmp
                M1 = self.My_inv(tmp)
                
                tmp = I - self.Mul_My_dot(rP,W,RM,W)
                # Inverse of tmp
                M2 = self.My_inv(tmp)
                
            # Update reflection / transmission responses
            RP = RP + self.Mul_My_dot(TM,W,rP,W,M1,TP)
            RM = rM + self.Mul_My_dot(tP,W,RM,W,M2,tM)
            TP = self.Mul_My_dot(tP,W,M1,TP)
            TM = self.Mul_My_dot(TM,W,M2,tM) 

            
        # This loop will compute RP of the lower bit of the medium
        # Start the first (downward) loop over lower layers from N+1 to len(dz)-1 (excluded)
        for n in range(N+1,NN-1):
            
            # Parameters of top layer
            dz1 = self.dzvec[n]
            cp1 = self.cpvec[n]
            cs1 = self.csvec[n]
            ro1 = self.rovec[n]
        
            # Parameters of bottom layer
            cp2 = self.cpvec[n+1]
            cs2 = self.csvec[n+1]
            ro2 = self.rovec[n+1]
            
            # Propagator through top layer
            qp = (1/cp1**2 - p**2)**0.5 # kz/w for p-waves
            qs = (1/cs1**2 - p**2)**0.5 # kz/w for s-waves
            
            W[:,0] = np.exp(1j*Wpos*qp*dz1)
#            if cs1 != 0:
            W[:,3] = np.exp(1j*Wpos*qs*dz1)   # for elastic layer
#            else: 
#                W[:,:,3] = 0                    # for purely acoustic layer
        
            rP,tP,rM,tM = self.RT_p_w(cp1,cs1,ro1,cp2,cs2,ro2,p,conv)[0:4]
            
            if mul == 1:
                M1 = self.My_inv( I - self.Mul_My_dot(RMb,W,rP,W) )
                M2 = self.My_inv( I - self.Mul_My_dot(rP,W,RMb,W) )
            
            RPb = RPb + self.Mul_My_dot(TMb,W,rP,W,M1,TPb)
            RMb = rM + self.Mul_My_dot(tP,W,RMb,W,M2,tM) 
            TPb = self.Mul_My_dot(tP,W,M1,TPb)
            TMb = self.Mul_My_dot(TMb,W,M2,tM)
        
        if remove == 0:
            # Parameters of bottom layer
            dz = self.dzvec[NN-1]
            cp = self.cpvec[NN-1]
            cs = self.csvec[NN-1]
            qp = (1/cp**2 - p**2)**0.5 # kz/w for p-waves
            qs = (1/cs**2 - p**2)**0.5 # kz/w for s-waves    
        
            W[:,0] = np.exp(1j*Wpos*qp*dz)
#            if cs != 0:
            W[:,3] = np.exp(1j*Wpos*qs*dz)   # for elastic layer
#            else: 
#                W[:,:,3] = 0                    # for purely acoustic layer
            
            # Propagate responses from below through the bottom layer
            TMb = self.My_dot(TMb,W)
            RMb = self.Mul_My_dot(W,RMb,W)
        
        # Remove layers
        self.Remove_layer(z)
        if initials != []:
            self.Remove_layer(initials[-1])
            
        # To avoid indexing errors when z = model depth
        if remove == 1:
#            self.Remove_layer(z+1)
            self.Remove_layer(np.cumsum(self.dzvec)[-1])
        
        # Save inital fields for next call of Gz2bound
        initials = [RP,RM,TP,TM,z]
            
        
        # Compute the Green's functions
        if mul == 1:
            
            # To surface
            M = self.My_inv( I - self.My_dot(RM,RPb) )
            GsPP =	self.My_dot(M,TP)
            GsMP =	self.Mul_My_dot(RPb,M,TP)
            
            # To bottom
            M = self.My_inv( I - self.My_dot(RPb,RM) )
            GbMM =	-self.My_dot(M,TMb)                   # Multiply by -1 because upgoing sources are defined with negative amplitude
            GbPM =	-self.Mul_My_dot(RM,M,TMb)            # Multiply by -1 because upgoing sources are defined with negative amplitude
            
        elif mul == 0:
            
            # To surface
            GsPP = TP
            GsMP = self.My_dot(RPb,TP)
            
            # To bottom
            GbMM =	-self.My_dot(TMb)                   # Multiply by -1 because upgoing sources are defined with negative amplitude
            GbPM =	-self.Mul_My_dot(RM,TMb)            # Multiply by -1 because upgoing sources are defined with negative amplitude

        # Verbose: Remove NaNs and Infs
        if self.verbose == 1:
            
            if ( np.isnan(GsPP).any() or np.isnan(GsMP).any() or np.isinf(GsPP).any() or np.isinf(GsMP).any() or
                 np.isnan(GbMM).any() or np.isnan(GbPM).any() or np.isinf(GbMM).any() or np.isinf(GbPM).any() ):
                print('\n')
                print('Gz2bound:')
                print('\n'+100*'-'+'\n')
                print('At least one of the modelled wavefields contains a NaN (Not a Number) or an Inf (infinite) element. '+
                      'In this step, NaN is replaced by zero, and infinity (-infinity) is replaced by the largest '+
                      '(smallest or most negative) floating point value that fits in the output dtype. Also see '+
                      'numpy.nan_to_num (in numpy or scipy documentation).')
                print('\n')
                
                if np.isnan(GsPP).any():
                    print('\t - GsPP contains '+np.count_nonzero(np.isnan(GsPP))+' NaN.')
                if np.isinf(GsPP).any():
                    print('\t - GsPP contains '+np.count_nonzero(np.isinf(GsPP))+' Inf.')
                if np.isnan(GsMP).any():
                    print('\t - GsMP contains '+np.count_nonzero(np.isnan(GsMP))+' NaN.')
                if np.isinf(GsMP).any():
                    print('\t - GsMP contains '+np.count_nonzero(np.isinf(GsMP))+' Inf.')
                if np.isnan(GbMM).any():
                    print('\t - GbMM contains '+np.count_nonzero(np.isnan(GbMM))+' NaN.')
                if np.isinf(GbMM).any():
                    print('\t - GbMM contains '+np.count_nonzero(np.isinf(GbMM))+' Inf.')
                if np.isnan(GbPM).any():
                    print('\t - GbPM contains '+np.count_nonzero(np.isnan(GbPM))+' NaN.')
                if np.isinf(GbPM).any():
                    print('\t - GbPM contains '+np.count_nonzero(np.isinf(GbPM))+' Inf.')
            
                print('\n')
        
        # G to surface
        
        # Conjugate wavefields
        GsPP = GsPP.conj()
        GsMP = GsMP.conj()
        
        # Delete NaN's and limit inf's
        GsPP = np.nan_to_num(GsPP)
        GsMP = np.nan_to_num(GsMP) 
        
        GsMP,GsPP = self.Sort_w(GsMP,GsPP)
        
        # Apply source - receiver reciprocity
        
        # GMM(p) = -GPP(-p).T
        GsMM = -self.My_T(GsPP)        # Transpose and sign invert GPP
        GsMM = self.Reverse_p(GsMM)    # Reverese p

        # GMP2(p) = GMP(-p).T
        GsMP2 = self.My_T(GsMP)          # Transpose GMP
        GsMP2 = self.Reverse_p(GsMP2)    # Reverese p
        
        # Save Green's functions in list
        # Elements 0:2 source at bottom
        # Elements 3:4 source in medium
        Gs = [GsMP,GsPP,GsMP2,GsMM]
        
        
        # G to bottom
        
        # Conjugate wavefields
        GbMM = GbMM.conj()
        GbPM = GbPM.conj()
        
        # Delete NaN's and limit inf's
        GbMM = np.nan_to_num(GbMM)
        GbPM = np.nan_to_num(GbPM)    
        
        GbPM,GbMM = self.Sort_w(GbPM,GbMM)
        
        # Apply source - receiver reciprocity
        
        # GMM(p) = -GPP(-p).T
        GbPP = -self.My_T(GbMM)        # Transpose and sign invert GPP
        GbPP = self.Reverse_p(GbPP)    # Reverese p

        # GMP2(p) = GMP(-p).T
        GbPM2 = self.My_T(GbPM)         # Transpose GMP
        GbPM2 = self.Reverse_p(GbPM2)   # Reverese p
        
        # Save Green's functions in list
        # Elements 0:2 source at bottom
        # Elements 3:4 source in medium
        Gb = [GbPM,GbMM,GbPM2,GbPP]
        
        return Gs,Gb,initials
    
    
    
    # Green's function between two depth levels: Receiver at zR, source at zS
    def GzRzS(self,p,zR,zS,mul=1,conv=1,eps=None,initials=[]):
        """
        G,initials = GzRzS(p,zR,zS,mul=1,conv=1,eps=None,initials=[])
        
        Compute one-way Green's matrices for a source just below zS and a receiver just below zR, for a single ray-parameter and all frequencies. 
        
        Inputs:
            p:          Ray-parameter (single value).
            zR:         Receiver depth level
            zS:         Source depth level
            mul:        Default mul=1. Set mul=1 to include internal multiples. Set mul=0 to exclude internal multiples.
            conv:       Default conv=1. Set conv=1 to include mode conversions. Set conv=0 to exclude mode conversions.
            eps:        Default eps=None. Imaginary constant that is added to the temporal frequencies (1) to avoid division by zero at zero frequency, (2) to reduce temporal wrap-arounds.
            initials:   Default initials=[]. If, for a fix source depth level, multiple receiver depth levels are modelled, part of the modelled data can be re-used at receiver depth level to reduce the computation time.
            
        Outputs:
            G:          A list containing the four one-way components of the Green's matrix associsated to a source at zS and a receiver at zR: G = [GPP13,GPM13,GMP13,GMM13].
            initials:   A list containing current source & receiver depth levels as well as the responses of the medium above zS/zR, between zS/zR and below zS/zR: initials = [zR,zS,RP1,RM1,TP1,TM1,RP2,RM2,TP2,TM2,RP3,RM3,TP3,TM3].
            
        """
        
        print('Modelling Greens functions between zS and zR for p = %.2f*1e-3 ...'%(p*1e3))
        
        # Insert layers
        self.Insert_layer(zS)
        self.Insert_layer(zR)
        
        if initials != []:
            zRold = initials[0]
            zSold = initials[1]
            self.Insert_layer(zRold)
            NRold = np.cumsum(self.dzvec).tolist().index(zRold)
        
        # To avoid indexing errors when z = model depth
        remove = 0
        if (zS == np.cumsum(self.dzvec)[-1]) or (zR == np.cumsum(self.dzvec)[-1]):
            self.Insert_layer(np.cumsum(self.dzvec)[-1] + 1)
            remove = 1
        
        if eps is None:
            eps = 3/(self.nf*self.dt)
        
        # Frequency vector
        Wfft = self.Wvec()[1]
    
        # Extract positive frequencies
        Wpos  = Wfft[0:self.nf,0] + 1j*eps
        
        # Propagation matrix of an infinitesimal layer without any contrast
        W = np.zeros((self.nf,4),dtype=complex)
        I = np.zeros((self.nf,4),dtype=complex)
        I[:,0] = 1
        I[:,3] = 1
        
        # Layer 
        NS = np.cumsum(self.dzvec).tolist().index(zS)
        NR = np.cumsum(self.dzvec).tolist().index(zR)
        
        # Number of layers
        NN = np.size(self.cpvec)
        
        # (1) Medium above the source depth level
        
        # Initial fields
        RP1 = np.zeros((self.nf,4),dtype=complex)
        RM1 = np.zeros((self.nf,4),dtype=complex)
        # Here every frequency and every wavenumber component have an amplitude 
        # equal to one. Hence, the total wavefield has a strength of sqrt(nt*nr)
        # When an inverse fft (ifft2) is applied the wavefield is scaled by 
        # 1/sqrt(nt*nr) hence in the time domain the wavefield has the an
        # amplitude equal to one.
        TP1 = I.copy()
        TM1 = I.copy()
        
#        if self.csvec[0] == 0:
#            TP1[:,:,3] = 0
#            TM1[:,:,3] = 0
            
        # (2) Medium below the source/receiver depth level
            
        # Middle part of the medium
        RP2  = np.zeros((self.nf,4),dtype=complex)
        RM2  = np.zeros((self.nf,4),dtype=complex)
        TP2 = I.copy()
        TM2 = I.copy()
        
#        if self.csvec[NS+1] == 0:
#            TP2[:,:,3] = 0
#            TM2[:,:,3] = 0
            
        # (3) Medium below the receiver depth level
            
        # Bottom part of the medium
        RP3  = np.zeros((self.nf,4),dtype=complex)
        RM3  = np.zeros((self.nf,4),dtype=complex)
        TP3 = I.copy()
        TM3 = I.copy()
        
#        if self.csvec[NR+1] == 0:
#            TP3[:,:,3] = 0
#            TM3[:,:,3] = 0
            
        # First layer of the recursion    
        Nstart = 0
            
        # Variable to check if all data needs to be modelled (Set to check=1 unless previous modelling data are passed to the function)
        check = 1
        
        def PropScat(RP,RM,TP,TM,W,I,Nstart,Nend,Wpos,p,mul,conv):
            """
            Propagate through a homogeneous layer and scatter through the bottom interface.
            
            Input:
                RP:     Initial reflection response from above in F-Kx domain.
                RM:     Initial reflection response from below in F-Kx domain.
                TP:     Initial Transmission response from above in F-Kx domain.
                TM:     Initial Transmission response from below in F-Kx domain.
                W:      Initial wavefield propagation matrix.
                I:      Identity matrix.
                Nstart: Index of 1st layer. Starts at top of the 1st layer.
                Nend:   Index of last layer. Ends at top of the last layer.
                Wpos:   Meshgrid of positive frequencies (and wavenumbers).
                p:      Ray-parameter (single value).
                mul:    Set mul=1 to include internal multiples.
                conv:   Set conv=1 to include mode conversions.
                
            Output:
                RP:     Reflection response from above of the layer stack between Nstart and Nend with reflection-free boundaries.
                RM:     Reflection response from below of the layer stack between Nstart and Nend with reflection-free boundaries.
                TP:     Transmission response from above of the layer stack between Nstart and Nend with reflection-free boundaries.
                TM:     Transmission response from below of the layer stack between Nstart and Nend with reflection-free boundaries.
            """
            # Start the first (downward) recursion loop over upper N-1 layers
            for n in range(Nstart,Nend):
            
                dz1 = self.dzvec[n]
                
                # Parameters of top layer
                cp1 = self.cpvec[n]
                cs1 = self.csvec[n]
                ro1 = self.rovec[n]
            
                # Parameters of bottom layer
                cp2 = self.cpvec[n+1]
                cs2 = self.csvec[n+1]
                ro2 = self.rovec[n+1]
                
                # Propagator through top layer
                qp = (1/cp1**2 - p**2)**0.5 # kz/w for p-waves
                qs = (1/cs1**2 - p**2)**0.5 # kz/w for s-waves
                
                W[:,0] = np.exp(1j*Wpos*qp*dz1)
#                if cs1 != 0:
                W[:,3] = np.exp(1j*Wpos*qs*dz1)   # for elastic layer
#                else: 
#                    W[:,:,3] = 0                    # for purely acoustic layer
            
                rP,tP,rM,tM = self.RT_p_w(cp1,cs1,ro1,cp2,cs2,ro2,p,conv)[0:4]
        
                if mul == 1:
                    tmp = I - self.Mul_My_dot(RM,W,rP,W)
                    # Inverse of tmp
                    M1 = self.My_inv(tmp)
                    
                    tmp = I - self.Mul_My_dot(rP,W,RM,W)
                    # Inverse of tmp
                    M2 = self.My_inv(tmp)
                    
                else:
                    M1 = I
                    M2 = I
                    
                # Update reflection / transmission responses
                RP = RP + self.Mul_My_dot(TM,W,rP,W,M1,TP)
                RM = rM + self.Mul_My_dot(tP,W,RM,W,M2,tM)
                TP = self.Mul_My_dot(tP,W,M1,TP)
                TM = self.Mul_My_dot(TM,W,M2,tM) 
                
            return RP,RM,TP,TM
        
        # It is necessary to distinguish the following three cases:
        # zS < zR
        # zS = zR
        # zS > zR
        if zS < zR:
            
            if initials != []:
                if (zSold == zS):
                    if zSold < zRold:
                        
                        if zRold < zR:
                        
                            # (1) Medium above the source depth level: Reuse previous responses
                            RP1,RM1,TP1,TM1 = initials[2:6]
                            
                            # (2) Medium between the source and receiver depth levels: Update previous responses
                            RP2,RM2,TP2,TM2 = initials[6:10]
                            
                            RP2,RM2,TP2,TM2 = PropScat(RP2,RM2,TP2,TM2,W,I,NRold+1,NR+1,Wpos,p,mul,conv)
                            
                            
                            # (3) Medium below the receiver depth level: Compute response
                            RP3,RM3,TP3,TM3 = PropScat(RP3,RM3,TP3,TM3,W,I,NR+1,NN-1,Wpos,p,mul,conv) 
                            
                            print("You saved some time by reusing previous computation results.")
                            
                            # Set check=0 to avoid remodelling.
                            check = 0
            
            if check == 1:
                
                # (1) Medium above the source depth level
                
                # Compute medium responses
                RP1,RM1,TP1,TM1 = PropScat(RP1,RM1,TP1,TM1,W,I,Nstart,NS+1,Wpos,p,mul,conv)
                
                # (2) Medium between the source and receiver depth levels
                    
                # Compute medium responses
                RP2,RM2,TP2,TM2 = PropScat(RP2,RM2,TP2,TM2,W,I,NS+1,NR+1,Wpos,p,mul,conv)    
                
                # (3) Medium below the receiver depth level
                
                # Compute medium responses
                RP3,RM3,TP3,TM3 = PropScat(RP3,RM3,TP3,TM3,W,I,NR+1,NN-1,Wpos,p,mul,conv)   
                
            # Compute the Green's functions G12 that exclude the medium below the receiver: GPP12,GPM12
            if mul == 1:
            
                # To bottom
                M = self.My_inv( I - self.My_dot(RM1,RP2) )
                GPP12 =	self.My_dot(TP2,M)                   
                GPM12 =	-self.Mul_My_dot(TP2,M,RM1)       # Multiply by -1 because upgoing sources are defined with negative amplitude
                
            elif mul == 0:
                
                # To bottom
                GPP12 =	TP2#self.My_dot(TP2)                   
                GPM12 =	-self.Mul_My_dot(TP2,RM1)         # Multiply by -1 because upgoing sources are defined with negative amplitude

            # Compute reflection from below of part 1+2 
            M = self.My_inv( I - self.My_dot(RP2,RM1) )
            RM12 = RM2 + self.Mul_My_dot(TP2,RM1,M,TM2)
            
            # Compute the Green's functions G13 for the complete medium: GPP13,GPM13,GMP13,GMM13
            M = self.My_inv( I - self.My_dot(RM12,RP3) )
            GPP13 = self.Mul_My_dot(M,GPP12)
            GPM13 = self.Mul_My_dot(M,GPM12)
            GMP13 = self.Mul_My_dot(RP3,M,GPP12)
            GMM13 = self.Mul_My_dot(RP3,M,GPM12)
            
        elif zS == zR:
            # In this case zS is exactly at zR. If zS and zR have an infinitesimal depth difference dz we need to correct GPP or GMM by subtracting an identity I.
            
            if initials != []:
                if (zSold == zS):
                    if zSold > zRold:
                        
                        # (1) Medium above the source depth level: Reuse previous responses
                        RP1,RM1,TP1,TM1 = PropScat(RP1,RM1,TP1,TM1,W,I,Nstart,NS+1,Wpos,p,mul,conv)
                        
                        # (2) Medium below the source and receiver depth levels: Reuse previous responses
                        RP2,RM2,TP2,TM2 = initials[10:14]
                        
                        # (3) Medium below the receiver depth level: Compute response
                        RP3,RM3,TP3,TM3 = initials[10:14]
                        
                        print("You saved some time by reusing previous computation results.")
                        
                        # Set check=0 to avoid remodelling.
                        check = 0
            
            if check == 1:
                
                # (1) Medium above the source depth level: Reuse previous responses
                
                # Compute medium responses
                RP1,RM1,TP1,TM1 = PropScat(RP1,RM1,TP1,TM1,W,I,Nstart,NS+1,Wpos,p,mul,conv)
                
                # (2) Medium below the source/receiver depth level
                    
                # Compute medium responses
                RP2,RM2,TP2,TM2 = PropScat(RP2,RM2,TP2,TM2,W,I,NS+1,NN-1,Wpos,p,mul,conv)
            
            
            # Compute the Green's functions G13 for the complete medium: GPP13,GPM13,GMP13,GMM13
            M = self.My_inv( I - self.My_dot(RM1,RP2) )
            GPP13 = M.copy()
            GPM13 = -self.Mul_My_dot(M,RM1)                  # Multiply by -1 because upgoing sources are defined with negative amplitude
            GMP13 = self.Mul_My_dot(RP2,M)
            GMM13 = -self.My_inv( I - self.My_dot(RP2,RM1) ) # Multiply by -1 because upgoing sources are defined with negative amplitude
            
        elif zS > zR:
            
            if initials != []:
                if (zSold == zS):
                    if zSold > zRold:
                        
                        if zRold < zR:
                        
                            # (1) Medium above the source depth level: Reuse previous responses
                            RP1,RM1,TP1,TM1 = initials[2:6]
                            RP1,RM1,TP1,TM1 = PropScat(RP1,RM1,TP1,TM1,W,I,NRold+1,NR+1,Wpos,p,mul,conv)
                            
                            # (2) Medium between the source and receiver depth levels: Update previous responses
                            RP2,RM2,TP2,TM2 = PropScat(RP2,RM2,TP2,TM2,W,I,NR+1,NS+1,Wpos,p,mul,conv)
                            
                            # (3) Medium below the receiver depth level: Compute response
                            RP3,RM3,TP3,TM3 = initials[10:14] 
                            
                            print("You saved some time by reusing previous computation results.")
                            
                            # Set check=0 to avoid remodelling.
                            check = 0
            
            if check == 1:
                
                # (1) Medium above the source depth level: Reuse previous responses
                
                # Compute medium responses
                RP1,RM1,TP1,TM1 = PropScat(RP1,RM1,TP1,TM1,W,I,Nstart,NR+1,Wpos,p,mul,conv)
                
                # (2) Medium between the receiver and source depth levels
                    
                # Compute medium responses
                RP2,RM2,TP2,TM2 = PropScat(RP2,RM2,TP2,TM2,W,I,NR+1,NS+1,Wpos,p,mul,conv)    
                
                # (3) Medium below the source depth level
                    
                # Compute medium responses
                RP3,RM3,TP3,TM3 = PropScat(RP3,RM3,TP3,TM3,W,I,NS+1,NN-1,Wpos,p,mul,conv)   
                
            
            # Compute the Green's functions G23 that exclude the medium above the receiver: GMP23,GMM23
            if mul == 1:
            
                # To top
                M = self.My_inv( I - self.My_dot(RP3,RM2) )
                GMP23 =	self.Mul_My_dot(TM2,M,RP3)                   
                GMM23 =	-self.Mul_My_dot(TM2,M)       # Multiply by -1 because upgoing sources are defined with negative amplitude
                
            elif mul == 0:
                
                # To top
                GMP23 =	self.My_dot(TM2,RP3)                   
                GMM23 =	-TM2                          # Multiply by -1 because upgoing sources are defined with negative amplitude

            # Compute reflection from above of part 2+3 
            M = self.My_inv( I - self.My_dot(RM2,RP3) )
            RP23 = RP2 + self.Mul_My_dot(TM2,RP3,M,TP2)
            
            # Compute the Green's functions G13 for the complete medium: GPP13,GPM13,GMP13,GMM13
            M = self.My_inv( I - self.My_dot(RP23,RM1) )
            GPP13 = self.Mul_My_dot(RM1,M,GMP23)
            GPM13 = self.Mul_My_dot(RM1,M,GMM23)
            GMP13 = self.Mul_My_dot(M,GMP23)
            GMM13 = self.Mul_My_dot(M,GMM23)

        # Remove layers
        self.Remove_layer(zS)
        if zR != zS:
            self.Remove_layer(zR)
        if initials != [] and zRold != zS:
            self.Remove_layer(zRold)
            
        # To avoid indexing errors when z = model depth
        if remove == 1:
            self.Remove_layer(np.cumsum(self.dzvec)[-1])
        
        # Delete NaN's and limit inf's
        GPP13 = np.nan_to_num(GPP13)
        GPM13 = np.nan_to_num(GPM13)
        GMP13 = np.nan_to_num(GMP13)
        GMM13 = np.nan_to_num(GMM13)
        
        # The highest negative frequency and highest negative wavenumber components are real-valued
        GPP13[self.nf-1,:] = GPP13[self.nf-1,:].real
        GPM13[self.nf-1,:] = GPM13[self.nf-1,:].real
        GMP13[self.nf-1,:] = GMP13[self.nf-1,:].real
        GMM13[self.nf-1,:] = GMM13[self.nf-1,:].real
        
        # Conjugate wavefields
        GPP13 = GPP13.conj()
        GPM13 = GPM13.conj()
        GMP13 = GMP13.conj()
        GMM13 = GMM13.conj()
        
        GPP13,GPM13,GMP13,GMM13 = self.Sort_w(GPP13,GPM13,GMP13,GMM13)
    
        G = [GPP13,GPM13,GMP13,GMM13]
        initials = [zR,zS,RP1,RM1,TP1,TM1,RP2,RM2,TP2,TM2,RP3,RM3,TP3,TM3]
        
        return G,initials
    
    
    
    
    
    # Insert a layer in the model    
    def Insert_layer(self,z0):
        """
        Insert_layer(z0)
        
        Insert a depth level if it does not exist yet.
        
        Input:
            z0: Depth
        """
        
        # Depth vector
        z = np.cumsum(self.dzvec)
        
        # Vector of depths smaller or equal to z0
        L = z[z<=z0] 
        
        # Case1: z0 smaller than self.dzvec[0]
        if L.size == 0:
            dzvec = np.array(self.dzvec)
            self.dzvec = np.hstack((z0,dzvec[0]-z0,dzvec[1:])).tolist()
            
            cpvec = np.array(self.cpvec)
            self.cpvec = np.hstack((cpvec[0],cpvec)).tolist()
            csvec = np.array(self.csvec)
            self.csvec = np.hstack((csvec[0],csvec)).tolist()
            rovec = np.array(self.rovec)
            self.rovec = np.hstack((rovec[0],rovec)).tolist()
            return
        
        # Case2: z0 coincides with an element of z = np.cumsum(self.dzvec)
        elif L[-1] == z0:
            return
        
        # Case 3: z0 is larger than z[-1] = = np.cumsum(self.dzvec)[-1]
        elif L.size == z.size:
            dzvec = np.array(self.dzvec)
            self.dzvec = np.hstack((dzvec,z0-z[-1])).tolist()
            
            cpvec = np.array(self.cpvec)
            self.cpvec = np.hstack((cpvec,cpvec[-1])).tolist()
            csvec = np.array(self.csvec)
            self.csvec = np.hstack((csvec,csvec[-1])).tolist()
            rovec = np.array(self.rovec)
            self.rovec = np.hstack((rovec,rovec[-1])).tolist()
            return
            
        # Case 4: z0 is between z[0] and z[-1] AND does not coincide with any element of z
        
        b = L[-1] 
        ind = z.tolist().index(b)
        
        dzvec = np.array(self.dzvec)
        self.dzvec = np.hstack((dzvec[:ind+1],z0-b,z[ind+1]-z0,dzvec[ind+2:])).tolist()
        
        # Parameters
        cpvec = np.array(self.cpvec)
        self.cpvec = np.hstack((cpvec[:ind+1],cpvec[ind+1],cpvec[ind+1],cpvec[ind+2:])).tolist()
        csvec = np.array(self.csvec)
        self.csvec = np.hstack((csvec[:ind+1],csvec[ind+1],csvec[ind+1],csvec[ind+2:])).tolist()
        rovec = np.array(self.rovec)
        self.rovec = np.hstack((rovec[:ind+1],rovec[ind+1],rovec[ind+1],rovec[ind+2:])).tolist()
        
        return
    
    # Remove a layer from the model
    def Remove_layer(self,z0):
        """
        Remove_layer(z0)
        
        Remove a depth level if it was introduced by Insert_layer.
        
        Input:
            z0: Depth
        """
        
        #if len(self.dzvec) == self.N:
        if z0 in self.zvec:
            return
        
        ind = np.cumsum(self.dzvec).tolist().index(z0) + 1
        
        if ind == len(self.dzvec):
            del self.dzvec[ind-1]
            del self.cpvec[ind-1]
            del self.csvec[ind-1]
            del self.rovec[ind-1]
            return
        
        self.dzvec[ind-1] = self.dzvec[ind-1] + self.dzvec[ind]
        del self.dzvec[ind]
        del self.cpvec[ind]
        del self.csvec[ind]
        del self.rovec[ind]
        
        return
    
    # Function to compose two-way wavefields
    def Compose(self,GPP,GPM,GMP,GMM,p,cp=None,cs=None,ro=None,z=None,eps=None,initials=[]):
        """
        To be tested! I have never used the w-p domain version of this code!
        
        G2,initials = Compose(self,GPP,GPM,GMP,GMM,cp=None,cs=None,ro=None,z=None,eps=None,initials=[])
        
        Compose() composes an elastodynamic two-way Green's tensor from a power-flux normalised one-way Green's tensor.
        The composition is done in the horizontal-wavenumber frequency domain. Hence, the input and output data are in the f-kx domain.
        Eqution: G2 = L G1 Linv
        
        Input:
            GPP:        One-way Green's matrix G^{+,+} for a single ray-parameter (nt x 4).
            GPM:        One-way Green's matrix G^{+,-} for a single ray-parameter (nt x 4).
            GMP:        One-way Green's matrix G^{-,+} for a single ray-parameter (nt x 4).
            GMM:        One-way Green's matrix G^{-,-} for a single ray-parameter (nt x 4).
            p:          Ray-parameter.
            cp:         P-wave velocity in layer of composition.
            cs:         S-wave velocity in layer of composition.
            ro:         Density in layer of composition.
            z:          Depth level of composition
            eps:        Imaginary constant added to temporal frequency omega.
            initials:
                
        Output:
            G2:         Two-way Green's matrix [ [G^{\tau,f} , G^{\tau,h}] , [G^{v,f} , G^{v,h}] ] for a single ray-parameter (nt x 4).
            initials:
        """
        
        if eps is None:
            eps = 3/(self.nf*self.dt)
        
        # Frequency vector
#        Wfft = self.Wvec()[1]
    
        # Extract positive frequencies including highest negative frequency value
#        Wpos  = Wfft[0:self.nf,0] + 1j*eps
        
        nt,_ = GPP.shape
        if nt == self.nt:
            GPP = GPP[:self.nf,:]
            GPM = GPM[:self.nf,:]
            GMP = GMP[:self.nf,:]
            GMM = GMM[:self.nf,:]
        
        if (cp is None) or (cs is None) or (ro is None):
            
            try:
                N = np.cumsum(self.dzvec).tolist().index(z)
                cp = self.cpvec[N+1]
                cs = self.csvec[N+1]
                ro = self.rovec[N+1]
            except:
                self.Insert_layer(z)    
                N = np.cumsum(self.dzvec).tolist().index(z)
                cp = self.cpvec[N]
                cs = self.csvec[N]
                ro = self.rovec[N]
                self.Remove_layer(z)
        
        check = 0
        if len(initials) == 11:
            if (cp == initials[0]) and (cs == initials[1]) and (ro == initials[2]):
                L1P = initials[3]
                L1M = initials[4]
                L2P = initials[5]
                L2M = initials[6]
                N1P = initials[7]
                N1M = initials[8]
                N2P = initials[9]
                N2M = initials[10]
                check = 1
            
        if check == 0:
            # Compute L block matrices
            L1P = self.L1P_kx_w(cp,cs,ro,p)
            L1M = self.L1M_kx_w(cp,cs,ro,p)
            L2P = self.L2P_kx_w(cp,cs,ro,p)
            L2M = self.L2M_kx_w(cp,cs,ro,p)
            
            # Compute N block matrices
            
            # N1P(+kx) = - L2M(-kx).T = - { -J L2M(+kx) J }.T = { J L2M(+kx) J }.T
            N1P = L2M.copy()       # Copy L2M
            N1P[:,1] = -L2M[:,2]   # Transpose and minus
            N1P[:,2] = -L2M[:,1]   # Transpose and minus
            
            # N1M(+p) = L2P(-p).T = { -J L2P(+kx) J }.T = -{ J L2P(+kx) J }.T
            N1M = -L2P.copy()          # Copy minus L2P
            N1M[:,1] = L2P[:,2]    # Transpose
            N1M[:,2] = L2P[:,1]    # Transpose
            
            # N2P(+kx) = L1M(-kx).T = { -J L1M(+kx) J }.T = -{ J L1M(+kx) J }.T
            N2P = -L1M.copy()            # Copy minus L1M
            N2P[:,1] =  L1M[:,2]     # Transpose 
            N2P[:,2] =  L1M[:,1]     # Transpose
            
            # N2M(+kx) = -L1P(-kx).T = -{ -J L1P(+kx) J }.T = { J L1P(+kx) J }.T
            N2M = L1P.copy()            # Copy L1P
            N2M[:,1] = -L1P[:,2]    # Transpose and minus
            N2M[:,2] = -L1P[:,1]    # Transpose and minus
        
        Gtf = self.Mul_My_dot(L1M,GMM,N1M) + self.Mul_My_dot(L1P,GPM,N1M) + self.Mul_My_dot(L1M,GMP,N1P) + self.Mul_My_dot(L1P,GPP,N1P)
        Gth = self.Mul_My_dot(L1M,GMM,N2M) + self.Mul_My_dot(L1P,GPM,N2M) + self.Mul_My_dot(L1M,GMP,N2P) + self.Mul_My_dot(L1P,GPP,N2P)
        Gvf = self.Mul_My_dot(L2M,GMM,N1M) + self.Mul_My_dot(L2P,GPM,N1M) + self.Mul_My_dot(L2M,GMP,N1P) + self.Mul_My_dot(L2P,GPP,N1P)
        Gvh = self.Mul_My_dot(L2M,GMM,N2M) + self.Mul_My_dot(L2P,GPM,N2M) + self.Mul_My_dot(L2M,GMP,N2P) + self.Mul_My_dot(L2P,GPP,N2P)
        
        # Is that solving the error of wmax?
        Gtf[self.nf-1,:,:] = Gtf[self.nf-1,:,:].real
        Gth[self.nf-1,:,:] = Gth[self.nf-1,:,:].real
        Gvf[self.nf-1,:,:] = Gvf[self.nf-1,:,:].real
        Gvh[self.nf-1,:,:] = Gvh[self.nf-1,:,:].real
        
        if (nt == self.nt):
            Gtf,Gth,Gvf,Gvh = self.Sort_w(Gtf,Gth,Gvf,Gvh)
            
        G2 = [Gtf,Gth,Gvf,Gvh]
        initials = [cp,cs,ro,L1P,L1M,L2P,L2M,N1P,N1M,N2P,N2M]
        
        return G2,initials
        
    def Multi_offset(self,field,p,t=0,w=0,eps=0):#,pdir=1):
        """
        I tested this function. I am not 100% sure whether the complex-valued 
        frequency eps is working correctly but results look correct to me. 
        I recommend to apply the Multi_offset() function in the frequncy domain
        in order to correctly attenuate the temporal wrap-around.
        
        I think I have to distinguish between up- and downgoing waves: For upgoing
        waves I should use -p to contruct the offset data.
        
        (Recommended setting t=0 and w=1)
        
        Inputs:
            field:  Wavefield (nt x 4) in w-p domain or in t-p domain.
            p:      Ray-parameter.
            t:      Set t=1 if field is in t-p domain.
            w:      Set w=1 if field is in w-p domain. 
            eps:    Constant that is multiplied by 1j and added to the frequency vector.
#            pdir:   Direction of propagation of the field. Set pdir=1 for downgoing waves.
                    Set pdir=-1 for upgoing waves. CHECK THIS WITH KEES AND GIOVANNI.
            
        Outputs:
            Fx:     Wavefield in t-x or w-x domain for a single ray-parameter (depending on input data).
        """
        N = field.shape[1]
        
#        p = pdir*p
        
        # Construct time-shift matrix S in the frequency domain
        X,Wfft = self.W_X_grid()[1:3]
        Wfft = Wfft - 1j*eps
        S = np.exp(1j*Wfft*p*X)
        S = np.tile(S.T,(N,1,1)).T
        
        if t == 1:
            
            # Transform to frequency domain
            if field.shape[0] == self.nf:
                field = self.Sort_w(field)
            field = np.fft.fft(field,n=None,axis=0)
            
        # Copy traces of field nr times: (nt x 4) -> (nt x nr x 4)
        F = np.tile(field,(self.nr,1,1)).swapaxes(0,1)
        
        # Multiply F with a phase shift matrix S elementwise
        Fx = F*S
                
        if w == 1:
            return Fx
        
        if t == 1:
            # Transform back to the time domain
            fx = np.fft.ifft(Fx,n=None,axis=0).real
            return fx
            
            
            