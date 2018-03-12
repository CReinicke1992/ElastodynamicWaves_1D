#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 13:27:28 2017

@author: christianreini
"""
from Wavefield_kx_w import Wavefield_kx_w
import numpy as np


# Responses of layered media are wavefields
# Hence they inherit all properties of wavefields
#class Layers_kx_w(Wavefield_kx_w.Wavefield_kx_w):
class Layers_kx_w(Wavefield_kx_w):
    """
    Layers
    
    Compute responses of layered media for multiple frequencies and offsets.
    
    Variables:
        nt:    Number of time/frequency samples
        dt:    Duration per time sample in seconds
        nr:    Number of space/wavenumber samples
        dx:    Distance per space samples in metres
        dzvec: List or array with the thickness of each layer
        cpvec: List or array with the P-wave veclocity in each layer
        csvec: List or array with the S-wave veclocity in each layer
        rovec: List or array with the density of each layer
        
    Data sorting: 
        nt x nr x 4
    """
    
    def __init__(self,nt,dt,nr,dx,dzvec,cpvec,csvec,rovec):
        Wavefield_kx_w.__init__(self,nt,dt,nr,dx)
        self.dzvec = dzvec
        self.cpvec = cpvec
        self.csvec = csvec
        self.rovec = rovec        
        
    def L1P_kx_w(self,cp,cs,ro,kx,om):
        """
        L1P = L1P_kx_w(self,cp,cs,ro,kx,om):
            
        Decomposition matrix L-one-plus for power-flux decomposition
        """
        kp  = om/cp
        ks  = om/cs
        kzp = (kp**2 - kx**2)**(1/4)
        kzs = (ks**2 - kx**2)**(1/4)
        fac = np.sqrt(ro*om/2)/ks**2
        L1P = np.zeros((2,2),dtype=complex)
        
        L1P[0,0] = 2*kx*kzp
        L1P[0,1] = -(ks**2 - 2*kx**2)/kzs
        L1P[1,0] = (ks**2 - 2*kx**2)/kzp
        L1P[1,1] = 2*kx*kzs
        L1P      = fac*L1P
        return L1P
    
    def L1M_kx_w(self,cp,cs,ro,kx,om):
        """
        L1M = L1M_kx_w(self,cp,cs,ro,kx,om):
            
        Decomposition matrix L-one-plus for power-flux decomposition
        """
        kp  = om/cp
        ks  = om/cs
        kzp = (kp**2 - kx**2)**(1/4)
        kzs = (ks**2 - kx**2)**(1/4)
        fac = np.sqrt(ro*om/2)/ks**2
        L1M = np.zeros((2,2),dtype=complex)
        
        L1M[0,0] = -2*kx*kzp
        L1M[0,1] = -(ks**2 - 2*kx**2)/kzs
        L1M[1,0] = (ks**2 - 2*kx**2)/kzp
        L1M[1,1] = -2*kx*kzs
        L1M      = fac*L1M
        return L1M
    
    def L2P_kx_w(self,cp,cs,ro,kx,om):
        """
        L2P = L2P_kx_w(self,cp,cs,ro,kx,om):
            
        Decomposition matrix L-two-plus for power-flux decomposition"
        """
        kp  = om/cp
        ks  = om/cs
        kzp = (kp**2 - kx**2)**(1/4)
        kzs = (ks**2 - kx**2)**(1/4)
        fac = 1/np.sqrt(2*ro*om)
        L2P = np.zeros((2,2),dtype=complex)
        
        L2P[0,0] = kx/kzp
        L2P[0,1] = -kzs
        L2P[1,0] = kzp
        L2P[1,1] = kx/kzs
        L2P      = fac*L2P
        return L2P
    
    def L2M_kx_w(self,cp,cs,ro,kx,om):
        """
        L2M = L2M_kx_w(self,cp,cs,ro,kx,om):
            
        Decomposition matrix L-two-minus for power-flux decomposition.
        """
        kp  = om/cp
        ks  = om/cs
        kzp = (kp**2 - kx**2)**(1/4)
        kzs = (ks**2 - kx**2)**(1/4)
        fac = 1/np.sqrt(2*ro*om)
        L2M = np.zeros((2,2),dtype=complex)
        
        L2M[0,0] = kx/kzp
        L2M[0,1] = kzs
        L2M[1,0] = -kzp
        L2M[1,1] = kx/kzs
        L2M      = fac*L2M
        return L2M
    
    def RT_kx_w(self,cp1,cs1,ro1,cp2,cs2,ro2,kx,om,conv):
        """
        Rplus,Tplus,Rmin,Tmin = RT_kx_w(self,cp1,cs1,ro1,cp2,cs2,ro2,kx,om,conv)
        
        Compute scattering matrices from above and below for a horizontal interface, and a single- wavenumber and frequency.
        """
        
        if cp1==cp2 and cs1==cs2 and ro1==ro2:
            Rplus = np.zeros((2,2))
            Rmin = np.zeros((2,2))
            Tplus = np.identity(2)
            Tmin = np.identity(2)
            return (Rplus,Tplus,Rmin,Tmin)
        
        # Pauli matrix
        J = np.array([[1,0],[0,-1]])
        
        # Compute L1 matrices
        #L1Pu = self.L1P_kx_w(cp1,cs1,ro1,kx,om)
        L1Pl = self.L1P_kx_w(cp2,cs2,ro2,kx,om)
        L1Mu = self.L1M_kx_w(cp1,cs1,ro1,kx,om)
        #L1Ml = self.L1M_kx_w(cp2,cs2,ro2,kx,om)
        
        # Compute L2 matrices
        #L2Pu = self.L2P_kx_w(cp1,cs1,ro1,kx,om)
        L2Pl = self.L2P_kx_w(cp2,cs2,ro2,kx,om)
        L2Mu = self.L2M_kx_w(cp1,cs1,ro1,kx,om)
        #L2Ml = self.L2M_kx_w(cp2,cs2,ro2,kx,om)
        
        # Compute N1 matrices
        #N1Pu = - self.L2M_kx_w(cp1,cs1,ro1,-kx,om).T
        # Here I use L2M_kx_w(-kx) = - J L2M_kx_w(+kx) J
        N1Pu = -(-J.dot(L2Mu).dot(J)).T
        N1Pl = - self.L2M_kx_w(cp2,cs2,ro2,-kx,om).T
        N1Mu =   self.L2P_kx_w(cp1,cs1,ro1,-kx,om).T
        #N1Ml =   L2P_kx_w(cp2,cs2,ro2,-kx,om).T
        
        # Compute N2 matrices
        #N2Pu =   self.L1M_kx_w(cp1,cs1,ro1,-kx,om).T
        # Here I use L1M_kx_w(-kx) = - J L1M_kx_w(+kx) J
        N2Pu = (-J.dot(L1Mu).dot(J)).T
        N2Pl =   self.L1M_kx_w(cp2,cs2,ro2,-kx,om).T
        N2Mu = - self.L1P_kx_w(cp1,cs1,ro1,-kx,om).T
        #N2Ml = - L1P_kx_w(cp2,cs2,ro2,-kx,om).T   
        
        # Compute scattering matrices
        Tplus = N1Pu.dot(L1Pl) + N2Pu.dot(L2Pl)
        # Inverse of Tplus
        Tplus = np.array([[Tplus[1,1],-Tplus[0,1]],[-Tplus[1,0],Tplus[0,0]]]) / (Tplus[0,0]*Tplus[1,1] - Tplus[0,1]*Tplus[1,0])
        Tmin = J.dot(Tplus.T).dot(J)
        Rplus = (N1Mu.dot(L1Pl) + N2Mu.dot(L2Pl)).dot(Tplus)
        Rmin = (N1Pl.dot(L1Mu) + N2Pl.dot(L2Mu)).dot(Tmin)
        
        if conv==0:
            Tplus[0,1]=0
            Tplus[1,0]=0
            Tmin[0,1]=0
            Tmin[1,0]=0
            Rplus[0,1]=0
            Rplus[1,0]=0
            Rmin[0,1]=0
            Rmin[1,0]=0
        
        return Rplus,Tplus,Rmin,Tmin
    
    def Layercel_kx_w(self,N,wvec,kx,mul,conv):
        """
        Compute reflection / transmission response for a single wavenumber and all positive frequencies as well as the highest negative frequency.
        
        Output:
            R: Reflection response (nf x 4), 1st element corresponds to zero frequency
            T: Transmission response (nf x 4), 1st element corresponds to zero frequency
        """
        
        # Frequency vector (all negative frequencies including zero)
        nf   = int(self.nt/2+1)
        
        # Preallocate reflection and transmission vectors (for speed)
        Rpp = np.zeros(nf,dtype=complex)
        Rps = np.zeros(nf,dtype=complex)
        Rsp = np.zeros(nf,dtype=complex)
        Rss = np.zeros(nf,dtype=complex)
        
        Tpp = np.zeros(nf,dtype=complex)
        Tps = np.zeros(nf,dtype=complex)
        Tsp = np.zeros(nf,dtype=complex)
        Tss = np.zeros(nf,dtype=complex)
    
        # Loop over all positive frequencies and the highest negative frequency
        for ww in range(0,nf):
            
            w =wvec[ww]
            
            # Propagation and scattering matrices of an infinitesimal layer without any contrast
            W = np.zeros((2,2),dtype=complex)
            RP = np.zeros((2,2),dtype=complex)
            RM = np.zeros((2,2),dtype=complex)
            I = np.eye(2,dtype=complex)
            M1 = I.copy()
            M2 = I.copy()
            # Here every frequency and every wavenumber component have an amplitude 
            # equal to one. Hence, the total wavefield has a strength of sqrt(nt*nr)
            # When an inverse fft (ifft2) is applied the wavefield is scaled by 
            # 1/sqrt(nt*nr) hence in the time domain the wavefield has the an
            # amplitude equal to one.
            TP = I.copy()
            TM = I.copy()
            
            # Loop over N-1 interfaces
            for n in range(0,N-1):
                
                dz1 = self.dzvec[n]
                cp1 = self.cpvec[n]
                cp2 = self.cpvec[n+1]
                cs1 = self.csvec[n]
                cs2 = self.csvec[n+1]
                ro1 = self.rovec[n]
                ro2 = self.rovec[n+1]
                kzp = np.sqrt(w**2/cp1**2-kx**2) # kz for p-waves
                kzs = np.sqrt(w**2/cs1**2-kx**2) # kz for s-waves
                
                W[0,0] = np.exp(1j*kzp*dz1)
                W[1,1] = np.exp(1j*kzs*dz1)
                
                rP,tP,rM,tM = self.RT_kx_w(cp1,cs1,ro1,cp2,cs2,ro2,kx,w,conv)
                
                if mul == 1:
                    tmp = I - RM.dot(W).dot(rP).dot(W)
                    # Inverse of tmp
                    M1 = np.array([[tmp[1,1],-tmp[0,1]],[-tmp[1,0],tmp[0,0]]]) / (tmp[0,0]*tmp[1,1] - tmp[0,1]*tmp[1,0])
                    
                    tmp = I - rP.dot(W).dot(RM).dot(W)
                    # Inverse of tmp
                    M2 = np.array([[tmp[1,1],-tmp[0,1]],[-tmp[1,0],tmp[0,0]]]) / (tmp[0,0]*tmp[1,1] - tmp[0,1]*tmp[1,0])
                    
                # Update reflection / transmission responses
                RP = RP + TM.dot(W).dot(rP).dot(W).dot(M1).dot(TP)
                RM = rM + tP.dot(W).dot(RM).dot(W).dot(M2).dot(tM)
                TP = tP.dot(W).dot(M1).dot(TP)
                TM = TM.dot(W).dot(M2).dot(tM)
            
            # Save reflection / transmission response of the frequency component ww
            Rpp[ww] = RP[0,0]
            Rps[ww] = RP[0,1]
            Rsp[ww] = RP[1,0]
            Rss[ww] = RP[1,1]
            
            Tpp[ww] = TP[0,0]
            Tps[ww] = TP[0,1]
            Tsp[ww] = TP[1,0]
            Tss[ww] = TP[1,1]
        
        # Save reflection / transmission response of all positive frequencies
        
        R = np.zeros((nf,4),dtype=complex)
        T = R.copy()
      
        R[:,0] = Rpp.T
        R[:,1] = Rps.T
        R[:,2] = Rsp.T
        R[:,3] = Rss.T
        
        T[:,0] = Tpp.T
        T[:,1] = Tps.T
        T[:,2] = Tsp.T
        T[:,3] = Tss.T
        
        # The highest negative frequency component is real-valued
        R[nf-1,:] = R[nf-1,:].real
        T[nf-1,:] = T[nf-1,:].real
        
        return R,T
    
    def Run_Layercel_kx_w(self,mul,conv,eps=None):
        """
        Rfull,Tfull,R,T = Run_Layercel_kx_w(self,mul,conv,eps)
        
        Compute the reflection and transmission response of a layered medium for multiple frequency and wavenumber components.
        
        Inputs:
            mul:  Set mul=1 to include internal multiples.
            conv: Set conv=1 to include P/S conversions.
            eps:  eps is a real-positive to avoid zero-divisions. I recommend to choose eps=3/tmax. If eps is unequal zero you have to apply a gain to the wavefield in the t-x domain.
        
        Outputs:
            Rfull: Reflection response (wavefield) with all frequencies and wavenumbers (nt x nr x 4)
            Tfull: Transmission response (wavefield) with all frequencies and wavenumbers (nt x nr x 4)
            R: Reflection response (wavefield) with only positive frequencies and positive wavenumbers (nf x nk x 4)
            T: Transmission response (wavefield) with only positive frequencies and positive wavenumbers (nf x nk x 4)
        """
            
        # Number of layers
        N = np.size(self.cpvec)
        
        # Number of positive frequency and positive wavenumber samples
        nk = int(self.nr/2) + 1
        nf = int(self.nt/2) + 1
        
        if eps is None:
            eps = 3/(nf*self.dt)
        
        # Frequency and wavenumber vectors
        wvec = self.Wvec()[1]
        kxvec = self.Kxvec()[1]
        
        # Extract positive frequencies and the highest negative frequency
        # Add an imaginary constant
        wvec = wvec[0:nf] + 1j*eps
        
        R     = np.zeros((nf,nk,4),dtype=complex)
        T     = R.copy()
    
        # Loop over all positive wavenumbers and the highest negative wavenumber
        for kk in range(0,nk):
#            if kk%20 == 0:
#                print("%.1f percent ..."%(kk/(nk)*100))
            kx = kxvec[kk]
            R[0:nf,kk,:],T[0:nf,kk,:] = self.Layercel_kx_w(N,wvec,kx,mul,conv)
        
        # The most negative wavenumber element is real-valued
        R[:,nk-1,:] = R[:,nk-1,:].real
        T[:,nk-1,:] = T[:,nk-1,:].real
        
        # Remove Nans and values above let's say 10
        R = np.nan_to_num(R)
        R[abs(R)>10] = 0
        T = np.nan_to_num(T)
        T[abs(T)>10] = 0
        
        # Conjugate wavefields
        R = R.conj()
        T = T.conj()
        
        # Write full reflection and transmission matrices
        Rfull,Tfull = self.Sort_kx_w(R,T)
        
        return Rfull,Tfull,R,T
    
    
    def Sort_kx_w(self,*args):
        """
        Out = Sort_kx_w(*args)
        
        *args are wavefield matrices that only contain positive w-kx elements.
        Sort_kx_w exploits symmetry of wavefields in layered media to contruct the negative wavenumbers.
        Sort_kx_w exploits causality to construct the negative frequency components of the reflection/transmission.
        
        Input: 
            *args: Wavefield matrix/matrices in w-kx-domain for zero-positive w-kx including most negative w-kx element (real-valued).
        
        Output: 
            Out: Wavefield matrix/matrices in w-kx-domain for positive and negative w-kx.
        """
        
        # Number of positive frequency and positive wavenumber samples
        nk = int(self.nr/2) + 1
        nf = int(self.nt/2) + 1

        Out = []

        for i in np.arange(0,len(args)):
            F = args[i]
            
            # Preallocate wavefield matrix
            Ffull = np.zeros((self.nt,self.nr,4),dtype=complex)
            
            # Copy all positive wavenumbers including most negative wavenumber
            Ffull[0:nf,0:nk,:] = F.copy()
            
            # Apply symmetry to get negative wavenumber elements: F(k) = J F(-k) J 
            Ffull[:,nk:,0] =  Ffull[:,nk-2:0:-1,0]  # Fpp(w,-k) =  Fpp(w,k)
            Ffull[:,nk:,1] = -Ffull[:,nk-2:0:-1,1]  # Fps(w,-k) = -Fps(w,k)  
            Ffull[:,nk:,2] = -Ffull[:,nk-2:0:-1,2]  # Fsp(w,-k) = -Fsp(w,k)
            Ffull[:,nk:,3] =  Ffull[:,nk-2:0:-1,3]  # Fss(w,-k) =  Fss(w,k)
            
            # Apply causality to get negative frequencies: F(-w,-k) = F(w,k)
            Ffull[nf:,:,:]  = Ffull[nf-2:0:-1,:,:]     # Copy positive w
            Ffull[nf:,1:,:] = Ffull[nf:,-1:0:-1,:]     # Mirror kx of negative w
            Ffull[nf:,:,:]  = Ffull[nf:,:,:].conj()    # Conjugate negative w
            
            Out.append(Ffull)
            
        if len(args) == 1:
            Out = Out[0]
        
        return Out