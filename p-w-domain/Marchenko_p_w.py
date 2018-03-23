#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:03:17 2017

@author: christianreini
"""

from Layers_p_w import Layers_p_w
import numpy as np
import matplotlib.pylab as plt
from scipy import optimize

# Focusing and Green's functions between a focal depth and 
# the surface for layered media with transparent top and 
# bottom boundaries.
# All properties of wavefields in layered media are inheritted
# form the class Layers_p_w
class Marchenko_p_w(Layers_p_w):  
    """
    Marchenko_p_w
    
    Compute focusing and Green's functions between a focal depth and the surface of layered media for multiple frequencies and offsets.
    
    Variables:
        nt:    Number of time/frequency samples
        dt:    Duration per time sample in seconds
        nr:    Number of space/wavenumber samples
        dx:    Distance per space sample in metres
        nf:     Number of time samples divided by 2 plus 1.
        nk:     Number of space samples divided by 2 plus 1.
        dzvec: List or array with the thickness of each layer
        cpvec: List or array with the P-wave veclocity in each layer
        csvec: List or array with the S-wave veclocity in each layer
        rovec: List or array with the density of each layer
        zF:    Focal depth
        
    Data sorting: 
        nt x 4
        
    Vectorised computation
    """
    
    # All properties of wavefields in layered media are inheritted from
    # Layers_p_w
    def __init__(self,nt,dt,nr,dx,dzvec,cpvec,csvec,rovec):
        Layers_p_w.__init__(self,nt,dt,nr,dx,dzvec,cpvec,csvec,rovec)
        # Focal depth
        self.zF = None
        self.p = None
        # Focusing functions for positive frequencies and wavenumbers including the highest negative frequency and wavenumber
        self.F1P = None
        self.F1M = None
        self.F1P_neps = None # Focusing funcion with negative eps
        self.F1M_neps = None # Focusing funcion with negative eps
        self.F1P0 = None
        
    # Truncate model
    def Truncate(self):
        """
        dzvec_tr,cpvec_tr,csvec_tr,rovec_tr = Truncate()
        
        Truncate the model just below the focal depth zF. Below zF the model is made reflection-free.
        """
        tr = np.cumsum(self.dzvec).tolist().index(self.zF)
        
        if tr == len(self.dzvec)-1:
            dzvec_tr = np.hstack([self.dzvec,10])
            cpvec_tr = np.hstack([self.cpvec,self.cpvec[-1]])
            csvec_tr = np.hstack([self.csvec,self.csvec[-1]])
            rovec_tr = np.hstack([self.rovec,self.rovec[-1]])
        
        else:
            dzvec_tr = np.hstack([self.dzvec[:tr+1],10])
            cpvec_tr = np.hstack([self.cpvec[:tr+1],self.cpvec[tr+1]])
            csvec_tr = np.hstack([self.csvec[:tr+1],self.csvec[tr+1]])
            rovec_tr = np.hstack([self.rovec[:tr+1],self.rovec[tr+1]])
        
        Trunc = Layers_p_w(nt=self.nt,dt=self.dt,nr=self.nr,dx=self.dx,dzvec=dzvec_tr,cpvec=cpvec_tr,csvec=csvec_tr,rovec=rovec_tr)
        
        return dzvec_tr,cpvec_tr,csvec_tr,rovec_tr,Trunc
    
    
    # Compute focusing functions
    def FocusFunc1_inv(self,p,zF,mul=1,conv=1,eps=None,neps=0):
        """
        F1Pfull,F1Mfull,RPfull,TPfull,F1P,F1M = FocusFunc1_inv(p,zF,mul=1,conv=1,eps=None)
        
        Reflection and transmission response of the truncated medium are modelled. The responses are corrected for the complex-valued frequency. Then the focusing functions are computed.
        
        Inputs:
            p:        Ray-parameter.
            zF:       Set a focal depth.
            mul:      Set mul=1 to include internal multiples.
            conv:     Set conv=1 to include P/S conversions.
            eps:      Set eps to add an imaginary constant to the frequency: w -> w + 1j*eps. This reduces the wrap-around in time but one has to correct for this by multiplying the data by exp(eps*t) in the time domain.
            neps:     Set neps=1 to compute the focusing functions also with negative eps
            
        Outputs:
            F1P:      Downgoing focusing function for all frequencies and a single ray-parameter. (nt,4)
            F1M:      Upgoing focusing function for all frequencies and a single ray-parameter. (nt,4)
            RP_tr:    Reflection response of truncated medium  for all frequencies and a single ray-parameter. (nt,4). (no correction for complex-valued frequency applied)
            TP_tr:    Transmission response of truncated medium  for all frequencies and a single ray-parameter. (nt,4). (no correction for complex-valued frequency applied, here eps is negative!)
            self.F1P: Downgoing focusing function for all positive and the highest negative frequencies. (nt,4)
            self.F1M: Upgoing focusing function for all positive and the highest negative frequencies. (nt,4)
            
        Question:
            Is it possible to invert TP without correcting for the complex-valued frequency, and to corrct F1P for the complex-valued frequency?
            
        """
        
        # Check if focusing depth has changed
        if (zF != self.zF) or (p != self.p):
            self.zF = zF
            self.p = p
            self.F1P = None
            self.F1M = None
            self.F1P_neps = None # Focusing funcion with negative eps
            self.F1M_neps = None # Focusing funcion with negative eps
            self.F1P0 = None        
        
        # Insert layer
        self.Insert_layer(self.zF)
        
        # Constant imaginary
        if eps is None:
            eps = 3/(self.nf*self.dt)
        
        if self.F1P is None:
            
            print('Computing focusing functions for $p = %.2f *1e-3$ ...'%(p*1e3))
            
            # Truncate medium    
            Trunc = self.Truncate()[-1]
            
            # Compute reflection/transmission response of truncated medium for all positive frequencies and wavenumbers
            RP,TP = Trunc.Layercel_p_w(p=p,mul=mul,conv=conv,eps=eps,sort=0)[0:2]
            
            # Compute F1plus
            self.F1P = self.My_inv(TP)
            
            # Compute F1minus
            self.F1M = self.My_dot(RP,self.F1P)
            
            # Verbose: Remove NaNs and Infs
            if self.verbose == 1:
                
                if np.isnan(self.F1P).any() or np.isnan(self.F1M).any() or np.isinf(self.F1P).any() or np.isinf(self.F1M).any():
                    print('\n')
                    print('FocusFunc1_inv:')
                    print('\n'+100*'-'+'\n')
                    print('At least one of the modelled wavefields contains a NaN (Not a Number) or an Inf (infinite) element. '+
                          'In this step, NaN is replaced by zero, and infinity (-infinity) is replaced by the largest '+
                          '(smallest or most negative) floating point value that fits in the output dtype. Also see '+
                          'numpy.nan_to_num (in numpy or scipy documentation).')
                    print('\n')
                    
                    if np.isnan(self.F1P).any():
                        print('\t - F1P contains '+np.count_nonzero(np.isnan(self.F1P))+' NaN.')
                    if np.isinf(self.F1P).any():
                        print('\t - F1P contains '+np.count_nonzero(np.isinf(self.F1P))+' Inf.')
                    if np.isnan(self.F1M).any():
                        print('\t - F1M contains '+np.count_nonzero(np.isnan(self.F1M))+' NaN.')
                    if np.isinf(self.F1M).any():
                        print('\t - F1M contains '+np.count_nonzero(np.isinf(self.F1M))+' Inf.')
                
                    print('\n')
            
            self.F1P = np.nan_to_num(self.F1P)
            self.F1M = np.nan_to_num(self.F1M)
            
            # Construct negative frequencies/wavenumbers
            F1Pfull,F1Mfull = self.Sort_w(self.F1P,self.F1M)
            
        else:
            print('Using previously computed focusing functions for $p = %.2f *1e-3$ ...'%(p*1e3))
        
        # Construct negative frequencies/wavenumbers
        F1Pfull,F1Mfull = self.Sort_w(self.F1P,self.F1M)
        
        if (self.F1P_neps is None or p != self.p) and (neps == 1):
            
            print('Computing focusing functions with negative eps for $p = %.2f *1e-3$ ...'%(p*1e3))
            
            # Truncate medium    
            Trunc = self.Truncate()[-1]
            
            # Compute reflection/transmission response of truncated medium for all positive frequencies and wavenumbers
            RP,TP = Trunc.Layercel_p_w(p=p,mul=mul,conv=conv,eps=-eps,sort=0)[0:2]
            
            # Compute F1plus
            self.F1P_neps = self.My_inv(TP)
            
            # Compute F1minus
            self.F1M_neps = self.My_dot(RP,self.F1P_neps)
            
            # Verbose: Remove NaNs and Infs
            if self.verbose == 1:
                
                if np.isnan(self.F1P_neps).any() or np.isnan(self.F1M_neps).any() or np.isinf(self.F1P_neps).any() or np.isinf(self.F1M_neps).any():
                    print('\n')
                    print('FocusFunc1_inv:')
                    print('\n'+100*'-'+'\n')
                    print('At least one of the modelled wavefields contains a NaN (Not a Number) or an Inf (infinite) element. '+
                          'In this step, NaN is replaced by zero, and infinity (-infinity) is replaced by the largest '+
                          '(smallest or most negative) floating point value that fits in the output dtype. Also see '+
                          'numpy.nan_to_num (in numpy or scipy documentation).')
                    print('\n')
                    
                    if np.isnan(self.F1P_neps).any():
                        print('\t - F1P_neps contains '+np.count_nonzero(np.isnan(self.F1P_neps))+' NaN.')
                    if np.isinf(self.F1P_neps).any():
                        print('\t - F1P_neps contains '+np.count_nonzero(np.isinf(self.F1P_neps))+' Inf.')
                    if np.isnan(self.F1M_neps).any():
                        print('\t - F1M_neps contains '+np.count_nonzero(np.isnan(self.F1M_neps))+' NaN.')
                    if np.isinf(self.F1M_neps).any():
                        print('\t - F1M_neps contains '+np.count_nonzero(np.isinf(self.F1M_neps))+' Inf.')
                
                    print('\n')
            
            self.F1P_neps = np.nan_to_num(self.F1P_neps)
            self.F1M_neps = np.nan_to_num(self.F1M_neps)
            
        else:
            print('Using previously computed focusing functions with negative eps for $p = %.2f *1e-3$ ...'%(p*1e3))
            
        # Remove layer
        self.Remove_layer(self.zF)
            
        return F1Pfull,F1Mfull,self.F1P,self.F1M,self.F1P_neps,self.F1M_neps
        
    
    # Compute Green's functions
    def GreensFunc(self,p,zF,mul=1,conv=1,eps=None,RPfull=None,mod=0):
        """
        GMPfull,GMMfull = GreensFunc(p,zF,mul=1,conv=1,eps=None,taperlen=None,RPfull=None)
        
        The Green's functions are computed via the Marchenko equations. Hence, the focusing functions and the full-medium's reflection response are used.
        
         Inputs:
            p:        Ray-parameter.
            zF:       Set a focal depth.
            mul:      Set mul=1 to include internal multiples.
            conv:     Set conv=1 to include P/S conversions.
            eps:      Set eps to add an imaginary constant to the frequency: w -> w + 1j*eps. This reduces the wrap-around in time but one has to correct for this by multiplying the data by exp(eps*t) in the time domain.
            RPfull:   (Optional) Reflection response in the F-Kx domain with a complex-valued frequency w' = w + 1j*eps
            
        Outputs:
            GMPfull: Green's function G-minus-plus between the focal depth zF and the surface. Real-valued frequencies. All frequencies and wavenumbers (nt x nr x 4).
            GMMfull: Green's function G-minus-minus between the focal depth zF and the surface. Real-valued frequencies. All frequencies and wavenumbers (nt x nr x 4).
            GMP:     Green's function G-minus-plus between the focal depth zF and the surface. Real-valued frequencies. Positive and the highest negative frequencies / wavenumbers (nf x nk x 4).
            GMM:     Green's function G-minus-minus between the focal depth zF and the surface. Real-valued frequencies. Positive and the highest negative frequencies / wavenumbers (nf x nk x 4).
        """
        
        print('Computing Greens functions via Marchenko equations for $p = %.2f *1e-3$ ...'%(p*1e3))
        
        # Constant imaginary
        if eps is None:
            eps = 3/(self.nf*self.dt)
         
        # Model truncated transmission and invert it
        if mod == 0:
            # FocusFunc1_inv will only compute something IF the focal depth has changed
            # That means zF is different to self.zF  
            # Since self.F1P is self. the below function does not require any output
            self.FocusFunc1_inv(p,zF,mul,conv,eps,neps=1)
            
        # Model focusing function layer by layer
        else:
            # FocusFunc1_mod will always compute something even IF the focal depth has not changed
            self.F1P,self.F1M = self.FocusFunc1_mod(p,zF,mul,conv,eps,sort=0)[2:4]
            self.F1P_neps,self.F1M_neps = self.FocusFunc1_mod(p,zF,mul,conv,-eps,sort=0)[2:4]
        
        # Compute reflection/transmission response of  medium for positive frequencies and wavenumbers
        if RPfull is None:         
            RP = self.Layercel_p_w(p=p,mul=mul,conv=conv,eps=eps,sort=0)[0]
        
        # 1st Marchenko equation
        GMP = self.My_dot(RP,self.F1P) - self.F1M
        GMP = np.nan_to_num(GMP)
        
        # Compute R dagger
        RPdagger = self.My_T(RP).conj()
        
        # 2nd Marchenko equation
        GMM = (self.My_dot(RPdagger,self.F1M_neps) - self.F1P_neps).conj()
        GMM = np.nan_to_num(GMM)
        
        # Get negative frequencies / wavenumbers
        GMPfull,GMMfull = self.Sort_w(GMP,GMM)
        
        # In the 2nd Marchenko equation the Green's function GMM is associated to a negative ray-parameter
        # Hence the p-axis of GMM is reversed
        GMMfull = self.Reverse_p(GMMfull)
        
        return GMPfull,GMMfull
    
    # An attempt to model the focusing function
    def FocusFunc1_mod(self,p,zF=None,mul=1,conv=1,eps=None,sort=1,initials=[]):
        """
        RP,TP,F1P,F1M,initials = FocusFunc1_mod(p,zF=None,mul=1,conv=1,eps=None,sort=1,initials=[])
        Model the down- and upgoing focusing functions by wavefield extrapolation.
        
        Inputs:
            p:          Ray-parameter.
            zF:         Focusing depth
            mul:        Set mul=1 to model internal multiples.
            conv:       Set conv=1 to model P/S conversions.
            eps:        Set eps to add an imaginary constant to the frequency: w -> w + 1j*eps. This reduces the wrap-around in time but one has to correct for this by multiplying the data by exp(eps*t) in the time domain.
            sort:       Set sort=1 (default) to get positive and negative frequencies and wavenumbers.
            initials:   If medium responses of the overburden are known they can be passed via the list initials to avoid double-computation of medium responses.
        
        Output:
            RP:         Reflection response from above of the truncated medium (nf,4), 1st element corresponds to zero frequency.
            TP:         Transmission response from above of the truncated medium (nf,4), 1st element corresponds to zero frequency.
            F1P:        Downgoing focusing function (nf,4).
            F1M:        Upgoing focusing function (nf,4).
            initials:   List with medium responses of the truncated medium that can be used to compute the focusing functions of deeper focusing depths.
        """
        
        # Check if focusing depth has changed
        if (zF != self.zF) or (p != self.p):
            self.zF = zF
            self.p = p
            self.F1P = None
            self.F1M = None
            self.F1P_neps = None # Focusing funcion with negative eps
            self.F1M_neps = None # Focusing funcion with negative eps
            self.F1P0 = None
        
        # Insert layers
        self.Insert_layer(self.zF)
        if initials != []:
            self.Insert_layer(initials[-1])
        
        # Truncate medium
        Trunc = self.Truncate()[-1]
        
        # Number of layers
        N = np.size(Trunc.cpvec)
        
        if eps is None:
            eps = 3/(self.nf*self.dt)
        
        # Frequency vector
        Wfft = self.Wvec()[1]
        Wpos  = Wfft[0:self.nf,0] + 1j*eps
        
        # Progagation matrix of an infinitesimal layer without any contrast
        W = np.zeros((self.nf,4),dtype=complex)
        Winv = np.zeros((self.nf,4),dtype=complex)
        
        # Default multiple matrix (in case mul=0)
        I = np.zeros((self.nf,4),dtype=complex)
        I[:,0] = 1
        I[:,3] = 1
        M1 = I.copy()
        M2 = I.copy()
        
        
        if initials == []:
        # Scattering matrices of an infinitesimal layer without any contrast
            
            RP = np.zeros((self.nf,4),dtype=complex)
            RM = np.zeros((self.nf,4),dtype=complex)
            
            # Here every frequency and every wavenumber component have an amplitude 
            # equal to one. Hence, the total wavefield has a strength of sqrt(nt*nr)
            # When an inverse fft (ifft2) is applied the wavefield is scaled by 
            # 1/sqrt(nt*nr) hence in the time domain the wavefield has the an
            # amplitude equal to one.
            TP  = I.copy()
            TM  = I.copy()
            F1P = I.copy()
            Nstart = 0
            
#            if self.csvec[0] == 0:
#                TP[:,:,3]  = 0
#                TM[:,:,3]  = 0
#                F1P[:,:,3] = 0
            
        else:
            RP,RM,TP,TM,F1P,zstart = initials 
            Nstart = np.cumsum(Trunc.dzvec).tolist().index(zstart) + 1
            
        
        # Loop over N-1 interfaces
        for n in range(Nstart,N-1):
            
            dz1 = Trunc.dzvec[n]
            
            # Parameters of top layer
            cp1 = Trunc.cpvec[n]
            cs1 = Trunc.csvec[n]
            ro1 = Trunc.rovec[n]
        
            # Parameters of bottom layer
            cp2 = Trunc.cpvec[n+1]
            cs2 = Trunc.csvec[n+1]
            ro2 = Trunc.rovec[n+1]
            
            qp = (1/cp1**2 - p**2)**0.5 # kz/w for p-waves
            qs = (1/cs1**2 - p**2)**0.5 # kz/w for s-waves
            
            W[:,0] = np.exp(1j*Wpos*qp*dz1)
#            if cs1 != 0:
            W[:,3] = np.exp(1j*Wpos*qs*dz1)   # for elastic layer
#            else: 
#                W[:,:,3] = 0                    # for purely acoustic layer
            
            Winv = np.zeros((self.nf,4),dtype=complex)
            Winv[:,0] = 1/W[:,0]
#            if cs1 != 0:
            Winv[:,3] = 1/W[:,3]
#            else:
#                Winv[:,:,3] = 0 # This is mathematically not defined. But in a physical interpretation the S-wavefield cannot be inverse propagated in a purely acoustic layer.
            Winv = np.nan_to_num(Winv)
            
            
            rP,tP,rM,tM,tPinv = self.RT_p_w(cp1,cs1,ro1,cp2,cs2,ro2,p,conv)
    
            if mul == 1:
                M1inv = I - self.Mul_My_dot(RM,W,rP,W)
                # Inverse of tmp
                M1 = self.My_inv(M1inv)
                
                M2inv = I - self.Mul_My_dot(rP,W,RM,W)
                # Inverse of tmp
                M2 = self.My_inv(M2inv)
            else:
                M1inv = I.copy()
                M2inv = I.copy()
                M2    = I.copy()
                
            # Update reflection / transmission responses
            RP = RP + self.Mul_My_dot(TM,W,rP,W,M1,TP)
            RM = rM + self.Mul_My_dot(tP,W,RM,W,M2,tM)
            TP = self.Mul_My_dot(tP,W,M1,TP)
            TM = self.Mul_My_dot(TM,W,M2,tM)   
            F1P = self.Mul_My_dot(F1P,M1inv,Winv,tPinv)
        
        # At the end compute F1M
        F1M = self.My_dot(RP,F1P)
        
        # Remove layers again
        self.Remove_layer(self.zF)
        if initials != []:
            self.Remove_layer(initials[-1])
        
        # Save initial fields for next call of FocusFunc1_mod()
        initials = [RP,RM,TP,TM,F1P,zF]
    
        # The highest negative frequency component is real-valued
        RP[self.nf-1,:] = RP[self.nf-1,:].real
        TP[self.nf-1,:] = TP[self.nf-1,:].real
        F1P[self.nf-1,:] = F1P[self.nf-1,:].real
        F1M[self.nf-1,:] = F1M[self.nf-1,:].real
        
        
        # Remove Nans and values above let's say 10
        RP = np.nan_to_num(RP)
        TP = np.nan_to_num(TP)
        F1P = np.nan_to_num(F1P)
        F1M = np.nan_to_num(F1M)
        
        # Conjugate wavefields
        RP = RP.conj()
        TP = TP.conj()
        F1P = F1P.conj()
        F1M = F1M.conj()
        
        # Write full reflection and transmission matrices
        if sort == 1:
#            Rfull,Tfull = self.Sort_RT_kx_w(RP,TP)
            RPfull,TPfull,F1Pfull,F1Mfull = self.Sort_w(RP,TP,F1P,F1M)
            return RPfull,TPfull,F1Pfull,F1Mfull,initials
        
        return RP,TP,F1P,F1M,initials
    
    
    
    
    
    # Compute initial focusing function using the forward-scattered transmission
    def F1plus0(self,p,zF,eps=0):
        """
        In testing phase.
        self.F1P0,F1P0full = F1plus0(self,p,zF,eps)
        
        The initial focusing function (inverse forward-scattered transmission) is computed. No option for complex-valued frequencies.
        
         Inputs:
            p:        Ray-parameter.
            zF:       Set a focal depth.
            eps:      Choose complex-valued frequency. Default is eps=0, i.e. the frequency is real-valued.
           
        Outputs:
            F1P0:       Initial focusing function (inverse forward-scattered transmission excluding internal multiples) only positive frequencies and wavenumbers.
            F1P0full:   Initial focusing function (inverse forward-scattered transmission excluding internal multiples) all frequencies and wavenumbers.
            
        """
        
        # Check if focusing depth has changed
        if (zF != self.zF) or (p != self.p):
            self.zF = zF
            self.p = p
            self.F1P = None
            self.F1M = None
            self.F1P_neps = None # Focusing funcion with negative eps
            self.F1M_neps = None # Focusing funcion with negative eps
            self.F1P0 = None
        
        # Only compute F1P0 if it does not exist yet
        if self.F1P0 is None:
            
            # Insert layer
            self.Insert_layer(self.zF)
            
            # Truncate medium    
            Trunc = self.Truncate()[-1]
            
            # Compute transmission response of truncated medium excluding internal multiples for all positive frequencies and wavenumbers
            TP = Trunc.Layercel_p_w(p=p,mul=0,conv=1,eps=eps,sort=0)[1]
            
            # Invert the transmission response
            self.F1P0 = self.My_inv(TP)
            self.F1P0 = np.nan_to_num(self.F1P0)
            
            # Remove layer
            self.Remove_layer(self.zF)
            
        else:
            TP = self.My_inv(self.F1P0)
            TP = np.nan_to_num(TP)
    
        F1P0full,TPfull = self.Sort_w(self.F1P0,TP)
        
        return self.F1P0,F1P0full,TP,TPfull
    
    # Compute time window
    def Window(self,p,zF,manual=1,Tfs=1,eps=None,f0=None,Tdata=None,Wtaplen=None,plot=1,vmin=None,vmax=None,threshold=1e-3):
        """
        In testing phase.
        W = Window(self,p,zF,Tfs=1,eps=None,f0=None,Tdata=None,Wtaplen=None,plot=1,vmin=None,vmax=None)
        
        Build the window matrix for a focusing depth zF. The window can be based on the forward-scattered transmission (Tfs) or on the direct transmission (Td). This function builds the window by interpolating times picked by the user.
        
         Inputs:
            p:        Ray-parameter.
            zF:       Set a focal depth.
            manual:   Set manual=1 to pick the time window manually. Set manual=0 to pick the time window automatically.
            Tfs:      Set Tfs=1 to base the window on the forward scattered transmission. Set Tfs=0 to base the window on the direct transmission.  
            f0:       (Optional) Set central frequency (Ricker wavelet) in Hertz (f not omega). Setting f0 avoids ringiness and hence makes the picking easier.
            Tdata:    (Optional) Pass the forward-scatterd transmission response in the frequency-wavenumber domain [nt x nr (x 4)]. If you pass a transmission response use an eps factor eps=3/(nf*self.dt) to get the correct amplitude scaling. Otherwise the (output) transmission response's amplitudes will be scaled, however, the window matrix should not be affected by that.
            Plot:     Set plot=1 to automatically plot the transmission data, the window as well as the windowed transmission data.
            threshold: 1e-3. Set threshold value for automatic first break picking.
            
        Outputs:
            Out: Dictionary containing outputs
                    {
                    W:       Window matrix (nt x nr x 4)
                    TP:      Forward-scattered transmission response in the frequency-wavenumber domain [nt x nr (x 4)]. If a transmission response Tdata was passed the amplitudes might be scaled!
                    tP:      Forward-scattered transmission response in the space-time domain [nt x nr (x 4)]. If a transmission response Tdata was passed the amplitudes might be scaled!
                    }
        """
        
        # If forward-scatterd transmission is not given model it 
        if Tdata is None:
            # Check if focusing depth has changed
            if (zF != self.zF) or (p != self.p):
                self.zF = zF
                self.p = p
                self.F1P = None
                self.F1M = None
                self.F1P_neps = None # Focusing funcion with negative eps
                self.F1M_neps = None # Focusing funcion with negative eps
                self.F1P0 = None
                
            # Insert layer
            self.Insert_layer(self.zF)
            
            # Truncate medium    
            Trunc = self.Truncate()[-1]
            
            # Remove layer
            self.Remove_layer(self.zF)
            
            # Add complex constant to the frequency vector to attenuate temporal wrap-around effect
            if eps is None:
                eps = 3/(self.nf*self.dt)
            
            if Tfs == 1:
                
                # Compute transmission response of truncated medium excluding internal multiples for all positive frequencies and wavenumbers
                TP = Trunc.Layercel_p_w(p=p,mul=0,conv=1,eps=eps,sort=1)[1]
                    
            elif Tfs == 0:
                print('Not yet implemented.')
                return
        
        else:
            TP = Tdata.copy()

        # Add multiple offsets to make a 1.5D plot
        TPx = self.Multi_offset(TP,p,w=1,eps=eps)

        # Transform to space-time domain
        tP  = self.WP2TP(TP,eps=eps)
        tPx = self.WP2TP(TPx,eps=eps)
        
        # If a central frequency is given convolve with Ricker wavelet
        if f0 is not None:
            tP  = self.ConvolveRicker(tP,f0)[0]
            tPx = self.ConvolveRicker(tPx,f0)[0]
            
            # Width of a Ricker wavelet from minimum to maximum
            # Remove times two to get time between maximum and minimum instead of time between the two minima
            # epsilon will be used to account for the finite width of the signal
#            epsilon = int(2*np.sqrt(3/2)/(np.pi*f0) / self.dt)
            # I increase the signal with by replacing factor 2 by 2.5 to achieve better separation
            epsilon = int(2.5*np.sqrt(3/2)/(np.pi*f0) / self.dt)
        else: 
            epsilon = 1
            
        # Build taper the window
        if Wtaplen is not None:
            # User-defined taper length
            taplen = Wtaplen
        elif f0 is not None:
            # If no taper length is defined use half the width of the Ricker wavelet as taper length
            # (if a central frequency f0 is given)
            # Add times two to get time between the two minima instead of between maximum and minimum
            taplen = int(np.sqrt(3/2)/(np.pi*f0) / self.dt)
        else:
            # Last option: Use a taper of length 10 (samples)
            taplen = 10
            
        # Cosine taper    
        tap = np.cos(np.linspace(0,np.pi/2,taplen+1))
        
        # Determine if number of elastic components (should be 1 or 4)
        if TP.ndim == 2:
            N = TP.shape[1]
        elif TP.ndim == 3:
            N = TP.shape[2]
        
        # Determine clipping values for plotting
        if (vmin is None) or (vmax is None):
            vmax = 0.005*tP.max()
            vmin = -vmax
        
        # Pick window automatically based on a threshold amplitude
        if manual == 0:
                
            # Determine threshold value for 1st arrival detection
            Max = threshold*np.max(tP,0)
            
            # Preallocate window W 
            # Perallocate vector to save onset indices of the window (to construct the negative times of the window)
            W = np.zeros_like(tP)
            Onset = np.zeros(N)
            
            for comp in range(N):
                tt = 0
                var = True
                while var:
                    W[tt,comp] = 1
                    tt += 1
                    
                    # Detect 1st arrival
                    if tP[tt,comp] >= Max[comp]:
                        var = False
                        # Correct for width of the wavelet
                        W[tt-epsilon+1:tt+1,comp] = 0
                        # Add taper
                        W[tt-epsilon+1-taplen:tt-epsilon+2,comp] = tap
                        # Save onset index
                        Onset[comp] = tt-epsilon
                        shift       = tt-epsilon - (self.nf-1)
                        Onset[comp] = Onset[comp] - 2*shift
                        
            # Construct window for negative times
            if N == 4:
                # For negative times the PS and SP components of the window are interchanged
                tmp = Onset.copy()
                Onset[1] = tmp[2]
                Onset[2] = tmp[1]
        
            for comp in range(N):
                
                # Mute all time samples before the (negative) onset time of the window
                ind = int(Onset[comp])-1
                W[:ind,comp] = 0
                # Add taper
                try:
                    W[ind:ind+taplen+1,comp] = tap[::-1] 
                except:
                    W[:,comp] = 0
    
            # Extend window to multiple offsets
            # I don't use self.Multi_offset because this function operates in the frequency. As a
            # result the window amplitudes are not exactly 1 and 0.
            W1D = W.copy()
            W = np.zeros((self.nt,self.nr,N))
            X = self.Xvec()[0]
            for xx in range(self.nr):
                x = X[xx,0]
                ts = abs(int(p*x/self.dt))
                if p >= 0:
                    if xx >= self.nk-1:
                        W[:self.nt-ts,xx,:] = W1D[ts:,:]
                    else:
                        W[ts:,xx,:] = W1D[:self.nt-ts,:]
                else:
                    if xx <= self.nk-1:
                        W[:self.nt-ts,xx,:] = W1D[ts:,:]
                    else:
                        W[ts:,xx,:] = W1D[:self.nt-ts,:]
        
        # Pick window manually
        elif manual == 1:
            # Global variable for picks
            global xvals,yvals
            
            xvals = []
            yvals = []
            
            # Function
            def onclick(event):
                xvals.append(event.xdata)
                yvals.append(event.ydata)
                
            # Number of picks    
            Nc = 3#5
                
            # Iterate over elastic components to pick 1st arrivals    
            for comp in range(0,N):
                
                # User instruction
                title1 = "\n1. Zoom in using a single click. \n"
                title2 = "2. Pick %d onset times of the 1st arrival (excluding the 1st arrival). \n"%(Nc)
                title  = title1 + title2
                
                # Plot one elastic component
                if N == 1:
                    if (vmin is None) or (vmax is None):
                        vmax = 0.005*tP.max()
                        vmin = -vmax
                    fig = self.Plot(tPx,tx=1,tvec=1,xvec=1,title=title,vmin=vmin,vmax=vmax)
                else:
                    if (vmin is None) or (vmax is None):
                        vmax = 0.005*tP[:,comp].max()
                        vmin = -vmax
                    fig = self.Plot(tPx[:,:,comp],tx=1,tvec=1,xvec=1,title=title,vmin=vmin,vmax=vmax)
                
                # Connect figure to the function onlick 
                cid = fig.canvas.mpl_connect('button_press_event', onclick)
            
                # Wait until zoom and Nc picks were done
                while len(xvals) < 1 + Nc*(comp+1):
                    plt.pause(1)
                
                # Disconnect and close figure
                fig.canvas.mpl_disconnect(cid)
                plt.close(fig)
                
                # Delete the zoom point
                del xvals[comp*Nc]  
                del yvals[comp*Nc]    
                
            # Cut-off time one time sample before the pick
            epsilon = self.dt
                
            # Convert lists to arrays and shift picks by epsilon (to earlier times)    
            xvals = np.array(xvals)
            yvals = np.array(yvals) - epsilon
    
            # Reshape arrays
            xvals = xvals.reshape((N,Nc))
            yvals = yvals.reshape((N,Nc))
            
            # Define a fitting funcion: Linear moveout related to a plane wave with ray-parameter p
            def Linear_moveout(x,t0):
                return -self.p*x+t0
            
            # Initiate window
            W = np.zeros((self.nt,self.nr,N))
            
            # Create array to store the onset times of the window
            Onset = np.zeros((self.nr,N))
            
            # Iterate over elastic components to interpolate picks with hyperbola    
            for comp in range(0,N):
                
                # (1) Manually pick 1st arrival onset for near-offsets and interpolate with hyperbola.
                
                # Find hyperpolic fitting parameters
                pars = optimize.curve_fit(Linear_moveout,xvals[comp,:],yvals[comp,:])[0]# I changed [comp,:] to [:comp]
                
                X = self.Xvec()[0]
                t = Linear_moveout(X,pars[0])
                tind = t/self.dt + int(self.nt/2)
                tind[tind>=self.nt-1] = self.nt-1
                tind = tind.astype(int)
                
                for xx in range(self.nr):
                    # Fill window matrix with ones  
                    ind = tind[xx,0]
                    W[:ind+1,xx,comp] = 1
                    
                    # Save Onset time
                    Onset[xx,comp] = tind[xx]
                
                    # Add taper
                    W[ind-taplen:ind+1,xx,comp] = tap
                
                shift         = tind[self.nk-1] - (self.nf-1)
                Onset[:,comp] = Onset[:,comp] - 2*shift
                
            Onset[Onset<0] = 0
            
            # I think the below was wrong, so I commented it: The window should mute GMM (not Tfs), hence, the window is symmetric in time.
#            if N == 4:
#                tmp = Onset.copy()
#                Onset[:,1] = tmp[:,2]
#                Onset[:,2] = tmp[:,1]
            
            for comp in range(N):
                for xx in range(self.nr):
                    # Fill window matrix with ones    
                    ind = int(Onset[xx,comp])
                    W[:ind,xx,comp] = 0
                
                    # Add taper
                    W[ind:ind+taplen+1,xx,comp] = tap[::-1] 
    
            # Delete global variables
            del xvals,yvals
            
        if plot == 1:
            self.Plot(W,tx=1,tvec=1,xvec=1,vmin=-1,vmax=1,title='Window matrix $\mathbf{W}$')
            self.Plot((1-W)*tPx,tx=1,tvec=1,xvec=1,vmin=vmin,vmax=vmax,title='$(\mathbf{I} - \mathbf{W}) \mathbf{T}_{fs}^+$')
            self.Plot(W*tPx,tx=1,tvec=1,xvec=1,vmin=vmin,vmax=vmax,title='$\mathbf{W} \mathbf{T}_{fs}^+$')
        
        W1D = W[:,self.nk-1,:]
        
        Out = {"W":W,"W1D":W1D,"TP":TP,"tP":tP}
        
        return Out
    
    def Marchenko_series(self,p,zF,R=None,F1P0=None,W=None,manual=0,f0=10,vmin=None,vmax=None,K=0,plot=1,Gref=0,mul=1,conv=1,eps=None,convergence=0):
        """
        Out = Marchenko_series(p,zF,R=None,F1P0=None,W=None,f0=10,vmin=None,vmax=None,K=0,plot=1,ComputeReferenceG=0,mul=1,conv=1)
        
        Evaluate K iterations of the Marchenko series. Input wavefieds can be given. However, it seems to be important to attenuate temporal 
        wrap-around effects to achieve convergence of the Marchenko series.
        
        Inputs:
            p:              Ray-parameter p.
            zF:             Focusing depth
            R:              (Optional) Reflection response in the F-P domain (nt,4) or (nf,4).
            F1P0:           (Optional) Initial focusing function in the F-P domain (nt,4) or (nf,4).
            W:              (Optional) Window for the Marchenko seris (nt x 4).
            manual:         Set manual=1 to pick time window manually. Else set manual=0
            f0:             Default f0=10. Frequency for Ricker wavelet in Hz.
            vmin:           (Optional) Minimum clipping amplitude of the forward-scattered transmission response to pick the time window.
            vmax:           (Optional) Maximum clipping amplitude of the forward-scattered transmission response to pick the time window.
            K:              Default K=0. Number of iterations of the Marchenko series.
            plot:           Default plot=1. Set plot=1 to automatically plot the outputs. Set plot=0 to suppress automatic plotting.
            Gref:           Default Gref=0. Set Gref=1 to model reference Green's functions.
            mul:            Default mul=1. Set mul=1 to include internal multiples. Set mul=0 to exclude internal multiples.
            conv:           Default conv=1. Set conv=1 to include P/S conversions. Set conv=0 to exclude P/S conversions.
            eps:            Constant that is multiplied by 1j and added to the frequency vector to attenuate temporal wrap-around effects. 
                            Default eps = 3 / (self.dt * self.nf)
            convergence:    Set convergence=1 to make a convergence plot.
        
        Output:
            Out: Dictionary containing relfection response, retrieved Green's functions, time window, 
            focusing functions (initial downgoing, downgoing, and upgoing), modelled reference Green's functions.
            {'R':R,
            'GMP':GMP,
            'GMM':GMM,
            'W':W,
            'F1P0':F1P0,
            'F1Pk':F1Pk,
            'F1Mk':F1Mk,
            'GMPref':GMPref,
            'GMMref':GMMref}
        
        """
        
        # Check if focusing depth has changed
        if (zF != self.zF) or (p != self.p):
            self.p = p
            self.zF = zF
            self.F1P = None
            self.F1M = None
            self.F1P_neps = None # Focusing funcion with negative eps
            self.F1M_neps = None # Focusing funcion with negative eps
            self.F1P0 = None
           
        # Constant to dampen the temporal wrap-around   
        if eps is None:
            eps = 3/(self.dt*self.nf)
        
        # Model reflection data if not given
        # I recommend to dampen the temporal wrap-around!
        if R is None:
            print('(2/6) Compute reflection data $\mathbf{R}$ for $p = %.2f*1e-3$ and $\epsilon = %.2f$.'%(p*1e3,eps))
            R = self.Layercel_p_w(p=p,mul=mul,conv=conv,eps=eps,sort=1)[0]
            r = self.WP2TP(R,eps=eps,norm='ortho',taperlen=30)
            R = np.fft.fft(np.fft.ifftshift(r,0),n=None,axis=0,norm='ortho')
        else:
            print('(2/6) Use reflection data $\mathbf{R}$ passed by the user.')
        
        # Select positive frquency elements and compute R dagger
        R = R[:self.nf,:]
        Rdag = self.My_T(R).conj()
          
        # Model F1P0 if not given
        if F1P0 is None:
            print('(3/6) Compute initial focusing function $\mathbf{F}_{1,0}^+$ for $p = %.2f*1e-3$ and $\epsilon = -%.2f$.'%(p*1e3,eps))
            F1P0,_,TP = self.F1plus0(p,zF,eps=-eps)[1:4]
            f1P0 = self.WP2TP(F1P0,eps=-eps,norm='ortho',taperlen=30)
            F1P0 = np.fft.fft(np.fft.ifftshift(f1P0,0),n=None,axis=0,norm='ortho')[:self.nf,:]
        
        else:
            print('(3/6) Use initial focusing function $\mathbf{F}_{1,0}^+$ passed by the user.')
            TP = None
            if F1P0.shape[0] == self.nt:
                F1P0 = F1P0[:self.nf,:]
        
        # Make window if not given
        if W is None:
            print('(4/6) Pick time window $\mathbf{W}$ for $p = %.2f*1e-3$.'%(p*1e3))
            Out = self.Window(p,zF,manual=manual,Tfs=1,f0=f0,Tdata=TP,plot=plot,eps=eps,vmin=vmin,vmax=vmax)
            W = Out['W1D']
            # W should mute GMM, i.e. compared to Tfs source and receiver are interchanged.
            W = self.My_T(W)
        else:
            print('(4/6) Use the time window $\mathbf{W}$ passed by the user.')
            
        # I think the window is symmetric in time (also for PS/SP). Hence, Wrev is not necessary    
        # Time-reverse time window: Take into account that the PS/SP components are not symmertic in time!
#        Wrev=W.copy()
#        Wrev[:,1]=W[:,2] # just checking 
#        Wrev[:,2]=W[:,1] # just checking 
#        Wrev[1:,:]=Wrev[1:,:][::-1,:] 
        
        
        # Compute initial upgoing focusing function
        F1M0 = self.My_dot(R,F1P0)
        f1M0 = W*self.WP2TP(self.Sort_w(F1M0),norm='ortho')
        F1M0 = np.fft.fft(np.fft.ifftshift(f1M0,0),n=None,axis=0,norm='ortho')[:self.nf,:]
        
        # Update focusing function
        F1Pk = F1P0.copy()
        F1Mk = F1M0.copy()
        
        if Gref == 1 or convergence == 1:          
            print('(5/6) Compute reference Greens function $\mathbf{G}_{ref}^{-,+}$ and $\mathbf{G}_{ref}^{-,-}$.')
            Gs            = self.Gz2bound(p,zF,mul=mul,conv=conv,eps=eps)[0]
            GMPref,GMMref = Gs[2:4]
            
            if convergence == 1:
                # 1st Marchenko equation
                GMP = self.Sort_w( self.My_dot(R,F1Pk) - F1Mk ) 
                
                # 2nd Marchenko equation
                GMM = self.Sort_w( (self.My_dot(Rdag,F1Mk) - F1Pk).conj() ) 
                GMM = self.Reverse_p(GMM)
                
                error = np.zeros((K+1,2))
                if np.linalg.norm(GMPref) != 0:
                    error[0,0] = np.linalg.norm(GMP-GMPref) / np.linalg.norm(GMPref)
                else:
                    error[0,0] = np.linalg.norm(GMP-GMPref)
                error[0,1] = np.linalg.norm(GMM-GMMref) / np.linalg.norm(GMMref)
                
                # Track energy conservation of the focusing function
                energy        = np.zeros((self.nf,F1Pk.shape[1],K+1),dtype=complex)
                energy[:,:,0] = self.My_dot(F1Pk,self.My_T(F1Pk).conj()) - self.My_dot(F1Mk,self.My_T(F1Mk).conj())
                
        print('(6/6) Compute %d Marchenko iterations for $p = %.2f*1e-3$.'%(K,p*1e3))
            
        # Iterate 
        for k in range(K):
            
            print('\t Iteration (%d/%d).'%(k+1,K))
            
            MP = self.My_dot(Rdag,F1Mk)
            
            # I think the window should be time reversed according to Eq. 27
            mP = W*self.WP2TP(self.Sort_w(MP),norm='ortho')
            MP = np.fft.fft(np.fft.ifftshift(mP,0),n=None,axis=0,norm='ortho')[:self.nf,:]
            
            dF1Mk = self.My_dot(R,MP)
            df1Mk = W*self.WP2TP(self.Sort_w(dF1Mk),norm='ortho')
            dF1Mk = np.fft.fft(np.fft.ifftshift(df1Mk,0),n=None,axis=0,norm='ortho')[:self.nf,:]
            
            F1Pk = F1P0 + MP
            F1Mk = F1M0 + dF1Mk
            
            if convergence == 1:
                # 1st Marchenko equation
                GMP = self.Sort_w( self.My_dot(R,F1Pk) - F1Mk ) 
                
                # 2nd Marchenko equation
                GMM = self.Sort_w( (self.My_dot(Rdag,F1Mk) - F1Pk).conj() ) 
                GMM = self.Reverse_p(GMM)
            
                if np.linalg.norm(GMPref) != 0:
                    error[k+1,0] = np.linalg.norm(GMP-GMPref) / np.linalg.norm(GMPref)
                else:
                    error[k+1,0] = np.linalg.norm(GMP-GMPref)
                error[k+1,1] = np.linalg.norm(GMM-GMMref) / np.linalg.norm(GMMref)
                
                # Track energy conservation of the focusing function
                energy[:,:,k+1] = self.My_dot(F1Pk,self.My_T(F1Pk).conj()) - self.My_dot(F1Mk,self.My_T(F1Mk).conj())
            
        # 1st Marchenko equation
        GMP = self.Sort_w( self.My_dot(R,F1Pk) - F1Mk ) 
        
        # 2nd Marchenko equation
        GMM = self.Sort_w( (self.My_dot(Rdag,F1Mk) - F1Pk).conj() ) 
        GMM = self.Reverse_p(GMM)
        
        # Compute negative frequencies and wavenumbers    
        R,F1Pk,F1Mk = self.Sort_w(R,F1Pk,F1Mk)
        
        if plot == 1:
            
            # Plot Marchenko Green's functions
            gMM  = self.WP2TP(GMM,norm='ortho')
            gMMw = self.ConvolveRicker(gMM,f0)[0]
            vmin = - np.max(gMMw)*5e-1
            vmax = -vmin
            self.Plot((1-0)*gMMw,t=1,tvec=1,vmin=vmin,vmax=vmax,title='$ \mathbf{G}_{K=%d}^{-,-}$, $p = %.2f*1e-3$'%(K,p*1e3))
            
            gMP = self.WP2TP(GMP,norm='ortho')
            gMPw = self.ConvolveRicker(gMP,f0)[0]
            self.Plot((1-W)*gMPw,t=1,tvec=1,vmin=vmin,vmax=vmax,title='$(\mathbf{I} - \mathbf{W}) \mathbf{G}_{K=%d}^{-,+}$, $p = %.2f*1e-3$'%(K,p*1e3))
            
            # Plot focusing functions
            f1Pkw = self.ConvolveRicker(self.WP2TP(F1Pk,norm='ortho'),f0)[0]
            self.Plot(f1Pkw,t=1,tvec=1,vmin=vmin,vmax=vmax,title='$\mathbf{F}_{1,K=%d}^{+}$, $p = %.2f*1e-3$'%(K,p*1e3))
            f1Mkw = self.ConvolveRicker(self.WP2TP(F1Mk,norm='ortho'),f0)[0]
            self.Plot(f1Mkw,t=1,tvec=1,vmin=vmin,vmax=vmax,title='$\mathbf{F}_{1,K=%d}^{-}$, $p = %.2f*1e-3$'%(K,p*1e3))
            
            if convergence == 1:
                plt.figure()
                plt.plot(error[:,0])
                plt.xlabel('Iteration')
                plt.ylabel('Relative error')
                plt.title('$L_2( \mathbf{G}^{-,+} - \mathbf{G}_{ref}^{-,+} ) / L_2( \mathbf{G}_{ref}^{-,+} ) $')
                
                plt.figure()
                plt.plot(error[:,1])
                plt.xlabel('Iteration')
                plt.ylabel('Relative error')
                plt.title('$L_2( \mathbf{G}^{-,-} - \mathbf{G}_{ref}^{-,-} ) / L_2( \mathbf{G}_{ref}^{-,-} ) $')
            
            if Gref == 1:          
                
                # Plot reference Green's functions
                gMPref = self.WP2TP(GMPref,eps=eps,norm='ortho')
                gMPrefw = self.ConvolveRicker(gMPref,f0)[0]
                self.Plot(gMPrefw,t=1,tvec=1,vmin=vmin,vmax=vmax,title='$\mathbf{G}_{ref}^{-,+}$, $p = %.2f*1e-3$'%(p*1e3))
                
                gMMref = self.WP2TP(GMMref,eps=eps,norm='ortho')
                gMMrefw = self.ConvolveRicker(gMMref,f0)[0]
                self.Plot(gMMrefw,t=1,tvec=1,vmin=vmin,vmax=vmax,title='$\mathbf{G}_{ref}^{-,-}$, $p = %.2f*1e-3$'%(p*1e3))
                
                self.Plot((1-W)*gMPw - gMPrefw,t=1,tvec=1,vmin=1e-1*vmin,vmax=1e-1*vmax,title='$(\mathbf{I} - \mathbf{W}) \mathbf{G}_{K=%d}^{-,+} - \mathbf{G}_{ref}^{-,+}$, $p = %.2f*1e-3$'%(K,p*1e3))
                self.Plot((1-0)*gMMw - gMMrefw,t=1,tvec=1,vmin=1e-1*vmin,vmax=1e-1*vmax,title='$\mathbf{G}_{K=%d}^{-,-} - \mathbf{G}_{ref}^{-,-}$, $p = %.2f*1e-3$'%(K,p*1e3))
        
        if convergence == 0:
            error = None
            energy = None
            
        if Gref == 0:
            GMPref = None
            GMMref = None
        
        Out = {'R':R,'GMP':GMP,'GMM':GMM,'W':W,'F1P0':F1P0,'F1Pk':F1Pk,'F1Mk':F1Mk,'GMPref':GMPref,'GMMref':GMMref,'error':error,'energy':energy}
        
        return Out
    
    def Projected_marchenko_series(self,p,zF,R=None,V1P0=None,P=None,f0=10,taplen=None,K=0,plot=1,Uref=0,mul=1,conv=1,eps=None,convergence=0):
        """
        Out = Projected_marchenko_series(self,p,zF,R=None,V1P0=None,P=None,f0=10,taplen=None,K=0,plot=1,Uref=0,mul=1,conv=1,eps=None,convergence=0)
        
        Evaluate K iterations of the projected Marchenko series. Input wavefieds can be given. However, it seems to be important to attenuate temporal 
        wrap-around effects to achieve convergence of the Marchenko series.
        
        Inputs:
            p:              Ray-parameter p.
            zF:             Focusing depth
            R:              (Optional) Reflection response in the F-P domain (nt,4) or (nf,4).
            V1P0:           (Optional) Initial projected focusing function in the F-P domain (nt,4) or (nf,4), otherwise it will be a delta function in space and time.
            P:              (Optional) Projected window for the projected Marchenko seris (nt x 4).
            f0:             Default f0=10. Frequency for Ricker wavelet in Hz.
            taplen:         Width of wavelet in seconds to build a projector that accounts for the finite bandwidth.
            K:              Default K=0. Number of iterations of the Marchenko series.
            plot:           Default plot=1. Set plot=1 to automatically plot the outputs. Set plot=0 to suppress automatic plotting.
            Uref:           Default Gref=0. Set Gref=1 to model projected reference Green's functions.
            mul:            Default mul=1. Set mul=1 to include internal multiples. Set mul=0 to exclude internal multiples.
            conv:           Default conv=1. Set conv=1 to include P/S conversions. Set conv=0 to exclude P/S conversions.
            eps:            Constant that is multiplied by 1j and added to the frequency vector to attenuate temporal wrap-around effects. 
                            Default eps = 3 / (self.dt * self.nf)
            convergence:    Set convergence=1 to make a convergence plot.
        
        Output:
            Out: Dictionary containing relfection response, retrieved projected Green's functions, projected time window, 
            projected focusing functions (initial downgoing, downgoing, and upgoing), modelled projected reference Green's functions.
            {'R':R,
            'UMP':UMP,
            'UMM':UMM,
            'P':P,
            'V1P0':V1P0,
            'V1Pk':V1Pk,
            'V1Mk':V1Mk,
            'UMPref':UMPref,
            'UMMref':UMMref}
        
        """
        
        # Check if focusing depth has changed
        if (zF != self.zF) or (p != self.p):
            self.p = p
            self.zF = zF
            self.F1P = None
            self.F1M = None
            self.F1P_neps = None # Focusing funcion with negative eps
            self.F1M_neps = None # Focusing funcion with negative eps
            self.F1P0 = None
           
        # Constant to dampen the temporal wrap-around   
        if eps is None:
            eps = 3/(self.dt*self.nf)
        
        # Model reflection data if not given
        # I recommend to dampen the temporal wrap-around!
        if R is None:
            print('(1/6) Compute reflection data R for $p = %.2f*1e-3$ and $\epsilon = %.2f$.'%(p*1e3,eps))
            R = self.Layercel_p_w(p=p,mul=mul,conv=conv,eps=eps,sort=1)[0]
            r = self.WP2TP(R,eps=eps,norm='ortho',taperlen=30)
            R = np.fft.fft(np.fft.ifftshift(r,0),n=None,axis=0,norm='ortho')
        else:
            print('(1/6) Use reflection data R passed by the user.')
        
        # Select positive frquency elements and compute R dagger
        R    = R[:self.nf,:]
        Rdag = self.My_T(R).conj()
          
        # Projected initial focusing function V1P0 is a delta function in space and time by default
        if V1P0 is None:
            print('(2/6) Set projected initial focusing function $V_{1,0}^+ = \delta(x) \delta(t)$.')
            V1P0      = np.zeros_like(R)
            V1P0[:,0] = 1
            V1P0[:,3] = 1
        else:
            print('2/6 Use projected initial focusing function $V_{1,0}^+$ passed by the user.')
            if V1P0.shape[0] == self.nt:
                V1P0 = V1P0[:self.nf,:]
        
        # Make projected window if not given
        if P is None:
            print('(3/6) Make projecyed time window $P$ for $p = %.2f*1e-3$.'%(p*1e3))
            
            # One way travel times
            self.Insert_layer(zF)
            Trunc = self.Truncate()[-1]
            self.Remove_layer(zF)
            t1p = Trunc.dzvec*(1/Trunc.cpvec**2 - p**2)**0.5
            t1s = Trunc.dzvec*(1/Trunc.csvec**2 - p**2)**0.5
            
            if len(t1p) > 2:
                t2pp = 2*t1p[0] + 2*np.sum(t1p[1:-1])
            elif len(t1p) == 2:
                t2pp = 2*t1p[0]
            else:
                print('Focusing depth is not below the acquisition surface!')
                
            t2ps = t2pp - t1p[0] + t1s[0]
            t2ss = t2pp - 2*t1p[0] + 2*t1s[0]
            
            # Early end of the projector
            t0pp = 0
            t0ps = -t1p[0]+t1s[0]
            t0sp = -t1s[0]+t1p[0]
            t0ss = 0
            
            picks = np.array([[t0pp,t0ps,t0sp,t0ss],[t2pp,t2ps,t2ps,t2ss]])
            
            if taplen == None:
                # Time between the two minima of a Rickr wavelet
                taplen = 2*np.sqrt(3/2)/(np.pi*f0)
            
            # Compute first arrival times
            t0pp = int(np.ceil((t0pp + taplen)/self.dt)) + self.nf - 1
            t0ps = int(np.ceil((t0ps - taplen)/self.dt)) + self.nf - 1
            t0sp = int(np.ceil((t0sp - taplen)/self.dt)) + self.nf - 1
            t2pp = int(np.floor((t2pp - taplen)/self.dt)) + self.nf - 1
            t2ps = int(np.floor((t2ps - taplen)/self.dt)) + self.nf - 1
            t2ss = int(np.floor((t2ss - taplen)/self.dt)) + self.nf - 1
            
            # Define the projector
            P = np.zeros((self.nt,4))
            P[t0pp:t2pp+1,0] = 1
            P[t0ps:t2ps+1,1] = 1
            P[t0sp:t2ps+1,2] = 1
            P[t0pp:t2ss+1,3] = 1
            
        else:
            print('(3/6) Use the projected time window $P$ passed by the user.')
        
        
        # Compute projected initial upgoing focusing function
        V1M0 = self.My_dot(R,V1P0)
        v1M0 = P*self.WP2TP(self.Sort_w(V1M0),norm='ortho')
        V1M0 = np.fft.fft(np.fft.ifftshift(v1M0,0),n=None,axis=0,norm='ortho')[:self.nf,:]
        
        # Update focusing function
        V1Pk = V1P0.copy()
        V1Mk = V1M0.copy()
        
#        if Gref == 1 or convergence == 1:          
#            print('(5/6) Compute reference Greens function $\mathbf{G}_{ref}^{-,+}$ and $\mathbf{G}_{ref}^{-,-}$.')
#            Gs            = self.Gz2bound(p,zF,mul=mul,conv=conv,eps=eps)[0]
#            GMPref,GMMref = Gs[2:4]
#            
#            if convergence == 1:
#                # 1st Marchenko equation
#                GMP = self.Sort_w( self.My_dot(R,F1Pk) - F1Mk ) 
#                
#                # 2nd Marchenko equation
#                GMM = self.Sort_w( (self.My_dot(Rdag,F1Mk) - F1Pk).conj() ) 
#                GMM = self.Reverse_p(GMM)
#                
#                error = np.zeros((K+1,2))
#                if np.linalg.norm(GMPref) != 0:
#                    error[0,0] = np.linalg.norm(GMP-GMPref) / np.linalg.norm(GMPref)
#                else:
#                    error[0,0] = np.linalg.norm(GMP-GMPref)
#                error[0,1] = np.linalg.norm(GMM-GMMref) / np.linalg.norm(GMMref)
#                
#                # Track energy conservation of the focusing function
#                energy        = np.zeros((self.nf,F1Pk.shape[1],K+1),dtype=complex)
#                energy[:,:,0] = self.My_dot(F1Pk,self.My_T(F1Pk).conj()) - self.My_dot(F1Mk,self.My_T(F1Mk).conj())
#                
        print('(4/6) Compute %d projected Marchenko iterations for $p = %.2f*1e-3$.'%(K,p*1e3))
            
        # Iterate 
        for k in range(K):
            
            print('\t Iteration (%d/%d).'%(k+1,K))
            
            MP = self.My_dot(Rdag,V1Mk)
            
            # I think the window should be time reversed according to Eq. 27
            mP = P*self.WP2TP(self.Sort_w(MP),norm='ortho')
            MP = np.fft.fft(np.fft.ifftshift(mP,0),n=None,axis=0,norm='ortho')[:self.nf,:]
            
            dV1Mk = self.My_dot(R,MP)
            dv1Mk = P*self.WP2TP(self.Sort_w(dV1Mk),norm='ortho')
            dV1Mk = np.fft.fft(np.fft.ifftshift(dv1Mk,0),n=None,axis=0,norm='ortho')[:self.nf,:]
            
            V1Pk = V1P0 + MP
            V1Mk = V1M0 + dV1Mk
            
#            if convergence == 1:
#                # 1st Marchenko equation
#                GMP = self.Sort_w( self.My_dot(R,F1Pk) - F1Mk ) 
#                
#                # 2nd Marchenko equation
#                GMM = self.Sort_w( (self.My_dot(Rdag,F1Mk) - F1Pk).conj() ) 
#                GMM = self.Reverse_p(GMM)
#            
#                if np.linalg.norm(GMPref) != 0:
#                    error[k+1,0] = np.linalg.norm(GMP-GMPref) / np.linalg.norm(GMPref)
#                else:
#                    error[k+1,0] = np.linalg.norm(GMP-GMPref)
#                error[k+1,1] = np.linalg.norm(GMM-GMMref) / np.linalg.norm(GMMref)
#                
#                # Track energy conservation of the focusing function
#                energy[:,:,k+1] = self.My_dot(F1Pk,self.My_T(F1Pk).conj()) - self.My_dot(F1Mk,self.My_T(F1Mk).conj())
            
        # 1st Marchenko equation
        UMP = self.Sort_w( self.My_dot(R,V1Pk) - V1Mk ) 
        
        # 2nd Marchenko equation
        UMM = self.Sort_w( (self.My_dot(Rdag,V1Mk) - V1Pk).conj() ) 
        UMM = self.Reverse_p(UMM)
        
        # Compute negative frequencies and wavenumbers    
        R,V1Pk,V1Mk = self.Sort_w(R,V1Pk,V1Mk)
#        
#        if plot == 1:
#            
#            # Plot Marchenko Green's functions
#            gMM  = self.WP2TP(GMM,norm='ortho')
#            gMMw = self.ConvolveRicker(gMM,f0)[0]
#            vmin = - np.max(gMMw)*5e-1
#            vmax = -vmin
#            self.Plot((1-0)*gMMw,t=1,tvec=1,vmin=vmin,vmax=vmax,title='$ \mathbf{G}_{K=%d}^{-,-}$, $p = %.2f*1e-3$'%(K,p*1e3))
#            
#            gMP = self.WP2TP(GMP,norm='ortho')
#            gMPw = self.ConvolveRicker(gMP,f0)[0]
#            self.Plot((1-W)*gMPw,t=1,tvec=1,vmin=vmin,vmax=vmax,title='$(\mathbf{I} - \mathbf{W}) \mathbf{G}_{K=%d}^{-,+}$, $p = %.2f*1e-3$'%(K,p*1e3))
#            
#            # Plot focusing functions
#            f1Pkw = self.ConvolveRicker(self.WP2TP(F1Pk,norm='ortho'),f0)[0]
#            self.Plot(f1Pkw,t=1,tvec=1,vmin=vmin,vmax=vmax,title='$\mathbf{F}_{1,K=%d}^{+}$, $p = %.2f*1e-3$'%(K,p*1e3))
#            f1Mkw = self.ConvolveRicker(self.WP2TP(F1Mk,norm='ortho'),f0)[0]
#            self.Plot(f1Mkw,t=1,tvec=1,vmin=vmin,vmax=vmax,title='$\mathbf{F}_{1,K=%d}^{-}$, $p = %.2f*1e-3$'%(K,p*1e3))
#            
#            if convergence == 1:
#                plt.figure()
#                plt.plot(error[:,0])
#                plt.xlabel('Iteration')
#                plt.ylabel('Relative error')
#                plt.title('$L_2( \mathbf{G}^{-,+} - \mathbf{G}_{ref}^{-,+} ) / L_2( \mathbf{G}_{ref}^{-,+} ) $')
#                
#                plt.figure()
#                plt.plot(error[:,1])
#                plt.xlabel('Iteration')
#                plt.ylabel('Relative error')
#                plt.title('$L_2( \mathbf{G}^{-,-} - \mathbf{G}_{ref}^{-,-} ) / L_2( \mathbf{G}_{ref}^{-,-} ) $')
#            
#            if Gref == 1:          
#                
#                # Plot reference Green's functions
#                gMPref = self.WP2TP(GMPref,eps=eps,norm='ortho')
#                gMPrefw = self.ConvolveRicker(gMPref,f0)[0]
#                self.Plot(gMPrefw,t=1,tvec=1,vmin=vmin,vmax=vmax,title='$\mathbf{G}_{ref}^{-,+}$, $p = %.2f*1e-3$'%(p*1e3))
#                
#                gMMref = self.WP2TP(GMMref,eps=eps,norm='ortho')
#                gMMrefw = self.ConvolveRicker(gMMref,f0)[0]
#                self.Plot(gMMrefw,t=1,tvec=1,vmin=vmin,vmax=vmax,title='$\mathbf{G}_{ref}^{-,-}$, $p = %.2f*1e-3$'%(p*1e3))
#                
#                self.Plot((1-W)*gMPw - gMPrefw,t=1,tvec=1,vmin=1e-1*vmin,vmax=1e-1*vmax,title='$(\mathbf{I} - \mathbf{W}) \mathbf{G}_{K=%d}^{-,+} - \mathbf{G}_{ref}^{-,+}$, $p = %.2f*1e-3$'%(K,p*1e3))
#                self.Plot((1-0)*gMMw - gMMrefw,t=1,tvec=1,vmin=1e-1*vmin,vmax=1e-1*vmax,title='$\mathbf{G}_{K=%d}^{-,-} - \mathbf{G}_{ref}^{-,-}$, $p = %.2f*1e-3$'%(K,p*1e3))
#        
#        if convergence == 0:
#            error = None
#            energy = None
#        
        Out = {'R':R,'P':P,'V1P0':V1P0,'picks':picks,'taplen':taplen,'V1M0':V1M0,'V1Pk':V1Pk,'V1Mk':V1Mk,'UMP':UMP,'UMM':UMM}
        
        return Out
    
#    # Old version of FocusFunc1_mod() without option to compute piecewise
##    # An attempt to model the focusing function
##    def FocusFunc1_mod(self,zF=None,mul=1,conv=1,eps=None,sort=1):
##        """
##        R,T = Layercel_kx_w(mul,conv,nf,nk)
##        Compute reflection / transmission response for a single wavenumber and all positive frequencies as well as the highest negative frequency.
##        
##        Inputs:
##            mul:  Set mul=1 to model internal multiples.
##            conv: Set conv=1 to model P/S conversions.
##            eps:  Set eps to add an imaginary constant to the frequency: w -> w + 1j*eps. This reduces the wrap-around in time but one has to correct for this by multiplying the data by exp(eps*t) in the time domain.
##            sort: Set sort=1 (default) to get positive and negative frequencies and wavenumbers.
##        
##        Output:
##            RP: Reflection response from above (nf x 4), 1st element corresponds to zero frequency
##            TP: Transmission response from above (nf x 4), 1st element corresponds to zero frequency
##        """
##        
##        # Check if focusing depth has changed
##        if zF != self.zF:
##            self.zF = zF
##            self.F1P = None
##            self.F1M = None
##            self.F1P_neps = None # Focusing funcion with negative eps
##            self.F1M_neps = None # Focusing funcion with negative eps
##        
##        # Truncate medium
##        Trunc = self.Truncate()[-1]
##        
##        # Number of layers
##        N = np.size(Trunc.cpvec)
##        
##        # Number of positive frequency and positive wavenumber samples
##        nk = int(self.nr/2) + 1
##        nf = int(self.nt/2) + 1
##        
##        if eps is None:
##            eps = 3/(nf*self.dt)
##        
##        # Frequency and wavenumber meshgrids
##        Wfft,Kxfft = self.W_Kx_grid()[2:4]
##    
##        # Extract positive frequencies and wavenumbers
##        Wpos  = Wfft[0:nf,0:nk] + 1j*eps
##        Kxpos = Kxfft[0:nf,0:nk]
##        
##        # Propagation and scattering matrices of an infinitesimal layer without any contrast
##        
##        W = np.zeros((nf,nk,4),dtype=complex)
##        Winv = np.zeros((nf,nk,4),dtype=complex)
##        
##        RP = np.zeros((nf,nk,4),dtype=complex)
##        RM = np.zeros((nf,nk,4),dtype=complex)
##        
##        I = np.zeros((nf,nk,4),dtype=complex)
##        I[:,:,0] = 1
##        I[:,:,3] = 1
##        M1 = I.copy()
##        M2 = I.copy()
##        
##        # Here every frequency and every wavenumber component have an amplitude 
##        # equal to one. Hence, the total wavefield has a strength of sqrt(nt*nr)
##        # When an inverse fft (ifft2) is applied the wavefield is scaled by 
##        # 1/sqrt(nt*nr) hence in the time domain the wavefield has the an
##        # amplitude equal to one.
##        TP = I.copy()
##        TM = I.copy()
##        F1P = I.copy()
##        F1M = RP.copy()
##        
##        # Loop over N-1 interfaces
##        for n in range(0,N-1):
##            
##            dz1 = Trunc.dzvec[n]
##            
##            # Parameters of top layer
##            cp1 = Trunc.cpvec[n]
##            cs1 = Trunc.csvec[n]
##            ro1 = Trunc.rovec[n]
##        
##            # Parameters of bottom layer
##            cp2 = Trunc.cpvec[n+1]
##            cs2 = Trunc.csvec[n+1]
##            ro2 = Trunc.rovec[n+1]
##            
##            kzp = np.sqrt(Wpos**2/cp1**2-Kxpos**2) # kz for p-waves
##            kzs = np.sqrt(Wpos**2/cs1**2-Kxpos**2) # kz for s-waves
##            
##            W[:,:,0] = np.exp(1j*kzp*dz1)
##            W[:,:,3] = np.exp(1j*kzs*dz1)
##            
##            Winv = self.My_inv(W)
##            Winv = np.nan_to_num(Winv)
##            
##            
##            rP,tP,rM,tM,tPinv = self.RT_kx_w(cp1,cs1,ro1,cp2,cs2,ro2,Kxpos,Wpos,conv,nf,nk)
##    
##            if mul == 1:
##                M1inv = I - self.Mul_My_dot(RM,W,rP,W)
##                # Inverse of tmp
##                M1 = self.My_inv(M1inv)
##                
##                M2inv = I - self.Mul_My_dot(rP,W,RM,W)
##                # Inverse of tmp
##                M2 = self.My_inv(M2inv)
##                
##            # Update reflection / transmission responses
##            RP = RP + self.Mul_My_dot(TM,W,rP,W,M1,TP)
##            RM = rM + self.Mul_My_dot(tP,W,RM,W,M2,tM)
##            TP = self.Mul_My_dot(tP,W,M1,TP)
##            TM = self.Mul_My_dot(TM,W,M2,tM)   
##            F1P = self.Mul_My_dot(F1P,M1inv,Winv,tPinv)
##        
##        
##        
##        # At the end compute F1M
##        F1M = self.My_dot(RP,F1P)
##                
##    
##        # The highest negative frequency and highest negative wavenumber components are real-valued
##        RP[nf-1,:] = RP[nf-1,:].real
##        TP[nf-1,:] = TP[nf-1,:].real
##        F1P[nf-1,:] = F1P[nf-1,:].real
##        F1M[nf-1,:] = F1M[nf-1,:].real
##        RP[:,nk-1] = RP[:,nk-1].real
##        TP[:,nk-1] = TP[:,nk-1].real
##        F1P[:,nk-1] = F1P[:,nk-1].real
##        F1M[:,nk-1] = F1M[:,nk-1].real
##        
##        
##        # Remove Nans and values above let's say 10
##        RP = np.nan_to_num(RP)
##        TP = np.nan_to_num(TP)
##        F1P = np.nan_to_num(F1P)
##        F1M = np.nan_to_num(F1M)
##        
##        # Conjugate wavefields
##        RP = RP.conj()
##        TP = TP.conj()
##        F1P = F1P.conj()
##        F1M = F1M.conj()
##        
##        # Write full reflection and transmission matrices
##        if sort == 1:
###            Rfull,Tfull = self.Sort_RT_kx_w(RP,TP)
##            RPfull,TPfull,F1Pfull,F1Mfull = self.Sort_kx_w(RP,TP,F1P,F1M)
##            return RPfull,TPfull,F1Pfull,F1Mfull,RP,TP,F1P,F1M
##        
##        return RP,TP,F1P,F1M