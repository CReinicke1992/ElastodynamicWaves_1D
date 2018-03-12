#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:03:17 2017

@author: christianreini
"""

from Layers_kx_w_vec import Layers_kx_w_vec
import numpy as np
import matplotlib.pylab as plt
import scipy as sc

# Focusing and Green's functions between a focal depth and 
# the surface for layered media with transparent top and 
# bottom boundaries.
# All properties of wavefields in layered media are inheritted
# form the class Layers_kx_w_vec
class Marchenko_kx_w(Layers_kx_w_vec):  
    """
    Marchenko_kx_w
    
    Compute focusing and Green's functions between a focal depth and the surface of layered media for multiple frequencies and offsets.
    
    Variables:
        nt:    Number of time/frequency samples
        dt:    Duration per time sample in seconds
        nr:    Number of space/wavenumber samples
        dx:    Distance per space samples in metres
        dzvec: List or array with the thickness of each layer
        cpvec: List or array with the P-wave veclocity in each layer
        csvec: List or array with the S-wave veclocity in each layer
        rovec: List or array with the density of each layer
        zF:    Focal depth
        
    Data sorting: 
        nt x nr x 4
        
    Vectorised computation
    """
    
    # All properties of wavefields in layered media are inheritted from
    # Layers_kx_w_vec
    def __init__(self,nt,dt,nr,dx,dzvec,cpvec,csvec,rovec):
        Layers_kx_w_vec.__init__(self,nt,dt,nr,dx,dzvec,cpvec,csvec,rovec)
        # Focal depth
        self.zF = None
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
        
        Trunc = Layers_kx_w_vec(nt=self.nt,dt=self.dt,nr=self.nr,dx=self.dx,dzvec=dzvec_tr,cpvec=cpvec_tr,csvec=csvec_tr,rovec=rovec_tr)
        
        return dzvec_tr,cpvec_tr,csvec_tr,rovec_tr,Trunc
    
    
    # Compute focusing functions
    def FocusFunc1_inv(self,zF,mul=1,conv=1,eps=None,neps=0):
        """
        F1Pfull,F1Mfull,RPfull,TPfull,F1P,F1M = FocusFunc1_inv(mul=1,conv=1,eps=None)
        
        Reflection and transmission response of the truncated medium are modelled. The responses are corrected for the complex-valued frequency. Then the focusing functions are computed.
        
        Inputs:
            zF:       Set a focal depth.
            mul:      Set mul=1 to include internal multiples.
            conv:     Set conv=1 to include P/S conversions.
            eps:      Set eps to add an imaginary constant to the frequency: w -> w + 1j*eps. This reduces the wrap-around in time but one has to correct for this by multiplying the data by exp(eps*t) in the time domain.
            neps:     Set neps=1 to compute the focusing functions also with negative eps
            
        Outputs:
            F1P:      Downgoing focusing function for all frequencies and wavenumbers. (nt,nr,4)
            F1M:      Upgoing focusing function for all frequencies and wavenumbers. (nt,nr,4)
            RP_tr:    Reflection response of truncated medium  for all frequencies and wavenumbers. (nt,nr,4). (no correction for complex-valued frequency applied)
            TP_tr:    Transmission response of truncated medium  for all frequencies and wavenumbers. (nt,nr,4). (no correction for complex-valued frequency applied, here eps is negative!)
            self.F1P: Downgoing focusing function for all positive and the highest negative frequencies / wavenumbers. (nt,nr,4)
            self.F1M: Upgoing focusing function for all positive and the highest negative frequencies / wavenumbers. (nt,nr,4)
            
        Question:
            Is it possible to invert TP without correcting for the complex-valued frequency, and to corrct F1P for the complex-valued frequency?
            
        """
        
        # Check if focusing depth has changed
        if zF != self.zF:
            self.zF = zF
            self.F1P = None
            self.F1M = None
            self.F1P_neps = None # Focusing funcion with negative eps
            self.F1M_neps = None # Focusing funcion with negative eps
            self.F1P0 = None
        
        # Number of frequency samples
        nf = int(self.nt/2)+1
        
        # Insert layer
        self.Insert_layer(self.zF)
        
        # Constant imaginary
        if eps is None:
            eps = 3/(nf*self.dt)
        
        if self.F1P is None:
            
            print('Computing focusing functions ...')
            
            # Truncate medium    
            Trunc = self.Truncate()[-1]
            
            # Compute reflection/transmission response of truncated medium for all positive frequencies and wavenumbers
            RP,TP = Trunc.Layercel_kx_w(mul=mul,conv=conv,eps=eps,sort=0)[0:2]
            
            # Compute F1plus
            self.F1P = self.My_inv(TP)
            self.F1P = np.nan_to_num(self.F1P)
            
            # Compute F1minus
            self.F1M = self.My_dot(RP,self.F1P)
            self.F1M = np.nan_to_num(self.F1M)
            
            # Construct negative frequencies/wavenumbers
            F1Pfull,F1Mfull = self.Sort_kx_w(self.F1P,self.F1M)
        
        # Construct negative frequencies/wavenumbers
        F1Pfull,F1Mfull = self.Sort_kx_w(self.F1P,self.F1M)
        
        if (self.F1P_neps is None) and (neps == 1):
            
            print('Computing focusing functions with negative eps ...')
            
            # Truncate medium    
            Trunc = self.Truncate()[-1]
            
            # Compute reflection/transmission response of truncated medium for all positive frequencies and wavenumbers
            RP,TP = Trunc.Layercel_kx_w(mul=mul,conv=conv,eps=-eps,sort=0)[0:2]
            
            # Compute F1plus
            self.F1P_neps = self.My_inv(TP)
            self.F1P_neps = np.nan_to_num(self.F1P_neps)
            
            # Compute F1minus
            self.F1M_neps = self.My_dot(RP,self.F1P_neps)
            self.F1M_neps = np.nan_to_num(self.F1M_neps)
            
        # Remove layer
        self.Remove_layer(self.zF)
            
        return F1Pfull,F1Mfull,self.F1P,self.F1M,self.F1P_neps,self.F1M_neps
        
    
    # Compute Green's functions
    def GreensFunc(self,zF,mul=1,conv=1,eps=None,RPfull=None,mod=0):
        """
        GMPfull,GMMfull,GMP,GMM = GreensFunc(zF,mul=1,conv=1,eps=None,taperlen=None,RPfull=None)
        
        The Green's functions are computed via the Marchenko equations. Hence, the focusing functions and the full-medium's reflection response are used.
        
         Inputs:
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
        
        print('Computing Greens functions ...')
        
        # Number of frequency samples
        nf = int(self.nt/2)+1
        
        # Constant imaginary
        if eps is None:
            eps = 3/(nf*self.dt)
         
        # Model truncated transmission and invert it
        if mod == 0:
            # FocusFunc1_inv will only compute something IF the focal depth has changed
            # That means zF is different to self.zF  
            # Since self.F1P is self. the below function does not require any output
            self.FocusFunc1_inv(zF,mul,conv,eps,neps=1)
            
        # Model focusing function layer by layer
        else:
            # FocusFunc1_mod will always compute something even IF the focal depth has not changed
            self.F1P,self.F1M = self.FocusFunc1_mod(zF,mul,conv,eps,sort=0)[2:4]
            self.F1P_neps,self.F1M_neps = self.FocusFunc1_mod(zF,mul,conv,-eps,sort=0)[2:4]
        
        # Compute reflection/transmission response of  medium for positive frequencies and wavenumbers
        if RPfull is None:         
            RP = self.Layercel_kx_w(mul=mul,conv=conv,eps=eps,sort=0)[0]
        
        # 1st Marchenko equation
        GMP = self.My_dot(RP,self.F1P) - self.F1M
        GMP = np.nan_to_num(GMP)
        
        # Compute R dagger
        RPdagger = self.My_T(RP).conj()
        
        # 2nd Marchenko equation
        GMM = (self.My_dot(RPdagger,self.F1M_neps) - self.F1P_neps).conj()
        GMM = np.nan_to_num(GMM)
        
        # Get negative frequencies / wavenumbers
        GMPfull,GMMfull = self.Sort_kx_w(GMP,GMM)
        
        # In the 2nd Marchenko equation the data is complex-conjugated
        # Hence the kx-axis is reversed
        GMMfull[:,1:,:] = GMMfull[:,-1:0:-1,:]
        
        return GMPfull,GMMfull
    
    # An attempt to model the focusing function
    def FocusFunc1_mod(self,zF=None,mul=1,conv=1,eps=None,sort=1,initials=[]):
        """
        R,T = Layercel_kx_w(mul,conv,nf,nk)
        Compute reflection / transmission response for a single wavenumber and all positive frequencies as well as the highest negative frequency.
        
        Inputs:
            mul:  Set mul=1 to model internal multiples.
            conv: Set conv=1 to model P/S conversions.
            eps:  Set eps to add an imaginary constant to the frequency: w -> w + 1j*eps. This reduces the wrap-around in time but one has to correct for this by multiplying the data by exp(eps*t) in the time domain.
            sort: Set sort=1 (default) to get positive and negative frequencies and wavenumbers.
        
        Output:
            RP: Reflection response from above (nf x 4), 1st element corresponds to zero frequency
            TP: Transmission response from above (nf x 4), 1st element corresponds to zero frequency
        """
        
        # Check if focusing depth has changed
        if zF != self.zF:
            self.zF = zF
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
        
        # Number of positive frequency and positive wavenumber samples
        nk = int(self.nr/2) + 1
        nf = int(self.nt/2) + 1
        
        if eps is None:
            eps = 3/(nf*self.dt)
        
        # Frequency and wavenumber meshgrids
        Wfft,Kxfft = self.W_Kx_grid()[2:4]
    
        # Extract positive frequencies and wavenumbers
        Wpos  = Wfft[0:nf,0:nk] + 1j*eps
        Kxpos = Kxfft[0:nf,0:nk]
        
        # Progagation matrix of an infinitesimal layer without any contrast
        W = np.zeros((nf,nk,4),dtype=complex)
        Winv = np.zeros((nf,nk,4),dtype=complex)
        
        # Default multiple matrix (in case mul=0)
        I = np.zeros((nf,nk,4),dtype=complex)
        I[:,:,0] = 1
        I[:,:,3] = 1
        M1 = I.copy()
        M2 = I.copy()
        
        
        if initials == []:
        # Scattering matrices of an infinitesimal layer without any contrast
            
            RP = np.zeros((nf,nk,4),dtype=complex)
            RM = np.zeros((nf,nk,4),dtype=complex)
            
            # Here every frequency and every wavenumber component have an amplitude 
            # equal to one. Hence, the total wavefield has a strength of sqrt(nt*nr)
            # When an inverse fft (ifft2) is applied the wavefield is scaled by 
            # 1/sqrt(nt*nr) hence in the time domain the wavefield has the an
            # amplitude equal to one.
            TP  = I.copy()
            TM  = I.copy()
            F1P = I.copy()
            Nstart = 0
            
            if self.csvec[0] == 0:
                TP[:,:,3]  = 0
                TM[:,:,3]  = 0
                F1P[:,:,3] = 0
            
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
            
            kzp = np.sqrt(Wpos**2/cp1**2-Kxpos**2) # kz for p-waves
            kzs = np.sqrt(Wpos**2/cs1**2-Kxpos**2) # kz for s-waves
            
            W[:,:,0] = np.exp(1j*kzp*dz1)
            if cs1 != 0:
                W[:,:,3] = np.exp(1j*kzs*dz1)   # for elastic layer
            else: 
                W[:,:,3] = 0                    # for purely acoustic layer
            
            Winv = np.zeros((nf,nk,4),dtype=complex)
            Winv[:,:,0] = 1/W[:,:,0]
            if cs1 != 0:
                Winv[:,:,3] = 1/W[:,:,3]
            else:
                Winv[:,:,3] = 0 # This is mathematically not defined. But in a physical interpretation the S-wavefield cannot be inverse propagated in a purely acoustic layer.
            Winv = np.nan_to_num(Winv)
            
            
            rP,tP,rM,tM,tPinv = self.RT_kx_w(cp1,cs1,ro1,cp2,cs2,ro2,Kxpos,Wpos,conv,nf,nk)
    
            if mul == 1:
                M1inv = I - self.Mul_My_dot(RM,W,rP,W)
                # Inverse of tmp
                M1 = self.My_inv(M1inv)
                
                M2inv = I - self.Mul_My_dot(rP,W,RM,W)
                # Inverse of tmp
                M2 = self.My_inv(M2inv)
                
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
    
        # The highest negative frequency and highest negative wavenumber components are real-valued
        RP[nf-1,:] = RP[nf-1,:].real
        TP[nf-1,:] = TP[nf-1,:].real
        F1P[nf-1,:] = F1P[nf-1,:].real
        F1M[nf-1,:] = F1M[nf-1,:].real
        RP[:,nk-1] = RP[:,nk-1].real
        TP[:,nk-1] = TP[:,nk-1].real
        F1P[:,nk-1] = F1P[:,nk-1].real
        F1M[:,nk-1] = F1M[:,nk-1].real
        
        
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
            RPfull,TPfull,F1Pfull,F1Mfull = self.Sort_kx_w(RP,TP,F1P,F1M)
            return RPfull,TPfull,F1Pfull,F1Mfull,initials
        
        return RP,TP,F1P,F1M,initials
    
    
    
    
    
    # Compute initial focusing function using the forward-scattered transmission
    def F1plus0(self,zF,eps=0):
        """
        In testing phase.
        self.F1P0,F1P0full = F1plus0(self,zF)
        
        The initial focusing function (inverse forward-scattered transmission) is computed. No option for complex-valued frequencies.
        
         Inputs:
            zF:       Set a focal depth.
            eps:      Choose complex-valued frequency. Default is eps=0, i.e. the frequency is real-valued.
           
        Outputs:
            F1P0:       Initial focusing function (inverse forward-scattered transmission excluding internal multiples) only positive frequencies and wavenumbers.
            F1P0full:   Initial focusing function (inverse forward-scattered transmission excluding internal multiples) all frequencies and wavenumbers.
            
        """
        
        # Check if focusing depth has changed
        if zF != self.zF:
            self.zF = zF
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
            TP = Trunc.Layercel_kx_w(mul=0,conv=1,eps=eps,sort=0)[1]
            
            # Invert the transmission response
            self.F1P0 = self.My_inv(TP)
            self.F1P0 = np.nan_to_num(self.F1P0)
            
            # Remove layer
            self.Remove_layer(self.zF)
            
        else:
            TP = self.My_inv(self.F1P0)
            TP = np.nan_to_num(TP)
    
        F1P0full,TPfull = self.Sort_kx_w(self.F1P0,TP)
        
        return self.F1P0,F1P0full,TP,TPfull
    
    # Compute time window
    def Window(self,zF,Tfs=1,eps=None,f0=None,Tdata=None,mask=None,Wtaplen=None,plot=1,vmin=None,vmax=None):
        """
        In testing phase.
        W = Window(self,zF,Tfs=1)
        
        Build the window matrix for a focusing depth zF. The window can be based on the forward-scattered transmission (Tfs) or on the direct transmission (Td). This function builds the window at near-offsets based on onset times picked by the user. The window at far-offsets is picked based on a amplitude threshold.
        
         Inputs:
            zF:       Set a focal depth.
            Tfs:      Set Tfs=1 to base the window on the forward scattered transmission. Set Tfs=0 to base the window on the direct transmission.  
            f0:       (Optional) Set central frequency (Ricker wavelet) in Hertz (f not omega). Setting f0 avoids ringiness and hence makes the picking easier.
            Tdata:    (Optional) Pass the forward-scatterd transmission response in the frequency-wavenumber domain [nt x nr (x 4)]. If you pass a transmission response use an eps factor eps=3/(nf*self.dt) to get the correct amplitude scaling. Otherwise the (output) transmission response's amplitudes will be scaled, however, the window matrix should not be affected by that.
            mask:     (Optional) Pass a specific F-Kx mask in the format [nt x nr (x 4)].
            Plot:     Set plot=1 to automatically plot the transmission data, the window as well as the windowed transmission data.
            
        Outputs:
            W:       Window matrix (nt x nr x 4)
            TP:      Forward-scattered transmission response in the frequency-wavenumber domain [nt x nr (x 4)]. If a transmission response Tdata was passed the amplitudes might be scaled!
            tP:      Forward-scattered transmission response in the space-time domain [nt x nr (x 4)]. If a transmission response Tdata was passed the amplitudes might be scaled!
        """
        
        nk = int(self.nr/2) + 1
        nf = int(self.nt/2) + 1
        
        # If forward-scatterd transmission is not given model it 
        if Tdata is None:
            # Check if focusing depth has changed
            if zF != self.zF:
                self.zF = zF
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
            
            if eps is None:
                eps = 3/(nf*self.dt)
            
            if Tfs == 1:
                
                # Compute transmission response of truncated medium excluding internal multiples for all positive frequencies and wavenumbers
                TP = Trunc.Layercel_kx_w(mul=0,conv=1,eps=eps,sort=1)[1]
                    
            elif Tfs == 0:
                print('Not yet implemented.')
                return
        
        else:
            TP = Tdata.copy()
        
        # If F-Kx Mask is not given pick it
        if mask is None:
            mask,cm,fm = self.Pick_FKx_Mask(TP,taplen=20)

        # Transform to space-time domain
        # If the complex-frequency value eps is wrong the amplitudes will be
        # scaled incorrectly but since this function only computes the window 
        # this is not a problem.
        tP = self.WKx2TX(TP*mask,eps=eps)
        
        # If a central frequency is given convolve with Ricker wavelet
        if f0 is not None:
            tP = self.ConvolveRicker(tP,f0)[0]
        
        # Global variable for picks
        global xvals,yvals
        
        xvals = []
        yvals = []
        
        # Function
        def onclick(event):
            xvals.append(event.xdata)
            yvals.append(event.ydata)
            
        # Determine whether array has format (nt x nr x 4) or (nt x nr)
        if TP.ndim == 2:
            N = 1
        elif TP.ndim == 3:
            N = 4
            
        # Number of picks    
        Nc = 6
            
        # Iterate over elastic components to pick 1st arrivals    
        for comp in range(0,N):
            
            # User instruction
            title1 = "\n1. Zoom in using a single click. Only positive offsets and times are required. \n"
            title2 = "2. Pick %d near-offset onset times of the 1st arrival (excluding the 1st arrival). \n"%(Nc)
            title  = title1 + title2
            
            # Plot one elastic component
            if N == 1:
                if (vmin is None) or (vmax is None):
                    vmax = 0.005*tP.max()
                    vmin = -vmax
                fig = self.Plot(tP,tx=1,tvec=1,xvec=1,title=title,vmin=vmin,vmax=vmax)
            else:
                if (vmin is None) or (vmax is None):
                    vmax = 0.005*tP[:,:,comp].max()
                    vmin = -vmax
                fig = self.Plot(tP[:,:,comp],tx=1,tvec=1,xvec=1,title=title,vmin=vmin,vmax=vmax)
            
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
        
        # Mirror negative offsets
        xvals = np.hstack((-xvals[:,::-1], xvals))
        yvals = np.hstack((yvals[:,::-1] , yvals))
        
        # Define a fitting funcion: Hyperbolic moveout
        def Hyperbolic_moveout(x,t0,v):
            return np.sqrt(t0**2 + (x/v)**2)
        
        # Initiate near-offset window
        W = np.zeros((self.nt,self.nr,4))
        
        # The autopicker uses an epsilon equal to the temporal distance of between two minima of the Ricker wavelet
        # If no central frequency is given epsilon remains unchanged (i.e. one time sample)
        if f0 is not None:
            epsilon = 2*np.sqrt(3/2)/(np.pi*f0) # remove times two to get time between maximum and minimum
        
        # Convert epsilon for seconds to time samples
        epsilon = int(epsilon/self.dt)
        
        # Create array to store the onset times of the window
        Onset = np.zeros((self.nr,N))
        
        # Iterate over elastic components to interpolate picks with hyperbola    
        for comp in range(0,N):
            
            # (1) Manually pick 1st arrival onset for near-offsets and interpolate with hyperbola.
            
            # Find hyperpolic fitting parameters
            pars = sc.optimize.curve_fit(Hyperbolic_moveout,xvals[comp,:],yvals[comp,:])[0]# I changed [comp,:] to [:comp]
            
            # In a layered medium there is no energy above the onset of the 1st arrival
            # The automatic picking is facilitated by deleting all amplitudes before 1st arrival
            t0 = int(pars[0]/self.dt)
            tP[:t0-5,:,comp] = 0 # The 5 is arbitrary, just to guarantee that the arrival is not deleted
            
            # Determine the transition from near to far offsets (depends only on the user's picks)
            xmax = xvals[comp,:].max()
            xend = self.Xvec()[0].tolist().index(-self.dx*int(xmax/self.dx))
            
            # Iterate over all positive offsets
            for xx in range(xend,nk):
                # Determine cut-off time for each offset X
                X = self.Xvec()[0][xx]
                t = Hyperbolic_moveout(X,pars[0],pars[1])
                tind = int(t/self.dt) + int(self.nt/2)
                if tind >= self.nt:
                    tind = self.nt
                    
                # Fill window matrix with ones    
                W[:tind,xx,comp] = 1
                
                # Save Onset time
                Onset[xx,comp] = tind
            
            # (2) Automatically pick 1st arrival onset for far-offsets and combine with near-offset picks.
            
            # Threshold: I set it to one permille of the maximum (absolute) amplitude
            th = 0.001*abs(tP[:,:,comp]).max()
            
            # Iterate over all positive offsets
            for xx in range(0,xend):
                trace = tP[:,xx,comp]
                if len(trace[abs(trace)>th]) == 0:
                    tind = self.nt
                else:
                    tind = trace.tolist().index(trace[abs(trace)>th][0]) - epsilon
                
                # Only pick onset times that are not wrapped around
                t = Hyperbolic_moveout(self.Xvec()[0][xx],pars[0],pars[1])
                tind_hyp = int(t/self.dt) + int(self.nt/2)
                if tind < tind_hyp and tind_hyp > self.nt:
                    tind = self.nt
                    
                W[:tind,xx,comp] = 1
        
                
                # Save Onset time
                Onset[xx,comp] = tind
                
            
        # Taper the window
        
        if f0 is not None:
            taplen = epsilon
        else:
            taplen = 10
            
        if Wtaplen is not None:
#            taplen = max(taplen,Wtaplen)
            taplen = Wtaplen.copy()
            
        tap = np.cos(np.linspace(0,np.pi/2,taplen+1))
            
        for comp in range(0,N):
            for xx in range(0,nk):
                tind = int(Onset[xx,comp])
#                if tind < self.nt-1:
#                    W[tind-taplen+1:tind+2,xx,comp] = tap
                index = int(np.floor(taplen/2))
                if tind < self.nt-1-index:
                    W[tind-taplen+index+1:tind+2+index,xx,comp] = tap
        
        # Use symmetry to construct complete far-offset window matrix
        # Watch out: For negative times the PS and SP components of the window are interchanged
        # I think the window is time-symmetric, also for PS/SP because it mutes GMM, not Tfs
        Wtmp = W.copy()
        W[1:nf-1,:,0] =  Wtmp[self.nt:nf-1:-1,:,0]   # Mirror times
#        W[1:nf-1,:,1] =  Wtmp[self.nt:nf-1:-1,:,2]   # Mirror times
#        W[1:nf-1,:,2] =  Wtmp[self.nt:nf-1:-1,:,1]   # Mirror times
        W[1:nf-1,:,1] =  Wtmp[self.nt:nf-1:-1,:,1]   # Mirror times
        W[1:nf-1,:,2] =  Wtmp[self.nt:nf-1:-1,:,2]   # Mirror times
        W[1:nf-1,:,3] =  Wtmp[self.nt:nf-1:-1,:,3]   # Mirror times
        W[:,nk:,:]    =  W[:,nk-2:0:-1,:]            # Mirror offsets

        # Delete global variables
        del xvals,yvals
            
        if plot == 1:
            self.Plot(W,tx=1,tvec=1,xvec=1,vmin=-1,vmax=1,title='Window matrix $\mathbf{W}$')
            self.Plot((1-W)*tP,tx=1,tvec=1,xvec=1,vmin=vmin,vmax=vmax,title='$(\mathbf{I} - \mathbf{W}) \mathbf{T}_{fs}^+$')
            self.Plot(W*tP,tx=1,tvec=1,xvec=1,vmin=vmin,vmax=vmax,title='$\mathbf{W} \mathbf{T}_{fs}^+$')
        
        return W,TP,tP
    
    def Marchenko_series(self,zF,R=None,F1P0=None,mask=None,fk_taplen=0,W=None,f0=10,vmin=None,vmax=None,K=0,plot=1,Gref=0,mul=1,conv=1):
        """
        Out = Marchenko_series(zF,R=None,F1P0=None,mask=None,fk_taplen=0,W=None,f0=10,vmin=None,vmax=None,K=0,plot=1,ComputeReferenceG=0,mul=1,conv=1)
        
        Evaluate K iterations of the Marchenko series. Input wavefieds can be given. However, it seems to be important to attenuate temporal 
        wrap-around effects to achieve convergence of the Marchenko series.
        
        Inputs:
            zF:         Focusing depth
            R:          (Optional) Reflection response in the F-Kx domain (nt x nr x 4) or (nf x nk x 4).
            F1P0:       (Optional) Initial focusing function in the F-Kx domain (nt x nr x 4) or (nf x nk x 4).
            mask:       (Optional) F-Kx mask (nt x nr x 4).
            fk_taplen:  Default fk_taplen=0. Taperlength (number of samples) for the F-Kx mask.
            W:          (Optional) Window for the Marchenko seris (nt x nr x 4).
            f0:         Default f0=10. Frequency for Ricker wavelet in Hz.
            vmin:       (Optional) Minimum clipping amplitude of the forward-scattered transmission response to pick the time window.
            vmax:       (Optional) Maximum clipping amplitude of the forward-scattered transmission response to pick the time window.
            K:          Default K=0. Number of iterations of the Marchenko series.
            plot:       Default plot=1. Set plot=1 to automatically plot the outputs. Set plot=0 to suppress automatic plotting.
            Gref:       Default Gref=0. Set Gref=1 to model reference Green's functions.
            mul:        Default mul=1. Set mul=1 to include internal multiples. Set mul=0 to exclude internal multiples.
            conv:       Default conv=1. Set conv=1 to include P/S conversions. Set conv=0 to exclude P/S conversions.
        
        Output:
            Out: Dictionary containing relfection response, retrieved Green's functions, F-Kx mask, time window, 
            focusing functions (initial downgoing, downgoing, and upgoing), modelled reference Green's functions.
            {'R':R,
            'GMP':GMP,
            'GMM':GMM,
            'mask':mask,
            'W':W,
            'F1P0':F1P0,
            'F1Pk':F1Pk,
            'F1Mk':F1Mk,
            'GMPref':GMPref,
            'GMMref':GMMref}
        
        
        """
        
        # Check if focusing depth has changed
        if zF != self.zF:
            self.zF = zF
            self.F1P = None
            self.F1M = None
            self.F1P_neps = None # Focusing funcion with negative eps
            self.F1M_neps = None # Focusing funcion with negative eps
            self.F1P0 = None
            
        nf = int(self.nt/2)+1 
        nk = int(self.nr/2)+1 
        
        # Make FKx mask if not given
        # Mask is determined by maximum velocity in the truncated medium
        if mask is None:

            print('(1/6) Compute $f$-$k_x$ mask.')
            
            # Determine maximum velocity of truncated medium
            self.Insert_layer(self.zF)
            N  = np.cumsum(self.dzvec).tolist().index(zF)   
            cm = max( np.array(self.cpvec[:N+1]).max() , np.array(self.csvec[:N+1]).max() )
            self.Remove_layer(self.zF)
            
            # Make mask
            fm   = -self.Kxvec()[0][0]*cm/2/np.pi # Maximum unaliased frequency
            mask = self.FKx_Mask(fm,cm,taplen=fk_taplen)[1]
           
        # Constant to dampen the temporal wrap-around    
        eps = 3/(self.dt*self.nt*0.5)
        
        # Model reflection data if not given
        # I recommend to dampen the temporal wrap-around!
        if R is None:
            print('(2/6) Compute reflection data $\mathbf{R}$.')
            R = mask*self.Layercel_kx_w(mul=mul,conv=conv,eps=eps,sort=1)[0]
            r = self.WKx2TX(R,eps=eps,norm='ortho',taperlen=30)
            R = np.fft.fft2(np.fft.ifftshift(r,(0,1)),s=None,axes=(0,1),norm='ortho')
        
        # Select positive f-kx elements and compute R dagger
        R = R[:nf,:nk,:]
        Rdag = self.My_T(R).conj()
          
        # Model F1P0 if not given
        if F1P0 is None:
            print('(3/6) Compute initial focusing function $\mathbf{F}_{1,0}^+$.')
            F1P0,_,TP = self.F1plus0(zF,eps=-eps)[1:4]
            f1P0 = self.WKx2TX(F1P0*mask,eps=-eps,norm='ortho',taperlen=30)
            F1P0 = np.fft.fft2(np.fft.ifftshift(f1P0,(0,1)),s=None,axes=(0,1),norm='ortho')[:nf,:nk,:]
            
        else:
            TP = None
            if F1P0.shape[0] == nf:
                F1P0 = self.Sort_kx_w(F1P0)
            F1P0 = F1P0*mask
            F1P0 = F1P0[:nf,:nk,:]
        
        # Make window if not given
        if W is None:
            print('(4/6) Pick time window $\mathbf{W}$.')
            W = self.Window(zF,Tfs=1,f0=f0,Tdata=TP,mask=mask,plot=plot,eps=eps,vmin=vmin,vmax=vmax)[0]
            
            print('check the below code')
            # W should mute GMM, i.e. compared to Tfs source and receiver are interchanged.
            W = self.My_T(W)
        
        # Time-reverse time window: Take into account that the PS/SP components are not symmertic in time!
        # I think the window is time-symmetric, also for PS/SP because it mutes GMM, not Tfs
#        Wrev=W.copy()
#        Wrev[:,:,1]=W[:,:,2] # just checking 
#        Wrev[:,:,2]=W[:,:,1] # just checking 
#        Wrev[1:,:,:]=Wrev[1:,:,:][::-1,:,:] 
        
        # Compute initial upgoing focusing function
        F1M0 = self.My_dot(R,F1P0)
        f1M0 = W*self.WKx2TX(self.Sort_kx_w(F1M0),norm='ortho')
        F1M0 = np.fft.fft2(np.fft.ifftshift(f1M0,(0,1)),s=None,axes=(0,1),norm='ortho')[:nf,:nk,:]
        
        # Update focusing function
        F1Pk = F1P0.copy()
        F1Mk = F1M0.copy()
        
        print('(5/6) Compute %d Marchenko iterations.'%(K))
            
        # Iterate 
        for k in range(0,K):
            
            print('\t Iteration (%d/%d).'%(k+1,K))
            
            MP = self.My_dot(Rdag,F1Mk)
            mP = W*self.WKx2TX(self.Sort_kx_w(MP),norm='ortho')
            MP = np.fft.fft2(np.fft.ifftshift(mP,(0,1)),s=None,axes=(0,1),norm='ortho')[:nf,:nk,:]
            
            dF1Mk = self.My_dot(R,MP)
            df1Mk = W*self.WKx2TX(self.Sort_kx_w(dF1Mk),norm='ortho')
            dF1Mk = np.fft.fft2(np.fft.ifftshift(df1Mk,(0,1)),s=None,axes=(0,1),norm='ortho')[:nf,:nk,:]
            
            F1Pk = F1P0 + MP
            F1Mk = F1M0 + dF1Mk
        
        # 1st Marchenko equation
        GMP = self.Sort_kx_w( self.My_dot(R,F1Pk) - F1Mk ) 
        
        # 2nd Marchenko equation
        GMM = self.Sort_kx_w( (self.My_dot(Rdag,F1Mk) - F1Pk).conj() ) 
        GMM[:,1:,:] = GMM[:,-1:0:-1,:]
        
        if Gref == 1:          
            print('(6/6) Compute reference Greens function $\mathbf{G}_{ref}^{-,+}$ and $\mathbf{G}_{ref}^{-,-}$.')
            Gs            = self.Gz2bound(zF,mul=mul,conv=conv,eps=eps)[0]
            GMPref,GMMref = Gs[2:4]
        
        # Compute negative frequencies and wavenumbers    
        R,F1Pk,F1Mk = self.Sort_kx_w(R,F1Pk,F1Mk)
        
        if plot == 1:
            
            # Plot Marchenko Green's functions
            gMM  = self.WKx2TX(GMM*mask,norm='ortho')
            gMMw = self.ConvolveRicker(gMM,f0)[0]
            vmin = - 10**(int(np.log10(61)))
            vmax = -vmin
            self.Plot(gMMw,tx=1,tvec=1,xvec=1,vmin=vmin,vmax=vmax,title='$\mathbf{G}_{K=%d}^{-,-}$'%(K))
            
            gMP = self.WKx2TX(GMP*mask,norm='ortho')
            gMPw = self.ConvolveRicker(gMP,f0)[0]
            self.Plot((1-W)*gMPw,tx=1,tvec=1,xvec=1,vmin=vmin,vmax=vmax,title='$(\mathbf{I} - \mathbf{W}) \mathbf{G}_{K=%d}^{-,+}$'%(K))
            
            # Plot focusing functions
            f1Pkw = self.ConvolveRicker(self.WKx2TX(F1Pk*mask,norm='ortho'),f0)[0]
            self.Plot(f1Pkw,tx=1,tvec=1,xvec=1,vmin=vmin,vmax=vmax,title='$\mathbf{F}_{1,K=%d}^{+}$'%(K))
            f1Mkw = self.ConvolveRicker(self.WKx2TX(F1Mk*mask,norm='ortho'),f0)[0]
            self.Plot(f1Mkw,tx=1,tvec=1,xvec=1,vmin=vmin,vmax=vmax,title='$\mathbf{F}_{1,K=%d}^{-}$'%(K))
            
            if Gref == 1:          
                
                # Plot reference Green's functions
                gMPref = self.WKx2TX(GMPref*mask,eps=eps,norm='ortho')
                gMPrefw = self.ConvolveRicker(gMPref,f0)[0]
                self.Plot(gMPrefw,tx=1,tvec=1,xvec=1,vmin=vmin,vmax=vmax,title='$\mathbf{G}_{ref}^{-,+}$')
                
                gMMref = self.WKx2TX(GMMref*mask,eps=eps,norm='ortho')
                gMMrefw = self.ConvolveRicker(gMMref,f0)[0]
                self.Plot(gMMrefw,tx=1,tvec=1,xvec=1,vmin=vmin,vmax=vmax,title='$\mathbf{G}_{ref}^{-,-}$')
                
                self.Plot((1-W)*gMPw - gMPrefw,tx=1,tvec=1,xvec=1,vmin=vmin,vmax=vmax,title='$(\mathbf{I} - \mathbf{W}) \mathbf{G}_{K=%d}^{-,+} - \mathbf{G}_{ref}^{-,+}$'%(K))
                self.Plot(gMMw - gMMrefw,tx=1,tvec=1,xvec=1,vmin=vmin,vmax=vmax,title='$\mathbf{G}_{K=%d}^{-,-} - \mathbf{G}_{ref}^{-,-}$'%(K))
            else:
                GMPref = 0
                GMMref = 0
        
        if Gref == 0:
            Out = {'R':R,'GMP':GMP,'GMM':GMM,'mask':mask,'W':W,'F1P0':F1P0,'F1Pk':F1Pk,'F1Mk':F1Mk}
        elif Gref == 1:
            Out = {'R':R,'GMP':GMP,'GMM':GMM,'mask':mask,'W':W,'F1P0':F1P0,'F1Pk':F1Pk,'F1Mk':F1Mk,'GMPref':GMPref,'GMMref':GMMref}
        return Out
    
    # Old version of FocusFunc1_mod() without option to compute piecewise
#    # An attempt to model the focusing function
#    def FocusFunc1_mod(self,zF=None,mul=1,conv=1,eps=None,sort=1):
#        """
#        R,T = Layercel_kx_w(mul,conv,nf,nk)
#        Compute reflection / transmission response for a single wavenumber and all positive frequencies as well as the highest negative frequency.
#        
#        Inputs:
#            mul:  Set mul=1 to model internal multiples.
#            conv: Set conv=1 to model P/S conversions.
#            eps:  Set eps to add an imaginary constant to the frequency: w -> w + 1j*eps. This reduces the wrap-around in time but one has to correct for this by multiplying the data by exp(eps*t) in the time domain.
#            sort: Set sort=1 (default) to get positive and negative frequencies and wavenumbers.
#        
#        Output:
#            RP: Reflection response from above (nf x 4), 1st element corresponds to zero frequency
#            TP: Transmission response from above (nf x 4), 1st element corresponds to zero frequency
#        """
#        
#        # Check if focusing depth has changed
#        if zF != self.zF:
#            self.zF = zF
#            self.F1P = None
#            self.F1M = None
#            self.F1P_neps = None # Focusing funcion with negative eps
#            self.F1M_neps = None # Focusing funcion with negative eps
#        
#        # Truncate medium
#        Trunc = self.Truncate()[-1]
#        
#        # Number of layers
#        N = np.size(Trunc.cpvec)
#        
#        # Number of positive frequency and positive wavenumber samples
#        nk = int(self.nr/2) + 1
#        nf = int(self.nt/2) + 1
#        
#        if eps is None:
#            eps = 3/(nf*self.dt)
#        
#        # Frequency and wavenumber meshgrids
#        Wfft,Kxfft = self.W_Kx_grid()[2:4]
#    
#        # Extract positive frequencies and wavenumbers
#        Wpos  = Wfft[0:nf,0:nk] + 1j*eps
#        Kxpos = Kxfft[0:nf,0:nk]
#        
#        # Propagation and scattering matrices of an infinitesimal layer without any contrast
#        
#        W = np.zeros((nf,nk,4),dtype=complex)
#        Winv = np.zeros((nf,nk,4),dtype=complex)
#        
#        RP = np.zeros((nf,nk,4),dtype=complex)
#        RM = np.zeros((nf,nk,4),dtype=complex)
#        
#        I = np.zeros((nf,nk,4),dtype=complex)
#        I[:,:,0] = 1
#        I[:,:,3] = 1
#        M1 = I.copy()
#        M2 = I.copy()
#        
#        # Here every frequency and every wavenumber component have an amplitude 
#        # equal to one. Hence, the total wavefield has a strength of sqrt(nt*nr)
#        # When an inverse fft (ifft2) is applied the wavefield is scaled by 
#        # 1/sqrt(nt*nr) hence in the time domain the wavefield has the an
#        # amplitude equal to one.
#        TP = I.copy()
#        TM = I.copy()
#        F1P = I.copy()
#        F1M = RP.copy()
#        
#        # Loop over N-1 interfaces
#        for n in range(0,N-1):
#            
#            dz1 = Trunc.dzvec[n]
#            
#            # Parameters of top layer
#            cp1 = Trunc.cpvec[n]
#            cs1 = Trunc.csvec[n]
#            ro1 = Trunc.rovec[n]
#        
#            # Parameters of bottom layer
#            cp2 = Trunc.cpvec[n+1]
#            cs2 = Trunc.csvec[n+1]
#            ro2 = Trunc.rovec[n+1]
#            
#            kzp = np.sqrt(Wpos**2/cp1**2-Kxpos**2) # kz for p-waves
#            kzs = np.sqrt(Wpos**2/cs1**2-Kxpos**2) # kz for s-waves
#            
#            W[:,:,0] = np.exp(1j*kzp*dz1)
#            W[:,:,3] = np.exp(1j*kzs*dz1)
#            
#            Winv = self.My_inv(W)
#            Winv = np.nan_to_num(Winv)
#            
#            
#            rP,tP,rM,tM,tPinv = self.RT_kx_w(cp1,cs1,ro1,cp2,cs2,ro2,Kxpos,Wpos,conv,nf,nk)
#    
#            if mul == 1:
#                M1inv = I - self.Mul_My_dot(RM,W,rP,W)
#                # Inverse of tmp
#                M1 = self.My_inv(M1inv)
#                
#                M2inv = I - self.Mul_My_dot(rP,W,RM,W)
#                # Inverse of tmp
#                M2 = self.My_inv(M2inv)
#                
#            # Update reflection / transmission responses
#            RP = RP + self.Mul_My_dot(TM,W,rP,W,M1,TP)
#            RM = rM + self.Mul_My_dot(tP,W,RM,W,M2,tM)
#            TP = self.Mul_My_dot(tP,W,M1,TP)
#            TM = self.Mul_My_dot(TM,W,M2,tM)   
#            F1P = self.Mul_My_dot(F1P,M1inv,Winv,tPinv)
#        
#        
#        
#        # At the end compute F1M
#        F1M = self.My_dot(RP,F1P)
#                
#    
#        # The highest negative frequency and highest negative wavenumber components are real-valued
#        RP[nf-1,:] = RP[nf-1,:].real
#        TP[nf-1,:] = TP[nf-1,:].real
#        F1P[nf-1,:] = F1P[nf-1,:].real
#        F1M[nf-1,:] = F1M[nf-1,:].real
#        RP[:,nk-1] = RP[:,nk-1].real
#        TP[:,nk-1] = TP[:,nk-1].real
#        F1P[:,nk-1] = F1P[:,nk-1].real
#        F1M[:,nk-1] = F1M[:,nk-1].real
#        
#        
#        # Remove Nans and values above let's say 10
#        RP = np.nan_to_num(RP)
#        TP = np.nan_to_num(TP)
#        F1P = np.nan_to_num(F1P)
#        F1M = np.nan_to_num(F1M)
#        
#        # Conjugate wavefields
#        RP = RP.conj()
#        TP = TP.conj()
#        F1P = F1P.conj()
#        F1M = F1M.conj()
#        
#        # Write full reflection and transmission matrices
#        if sort == 1:
##            Rfull,Tfull = self.Sort_RT_kx_w(RP,TP)
#            RPfull,TPfull,F1Pfull,F1Mfull = self.Sort_kx_w(RP,TP,F1P,F1M)
#            return RPfull,TPfull,F1Pfull,F1Mfull,RP,TP,F1P,F1M
#        
#        return RP,TP,F1P,F1M