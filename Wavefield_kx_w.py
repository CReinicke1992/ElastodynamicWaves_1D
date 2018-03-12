#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:24:56 2017

@author: christianreini
"""

import numpy as np
import matplotlib.pylab as plt

class Wavefield_kx_w:
    """
    Wavefield_kx_w
    
    Describes wavefields which have multiple frequencies and wavenumbers as well as four elastic components PP, PS, SP, SS.
    
    Variables:
        nt: Number of time/frequency samples
        dt: Duration per time sample in seconds
        nr: Number of space/wavenumber samples
        dx: Distance per space samples in metres
        
    Data sorting: 
        nt x nr x 4
    """
    
    def __init__(self,nt,dt,nr,dx):
        self.nt = int(nt)
        self.dt = dt
        self.nr = int(nr)
        self.dx = dx
        self.author = "Christian Reinicke"
       
    # Frequency sampling    
    def Dw(self):
        """
        dw = Dw()
        
        Frequency sampling in radian.
        """
        return 2*np.pi/(self.dt*self.nt)
      
    # Frequency vector    
    def Wvec(self):
        """
        wvec, wvecfft = Wvec()
        
        Make a vector with all frequencies:
            
            (1) sorted from -wmax to +wmax.
            (2) sorted according to fft standard (ifftshift of 1).
        """
        dw = 2*np.pi/(self.dt*self.nt)
        wvec = dw*np.arange(-self.nt/2,self.nt/2)
        wvecfft = np.fft.ifftshift(wvec)
        return wvec,wvecfft
    
    # Wavenumber sampling
    def Dkx(self):
        """
        dkx = Dkx()
        
        Wavenumber sampling per metre.
        """
        return 2*np.pi/(self.dx*self.nr)
    
    # Wavenumber vector    
    def Kxvec(self):
        """
        kxvec, kxvecfft = Kxvec()
        
        Make a vector with all wavenumbers:
            
            (1) sorted from -kxmax to +kxmax.
            (2) sorted according to fft standard (ifftshift of 1).
        """
        dkx = 2*np.pi/(self.dx*self.nr)
        kxvec = dkx*np.arange(-self.nr/2,self.nr/2)
        kxvecfft = np.fft.ifftshift(kxvec)
        return kxvec,kxvecfft
    
    # Space vector
    def Xvec(self):
        """
        xvec, xvecfft = Xvec()
        
        Make a vector with all x positions:
            
            (1) sorted from -xmax to +xmax.
            (2) sorted according to fft standard (ifftshift of 1).
        """
        xvec = self.dx*np.arange(-self.nr/2,self.nr/2)
        xvecfft = np.fft.ifftshift(xvec)
        return xvec, xvecfft
    
    # Time vector
    def Tvec(self):
        """
        tvec, tvecfft = Tvec()
        
        Make a vector with all times:
            
            (1) sorted from -tmax to +tmax.
            (2) sorted according to fft standard (ifftshift of 1).
        """
        tvec = self.dt*np.arange(-self.nt/2,self.nt/2)
        tvecfft = np.fft.ifftshift(tvec)
        return tvec, tvecfft
    
    # Make a 2D meshgrid in w-kx-domain
    def W_Kx_grid(self):
        """
        W,Kx,Wfft,Kxfft = W_Kx_grid()
        
        Outputs:
            W:     Meshgrid with a frequency vector wvec along the 1st dimension and nr copies of it along the 2nd dimension.
            Kx:    Meshgrid with a wavenumber vector kxvec along the 2nd dimension and nt copies of it along the 1st dimension.
            Wfft:  Shifted version of W sorted according to fft standard (ifftshift of W).
            Kxfft: Shifted version of Kx sorted according to fft standard (ifftshift of Kx).
        """
        wvec,wvecfft   = self.Wvec()
        kxvec,kxvecfft = self.Kxvec()
        Kx,W = np.meshgrid(kxvec,wvec)
        Kxfft,Wfft = np.meshgrid(kxvecfft,wvecfft)
        return W,Kx,Wfft,Kxfft
    
    # Make a 2D meshgrid in t-x-domain
    def T_X_grid(self):
        """
        Tgrid,Xgrid,Tgridfft,Xgridfft = T_X_grid()
        
        Outputs:
            Tgrid:    Meshgrid with a time vector tvec along the 1st dimension and nr copies of it along the 2nd dimension.
            Xgrid:    Meshgrid with a space vector xvec along the 2nd dimension and nt copies of it along the 1st dimension.
            Tgridfft: Shifted version of Tgrid sorted according to fft standard (ifftshift of Tgrid).
            Xgridfft: Shifted version of Xgrid sorted according to fft standard (ifftshift of Xgrid).
        """
        tvec,tvecfft   = self.Tvec()
        xvec,xvecfft = self.Xvec()
        Xgrid,Tgrid = np.meshgrid(xvec,tvec)
        Xgridfft,Tgridfft = np.meshgrid(xvecfft,tvecfft)
        return Tgrid,Xgrid,Tgridfft,Xgridfft
    
    # Taper
    def Taper(self,dim,start,end,taperlen):
        """
        taper,taper_full = Taper(dim,start,end,taperlen)
        
        Construct an array ones of the dimensions nt x nr x 4.
        
        Taper the start (if start=1) and the end (if end=1) of the dimension dim with a cosine of length taperlen.
        
        Variables:
            dim:      Dimension to be tapered
            start:    If start = 1 taper the start of dimension dim
            end:      If end = 1 taper the end of dimension dim
            taperlen: Length of the cosine taper
            
        Outputs:
            taper:      Taper matrix, nt x nr
            taper_full: Taper matrix copied to all four elastic components,
                        nt x nr x 4
        
        """
        tap = np.cos(np.linspace(0,np.pi/2,taperlen+1))
        taper = np.ones((self.nt,self.nr))
        if dim == 0:
            if start == 1:
                taper[0:taperlen+1,:] = np.tile(tap[::-1],[self.nr,1]).T
            if end == 1:
                taper[-taperlen-1:,:] = np.tile(tap,[self.nr,1]).T
        elif dim == 1:
            if start == 1:
                taper[:,0:taperlen+1] = np.tile(tap[::-1],[self.nt,1])
            if end == 1:
                taper[:,-taperlen-1:] = np.tile(tap,[self.nt,1])
                
        taper_full = np.dstack((taper,taper,taper,taper))
        return taper,taper_full
    
    # Ricker wavelet (nt x nr) and (nt x nr x 4)
    def Ricker(self,f0,norm=0):
        """
        wav,wavfft,wav_full,wavfft_full = Ricker(f0,norm=0)
        
        Make a Ricker wavelet 
            
        Inputs:
            f0:              Central frequency f in hertz (not omega!)
            norm (optional): Set to one to normalise the wavelet to norm(wav)=1. Otherwise the wavelet is scaled such that at time zero the amplitude equals 1. In this configuration the norm of the wavelet is NOT equal to 1.
            
        Outputs:
            wav:         Ricker wavelet (1) sorted from -tmax to +tmax. Dimension nt x nr 
            wavfft:      Ricker wavelet (2) sorted according to fft standard (ifftshift of 1). Dimension nt x nr
            wav_full:    Copy of wav for each elastic component. Dimension nt x nr x 4
            wavfft_full: Copy of wavfft for each elastic component. Dimension nt x nr x 4
            
            
        The expression for the Ricker wavelet is according to Wikipedia with sigma = 1/(2pif0). Wikipedia uses an additional scaling factor which I omitted here.
        """
        
        # Define sigma which is a function of the central frequency
        sig = 1/(np.sqrt(2)*np.pi*f0) 
        tvec = self.dt*np.arange(-self.nt/2,self.nt/2)
        # I omit the scaling factor fac1 because I don't see a purpose of it (except Wikipedia's definition of a Ricker wavelet)
        #fac1 = 2/(np.sqrt(3*sig)*np.power(np.pi,0.25))
        fac2 = 1 - (tvec/sig)**2
        fac3 = np.exp(-0.5*(tvec/sig)**2)
        wav = fac2*fac3
        #wav = fac1*fac2*fac3
        # Check how norm can be made optional and set to norm=0 as default!
        if norm ==1:
            wav = wav/np.linalg.norm(wav)
        wavfft = np.fft.ifftshift(wav)
        
        # Copy wavelet for each spatial position
        wav = np.tile(wav,[self.nr,1]).T
        wavfft = np.tile(wavfft,[self.nr,1]).T
        
        # Copy wavelet to each elastic component
        wav_full = np.dstack((wav,wav,wav,wav))
        wavfft_full = np.dstack((wavfft,wavfft,wavfft,wavfft))
        
        return wav,wavfft,wav_full,wavfft_full

    # 1D Ricker wavelet (nt x nr) and (nt x nr x 4)
    def Ricker1D(self,f0,norm=0):
        """
        wav,wavfft = Ricker(f0,norm=0)
        
        Make a 1D Ricker wavelet 
            
        Inputs:
            f0:              Central frequency f in hertz (not omega!)
            norm (optional): Set to one to normalise the wavelet to norm(wav)=1. Otherwise the wavelet is scaled such that at time zero the amplitude equals 1. In this configuration the norm of the wavelet is NOT equal to 1.
            
        Outputs:
            wav:         Ricker wavelet (1) sorted from -tmax to +tmax. Dimension nt 
            wavfft:      Ricker wavelet (2) sorted according to fft standard (ifftshift of 1). Dimension nt
            
        The expression for the Ricker wavelet is according to Wikipedia with sigma = 1/(2pif0). Wikipedia uses an additional scaling factor which I omitted here.
        """
        
        # Define sigma which is a function of the central frequency
        sig = 1/(np.sqrt(2)*np.pi*f0) 
        tvec = self.dt*np.arange(-self.nt/2,self.nt/2)
        # I omit the scaling factor fac1 because I don't see a purpose of it (except Wikipedia's definition of a Ricker wavelet)
        #fac1 = 2/(np.sqrt(3*sig)*np.power(np.pi,0.25))
        fac2 = 1 - (tvec/sig)**2
        fac3 = np.exp(-0.5*(tvec/sig)**2)
        wav = fac2*fac3
        #wav = fac1*fac2*fac3
        # Check how norm can be made optional and set to norm=0 as default!
        if norm ==1:
            wav = wav/np.linalg.norm(wav)
        wavfft = np.fft.ifftshift(wav)
        
        return wav,wavfft            
   
    def Convolve(self,field1,field2,dim):
        """
        con,confft = Convolve(self,field1,field2,dim)
        
        Convolve field1 with field2 along the dimensions dim.
        
        Inputs:
            field1: Wavefield (nt x nr) or (nt x nr x 4)
            field1: Wavefield (nt x nr) or (nt x nr x 4)
            dim:    Choose dim=0 or dim=1 or dim=(0,1)
            
        Outputs:
            con:    Convolution result (1) sorted from -tmax to +tmax and -xmax to +xmax. Dimension nt x nr (x4)
            confft: Convolution result (2) sorted according to fft standard (ifftshift of 1). Dimension nt x nr (x4)
        """
        
        # 1D concolution
        if isinstance(dim,int):
            F1 = np.fft.fft(field1,n=None,axis=dim)
            F2 = np.fft.fft(field2,n=None,axis=dim)
            confft = np.fft.ifft(F1*F2,n=None,axis=dim).real
            con = np.fft.fftshift(confft,axes=dim)
          
        # 2D concolution
        elif isinstance(dim,tuple):
        #if dim ==3:    
            F1 = np.fft.fft2(field1,s=None,axes=dim)
            F2 = np.fft.fft2(field2,s=None,axes=dim)
            confft = np.fft.ifft2(F1*F2,s=None,axes=dim).real
            con = np.fft.fftshift(confft,axes=dim)
            
        return con,confft
    
    def ConvolveRicker(self,field,f0):
        """
        con,confft = ConvolveRicker(self,field,f0)
        
        Convolve field with a Ricker wavelet.
        
        Inputs:
            field: Wavefield (nt x nr) or (nt x nr x4)
            f0:    Central frequency of Ricker wavelet.
            
        Outputs:
            con:    Convolution result (1) sorted from -tmax to +tmax and -xmax to +xmax. Dimension nt x nr (x4)
            confft: Convolution result (2) sorted according to fft standard (ifftshift of 1). Dimension nt x nr (x4)
        """
        if field.ndim == 2:
            wav = self.Ricker(f0)[0]
        elif field.ndim == 3:
            wav = self.Ricker(f0)[2]
        
        con,confft = self.Convolve(field,wav,0)
        
        return con,confft
    
    # Temporal gain to correct for complex-valued frequency omega
    def Gain(self,eps=0,taperlen=0):
        """
        gain = Gain(eps)
        
        Input:
            eps:      Gain factor.
            taperlen: Length of cosine taper.
        
        Output: 
            gain: Dimension (nt x nr x 4) with a gain exp(eps*t) along the 1st dimension.
        """
        
        if eps == 0:
            gain = np.ones((self.nt,self.nr))
            gain_full = np.ones((self.nt,self.nr,4))
            return  gain,gain_full
        
        tvec = self.Tvec()[0]
        xvec = np.ones(self.nr,dtype=complex)
        tgrid = np.meshgrid(xvec,tvec)[1]
        vec = np.ones(4,dtype=complex)
        tgrid_full = np.meshgrid(xvec,tvec,vec)[1]
        
        taper,taperfull = self.Taper(dim=0,start=1,end=1,taperlen=taperlen)
        
        gain = np.exp(tgrid*eps)*taper
        gain_full = np.exp(tgrid_full*eps)*taperfull
        
        return gain,gain_full
    
    # Correct for complex-valued frequency and return data to W-Kx domain
    def Correct_eps_kx_w(self,array,eps=0,taperlen=0):
        """
        array = Correct_eps_kx_w(array,eps=0,taperlen=0)
        
        (1) The input wavefield is transformed from the w-kx domain to the t-x domain.
        (2) A gain exp(eps*t) is a applied to correct for the complex-valued frequency/.
        (3) The wavefield is back-transformed to the w-kx domain.
        
        Inputs:
            array:    Wavefield in w-kx domain, which was modelled with a complex-valued frequency w' = w + 1j*eps
            eps:      Constant that was added to the frequency during modelling w' = w + 1j*eps.
            taperlen: After applying a gain in t-x domain the highest negative and postitibe time sample have a strong amplitude difference which can lead to fft artefacts. This can be reduced by using a cosine taper.
       
        Outputs:
            array: Wavefield in the w-kx domain with real-valued frequencies w.
        
        """
        if eps == 0:
            return array
        
        array_tx = np.fft.ifft2(array,s=None,axes=(0,1)).real
        
        if np.ndim(array) == 2:
            gain = self.Gain(eps,taperlen)[0]
        elif np.ndim(array) == 3:
            gain = self.Gain(eps,taperlen)[1]
        
        gain = np.fft.ifftshift(gain,(0,1))
        array = np.fft.fft2(array_tx*gain,s=None,axes=(0,1))
        
        return array
    
    # Transform wavefield from w-kx-domain to t-x-domain
    def WKx2TX(self,array,eps=0,taperlen=0,threshold=None,norm=None):
        """
        array_tx = WKx2TX(array,eps=0,taperlen=0)
        
        Input:
            eps:      Gain factor.
            taperlen: Length of cosine taper.
        
        Output: 
            array_tx: Array in tx, fftsift and gain are applied.
        """
        if threshold != None:
            array[abs(array)>threshold] = threshold
        
        if norm is None:
            array_tx = np.fft.ifft2(array,s=None,axes=(0,1)).real
        elif norm == 'ortho':
            array_tx = np.fft.ifft2(array,s=None,axes=(0,1),norm='ortho').real
        array_tx = np.fft.fftshift(array_tx,(0,1))
        
        if np.ndim(array) == 2:
            gain = self.Gain(eps,taperlen)[0]
        elif np.ndim(array) == 3:
            gain = self.Gain(eps,taperlen)[1]
            
        return array_tx*gain
    
    # Martix multiplication of each w-kx-component of an elastic wavefield (with PP,PS,SP,SS)
    def My_dot(self,B,C):
        """
        A = My_dot(B,C)
        
        Input:
            B: Array of dimensions nf x nk x 4
            C: Array of dimensions nf x nk x 4
            
        Output:
            A: Matrix product between B and C for every frequency and wavenumber
            
        """
        A = np.zeros_like(B,dtype=complex)
        A[:,:,0] = B[:,:,0]*C[:,:,0] + B[:,:,1]*C[:,:,2]
        A[:,:,1] = B[:,:,0]*C[:,:,1] + B[:,:,1]*C[:,:,3]
        A[:,:,2] = B[:,:,2]*C[:,:,0] + B[:,:,3]*C[:,:,2]
        A[:,:,3] = B[:,:,2]*C[:,:,1] + B[:,:,3]*C[:,:,3]
        return A     
    
    # Multiple martix multiplication of each w-kx-component of an elastic wavefield (with PP,PS,SP,SS)
    def Mul_My_dot(self,*args):
        """
        B = Mul_My_dot(*args)
        
        Mul_My_dot applies the function My_dot to a sequence of input arrays.
        """
        B = args[0]
        for C in args[1:]:
            B = self.My_dot(B,C)
        return B
    
    # Martix inversion of each w-kx-component of an elastic wavefield (with PP,PS,SP,SS)
    def My_inv(self,A):
        """
        Ainv = My_inv(A)
        
        Input:
            A: Array of dimensions (nf x nk x 4)
            
        Output:
            Ainv: For every w-kx element sort A in a 2x2 matrix, invert the 2x2 matrix and sort the result in an array like A (nf x nk x 4)
        """
        Ainv = A.copy()
        Ainv[:,:,0] =  A[:,:,3]
        Ainv[:,:,1] = -A[:,:,1]
        Ainv[:,:,2] = -A[:,:,2]
        Ainv[:,:,3] =  A[:,:,0]
        det = A[:,:,0]*A[:,:,3] - A[:,:,1]*A[:,:,2]
        det = np.dstack((det,det,det,det))
        Ainv = Ainv/det
        return Ainv
    
    def My_T(self,A):
        """
        AT = My_T(A)
        
        Input:
            A: Array of dimensions (nf x nk x 4)
            
        Output:
            AT: For every w-kx element sort A in a 2x2 matrix, transpose the 2x2 matrix and sort the result in an array like A (nf x nk x 4)
        """
        AT = A.copy()
        AT[:,:,1] = A[:,:,2]
        AT[:,:,2] = A[:,:,1]
        return AT
    
    # F-Kx mask
    def FKx_Mask(self,fmax,cmin,taplen=0,array=None):
        """
        mask,maskfull,array = FKx_Mask(fmax,cmin,taplen=0,array=None)
        
        Construct an F-Kx mask which only passes frequencies below fmax (including fmax) and signal with a velocity above cmin. The edge of the mask is tapered with a cosine taper.
        
        Inputs:
            fmax:   Cut-off frequency in hertz. (wmax = 2pi fmax).
            cmin:   Minimum velocity that is passed. (metre/second)
            taplen: Length of cosine taper at the edge of the f-k mask. (length in sample numbers)
            array:  (Optional). If an array is given the F-Kx mask is applied to the array.
            
        Outputs:
            mask:     F-Kx mask (nt x nr)
            maskfull: F-Kx mask copied for each elastic component (nt x nr x 4)
            array:    F-Kx mask applied to the input array.
       
        """
        
        # Preallocate mask
        mask = np.zeros((self.nt,self.nr))
        
        # Number of positive frequency/wavenumber samples
        nf = int(self.nt/2)+1
        nk = int(self.nr/2)+1
        
        # Index of cut-off frequency
        ind = int(2*np.pi*fmax/self.Dw())
        
        # Taper
        tap = np.cos(np.linspace(0,np.pi/2,taplen+1))
        
        # Loop over passed frequencies
        for ff in range(0,ind+1):
            
            # Determine maximum wavenumber for each frequency
            kmax = ff*self.Dw()/cmin
            cut = int(kmax/self.Dkx())
            
            # These variables are necessary when the edge of the mask reaches the edge of the array
            if cut >= nk:
                #cut2 = cut - nk
                cut = nk
                
            # Choose passed wavenumbers
            mask[ff,:cut] = 1
            
            # Taper in kx direction
            if (cut > taplen) and (cut < nk):
                mask[ff,cut-taplen:cut+1] = tap
            # I commented the below elif to avoid non-zero values at kxmax
#            elif  (cut == nk) and (cut2 <= taplen):
#                mask[ff,cut-taplen+cut2:cut] = tap[:-cut2-1]
            elif  (cut == nk):
                mask[ff,cut-taplen:cut+1] = tap
            elif cut <= taplen:
                mask[ff,:cut+1] = tap[-cut-1:]
        
        # Taper in f direction
        mask[ind-taplen:ind+1,:] = mask[ind-taplen:ind+1,:]*np.tile(tap,(self.nr,1)).T
        
        # Copy mask for negative wavenumbers and frequencies
        mask[:,nk:] = mask[:,nk-2:0:-1]
        mask[nf:,:] = mask[nf-2:0:-1,:]

        maskfull = np.dstack((mask,mask,mask,mask))
        
        if array is None:
            return mask,maskfull
        
        # If an array is input, apply F-Kx mask to the array
        
        if array.ndim == 2:
            array = array*mask
        elif array.ndim == 3:
            array = array*maskfull
            
        return mask,maskfull,array
    
    def Pick_FKx_Mask(self,array,vmin=0,vmax=2,taplen=0):
        """
        mask,cm,fm = Pick_FKx_Mask(array,vmin=0,vmax=2,taplen=0)
        
        Inputs: 
            array:  Input data that for which the F-Kx mask is constructed
            vmin:   Optional set minimum z-axis value
            vmax:   Optional set maximum z-axis value
            taplen: Set lenght for taper along the edge of the F-Kx mask
                            
        Outputs:
            mask: F-Kx mask, identical shape as array
            cm:   List of minimum signal velocities for each elastic component
            fm:   Maximum frequency (identical for each elastic component)
        """
        
        global cmin,fmax
        
        cmin = []
        fmax = []
        
        # Function to 
        #   compute minimum signal velocity
        #   pick maximum unaliased frequency
        def onclick(event):
            cmin.append(event.ydata/event.xdata)
            fmax.append(event.ydata/(2*np.pi))
        
        # Determine whether array has format (nt x nr x 4) or (nt x nr)
        if array.ndim == 2:
            N = 1
        elif array.ndim ==3:
            N = 4
            
        # Iterate over elastic components to pick the fk mask    
        for comp in range(0,N):
            
            # User instruction
            title = "1. Pick edge of propagating wavefield. \n2. Pick maximum non-aliased frequency. \n"
    
            # Plot one elastic component
            if N == 1:
                fig = self.Plot(array,wkx=1,wvec=1,kxvec=1,title=title,vmin=vmin,vmax=vmax)
            else:
                fig = self.Plot(array[:,:,comp],wkx=1,wvec=1,kxvec=1,title=title,vmin=vmin,vmax=vmax)
            
            # Connect figure to the function onlick 
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
        
            # Wait until two picks were done
            while len(cmin) < 2*(comp+1):
                plt.pause(1)
            
            # Disconnect and close figure
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
            
        # Construct F-Kx mask for each elastic component    
        if N == 1:
            cm=cmin[0]
            fm=fmax[1]
            mask = self.FKx_Mask(fmax=fm,taplen=taplen,cmin=cm)[0]
        else:
            cm=cmin[0:-1:2]
            fm=fmax[1:-1:2]
            fm = np.array(fm).min()
            mask = np.zeros((self.nt,self.nr,N))
            for comp in range(0,N):
                mask[:,:,comp] = self.FKx_Mask(fmax=fm,taplen=taplen,cmin=cm[comp])[0]
    
        # Delete global variables
        del cmin,fmax
            
        return mask,cm,fm
        
        
    # Plot wavefield
    def Plot(self,field,tx=0,wkx=0,tvec=0,xvec=0,wvec=0,kxvec=0,cmap=1,title="",kill_NaN=1,vmin=None,vmax=None):
        """
        Plot(self,field,tx=0,wkx=0,tvec=0,xvec=0,wvec=0,kxvec=0,title="")
        
        Plot a (nt x nr) or (nt x nr x 4) wavefield. 
        
        Inputs:
            field:    Wavefield (nt x nr) or (nt x nr x 4).  
            tx:       Set tx=1 if the wavefield is in the t-x domain (should be real-valued).
            wkx:      Set wkx=1 if the wavefield is in the w-kx domain (can be complex-valued). field will be fftshifted.
            tvex:     Set tvec=1 to get a time axis.
            xvex:     Set xvec=1 to get a space axis.
            wvex:     Set wvec=1 to get a frequency axis.
            kxvex:    Set kxvec=1 to get a wavenumber axis.
            cmap:     Set cmap=1 to get a colorbar.
            title:    Set a title by inserting a string.
            kill_NaN: Set kill_NaN=1 to set NaNs to zero and infs to high values.
            
        Outputs:
            Option 1: If field has dimensions nt x nr a image is plotted.
            Option 2: If field has dimensions nt x nr x 4 a image with four subplots is plotted.
            
        """
        
        if kill_NaN == 1:
            field[np.isnan(field)==True] = 0 
        
        ex1min = 0
        ex1max = self.nt
        ex2min = 0
        ex2max = self.nr
        mycmap = None
        xlab = ""
        ylab = ""
        
        if tx == 1:
            mycmap = 'seismic'
            if vmin is None:
                myvmax = field.max()
                myvmin = -myvmax    
            else:
                myvmax = vmax
                myvmin = vmin
            # The field should be real-valued. I do not automatically take the real part of the field to make it easier to spot errors.
        if wkx == 1:
            mycmap = 'hot'
            field = np.abs(field)
            field = np.fft.fftshift(field,(0,1))
            if vmin is None:
                myvmin = field.min()    
                myvmax = field.max()  
            else:
                myvmax = vmax
                myvmin = vmin
        if tvec == 1:
            vec = self.Tvec()[0]
            ex1min = vec.max()
            ex1max = vec.min()
            ylab = 'Time (s)'
        if xvec == 1:
            vec = self.Xvec()[0]
            ex2min = vec.min()
            ex2max = vec.max()
            xlab = 'Offset (m)'
        if wvec == 1:
            vec = self.Wvec()[0]
            ex1min = vec.max()
            ex1max = vec.min()
            ylab = 'Circular frequency ($\mathrm{s}^{-1}$)'
        if kxvec == 1:
            vec = self.Kxvec()[0]
            ex2min = vec.min()
            ex2max = vec.max()
            xlab = 'Horizontal wavenumber ($m^{-1}$)'
        
        
        # Plot one elastic component
        if field.ndim == 2:
            
            fig = plt.figure()
            plt.imshow(field,cmap=mycmap,extent=[ex2min,ex2max,ex1min,ex1max], aspect=np.abs(ex2max/ex1max), vmin=myvmin, vmax=myvmax)
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.title(title)
            
        # Plot four elastic components    
        elif field.ndim == 3:
            
            fig = plt.figure()
            ax0 = plt.subplot2grid((2, 2), (0, 0),colspan=1)
            ax1 = plt.subplot2grid((2, 2), (0, 1),colspan=1)
            ax2 = plt.subplot2grid((2, 2), (1, 0),colspan=1)
            ax3 = plt.subplot2grid((2, 2), (1, 1),colspan=1)
            #ax4 = plt.subplot2grid((2, 3), (0, 2),colspan=1,rowspan=3)
            
            im0 = ax0.imshow(field[:,:,0],cmap=mycmap,extent=[ex2min,ex2max,ex1min,ex1max], aspect=np.abs(ex2max/ex1max), vmin=myvmin, vmax=myvmax)
            im1 = ax1.imshow(field[:,:,1],cmap=mycmap,extent=[ex2min,ex2max,ex1min,ex1max], aspect=np.abs(ex2max/ex1max), vmin=myvmin, vmax=myvmax)
            im2 = ax2.imshow(field[:,:,2],cmap=mycmap,extent=[ex2min,ex2max,ex1min,ex1max], aspect=np.abs(ex2max/ex1max), vmin=myvmin, vmax=myvmax)
            im3 = ax3.imshow(field[:,:,3],cmap=mycmap,extent=[ex2min,ex2max,ex1min,ex1max], aspect=np.abs(ex2max/ex1max), vmin=myvmin, vmax=myvmax)
            
            ax0.set_xlabel(xlab)
            ax1.set_xlabel(xlab)
            ax2.set_xlabel(xlab)
            ax3.set_xlabel(xlab)
            
            ax0.set_ylabel(ylab)
            ax1.set_ylabel(ylab)
            ax2.set_ylabel(ylab)
            ax3.set_ylabel(ylab)
            
            
            ax0.set_title("PP")
            ax1.set_title("PS")
            ax2.set_title("SP")
            ax3.set_title("SS")
            
            if cmap == 1:
                fig.colorbar(im0, ax=ax0)
                fig.colorbar(im1, ax=ax1)
                fig.colorbar(im2, ax=ax2)
                fig.colorbar(im3, ax=ax3)
            
            plt.suptitle(title)
            
        return fig
        
    
    
    
    
    
    
    
    
    