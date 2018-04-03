#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:07:23 2017

@author: christianreini
"""

import numpy as np
import matplotlib.pyplot as plt

class Wavefield_p_w:
    """
    Wavefield_p_w
    
    Describes wavefields which have multiple frequencies and a single ray-parameter as well as four elastic components PP, PS, SP, SS.
    
    Variables:
        nt:      Number of time/frequency samples
        dt:      Duration per time sample in seconds
        nr:      Number of space samples (only to produce 1.5D plots)
        dx:      Distance per space samples in metres
        nf:      Number of time samples divided by 2 plus 1.
        nr:      Number of space samples divided by 2 plus 1.
        verbose: Set verbose=1 to gt some feedback about processes.
        
    Data sorting: 
        nt x 4
    """
    
    def __init__(self,nt,dt,nr,dx):
        self.nt = int(nt)
        self.dt = dt
        self.nr = int(nr)
        self.dx = dx
        self.nf = int(self.nt/2) + 1
        self.nk = int(self.nr/2) + 1
        self.author = "Christian Reinicke"
        self.verbose = 0
       
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
        wvec = np.zeros((self.nt,1))
        wvec[:,0] = dw*np.arange(-self.nt/2,self.nt/2)
        wvecfft = np.fft.ifftshift(wvec)
        return wvec,wvecfft
    
    # Space vector
    def Xvec(self):
        """
        xvec, xvecfft = Xvec()
        
        Make a vector with all x positions:
            
            (1) sorted from -xmax to +xmax.
            (2) sorted according to fft standard (ifftshift of 1).
        """
        xvec = np.zeros((self.nr,1))
        xvec[:,0] = self.dx*np.arange(-self.nr/2,self.nr/2)
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
        tvec = np.zeros((self.nt,1))
        tvec[:,0] = self.dt*np.arange(-self.nt/2,self.nt/2)
        tvecfft = np.fft.ifftshift(tvec)
        return tvec, tvecfft
    
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
    
    # Make a 2D meshgrid in w-x-domain
    def W_X_grid(self):
        """
        W,X,Wfft,Xfft = W_X_grid()
        
        Outputs:
            W:     Meshgrid with a frequency vector wvec along the 1st dimension and nr copies of it along the 2nd dimension.
            X:     Meshgrid with a space vector kxvec along the 2nd dimension and nt copies of it along the 1st dimension.
            Wfft:  Shifted version of W sorted according to fft standard (ifftshift of W).
            Kxfft: Shifted version of X sorted according to fft standard (ifftshift of X).
        """
        wvec,wvecfft   = self.Wvec()
        xvec,xvecfft = self.Xvec()
        X,W = np.meshgrid(xvec,wvec)
        Xfft,Wfft = np.meshgrid(xvecfft,wvecfft)
        return W,X,Wfft,Xfft
    
    # Taper
    def Taper(self,start,end,taperlen):
        """
        taper,taper_full = Taper(start,end,taperlen)
        
        Construct an array ones of the dimensions nt x 4.
        
        Taper the start (if start=1) and the end (if end=1) of the dimension dim with a cosine of length taperlen.
        
        Variables:
            start:    If start = 1 taper the start of dimension dim
            end:      If end = 1 taper the end of dimension dim
            taperlen: Length of the cosine taper
            
        Outputs:
            taper:      Taper matrix, nt x 1
            taper_full: Taper matrix copied to all four elastic components,
                        nt x 4
        
        """
        tap = np.cos(np.linspace(0,np.pi/2,taperlen+1))
        taper = np.ones((self.nt,1))

        if start == 1:
            taper[0:taperlen+1,0] = tap[::-1]
        if end == 1:
            taper[-taperlen-1:,0] = tap
                
        taper_full = np.tile(taper,(1,4))
        return taper,taper_full
    
    # Ricker wavelet (nt x 1) and (nt x 4)
    def Ricker(self,f0,norm=0):
        """
        wav,wavfft,wav_full,wavfft_full = Ricker(f0,norm=0)
        
        Make a Ricker wavelet 
            
        Inputs:
            f0:              Central frequency f in hertz (not omega!)
            norm (optional): Set to one to normalise the wavelet to norm(wav)=1. Otherwise the wavelet is scaled such that at time zero the amplitude equals 1. In this configuration the norm of the wavelet is NOT equal to 1.
            
        Outputs:
            wav:         Ricker wavelet (1) sorted from -tmax to +tmax. Dimension nt x 1 
            wavfft:      Ricker wavelet (2) sorted according to fft standard (ifftshift of 1). Dimension nt x nr
            wav_full:    Copy of wav for each elastic component. Dimension nt x 4
            wavfft_full: Copy of wavfft for each elastic component. Dimension nt x 4
            
            
        The expression for the Ricker wavelet is according to Wikipedia with sigma = 1/(2pif0). Wikipedia uses an additional scaling factor which I omitted here.
        """
        
        # Define sigma which is a function of the central frequency
        sig = 1/(np.sqrt(2)*np.pi*f0) 
        tvec = self.Tvec()[0]
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
        
        # Copy wavelet to each elastic component
        wav_full = np.tile(wav,(1,4))
        wavfft_full = np.tile(wavfft,(1,4))
        
        return wav,wavfft,wav_full,wavfft_full
   
    def Convolve(self,field1,field2,zeropad=0):
        """
        con,confft = Convolve(self,field1,field2,dim)
        
        Convolve field1 with field2 along the time dimension.
        
        Inputs:
            field1: Wavefield (nt x 1) or (nt x 4)
            field1: Wavefield (nt x 1) or (nt x 4)
            dim:    Choose dim=0 or dim=1 or dim=(0,1)
            zeropad Set zeropad=1 to use zeropadding.
            
        Outputs:
            con:    Convolution result (1) sorted from -tmax to +tmax and -xmax to +xmax. Dimension nt (x4)
            confft: Convolution result (2) sorted according to fft standard (ifftshift of 1). Dimension nt (x4)
        """
        
        # 1D concolution
        
        F1 = np.fft.fft(field1,n=None,axis=0)
        F2 = np.fft.fft(field2,n=None,axis=0)
        if zeropad == 0:
            confft = np.fft.ifft(F1*F2,n=None,axis=0).real
        else:
            confft             = 2*np.fft.ifft(F1*F2,n=2*self.nt,axis=0).real
            confft[self.nt:,:] = 0
            confft             = confft[::2,:]
        con = np.fft.fftshift(confft,axes=0)
        return con,confft
    
    def ConvolveRicker(self,field,f0):
        """
        con,confft = ConvolveRicker(self,field,f0)
        
        Convolve field with a Ricker wavelet.
        
        Inputs:
            field: Wavefield (nt x 1) or (nt x 4)
            f0:    Central frequency of Ricker wavelet.
            
        Outputs:
            con:    Convolution result (1) sorted from -tmax to +tmax and -xmax to +xmax. Dimension nt (x4)
            confft: Convolution result (2) sorted according to fft standard (ifftshift of 1). Dimension nt (x4)
        """
        if field.shape[1] != 4:
            wav = self.Ricker(f0)[0]
        elif field.shape[1] == 4:
            wav = self.Ricker(f0)[2]
            
        if field.ndim == 3: 
            wav = np.tile(wav,(self.nr,1,1)).swapaxes(0,1)
        
        con,confft = self.Convolve(field,wav)
        
        return con,confft
    
    # Temporal gain to correct for complex-valued frequency omega
    def Gain(self,eps=0,taperlen=0):
        """
        gain = Gain(eps)
        
        Input:
            eps:      Gain factor.
            taperlen: Length of cosine taper.
        
        Output: 
            gain: Dimension (nt x 4) with a gain exp(eps*t) along the 1st dimension.
        """
        
        if eps == 0:
            gain = np.ones((self.nt,1))
            gain_full = np.ones((self.nt,4))
            return  gain,gain_full
        
        tvec = self.Tvec()[0]
        tvecfull = np.tile(tvec,(1,4))
        
        taper,taperfull = self.Taper(start=1,end=1,taperlen=taperlen)
        
        gain = np.exp(tvec*eps)*taper
        gain_full = np.exp(tvecfull*eps)*taperfull
        
        return gain,gain_full
    
    # Correct for complex-valued frequency and return data to W-p domain
    def Correct_eps_p_w(self,array,eps=0,taperlen=0):
        """
        array = Correct_eps_p_w(array,eps=0,taperlen=0)
        
        (1) The input wavefield is transformed from the w-p domain to the t-p domain.
        (2) A gain exp(eps*t) is a applied to correct for the complex-valued frequency.
        (3) The wavefield is back-transformed to the w-p domain.
        
        Inputs:
            array:    Wavefield in w-kx domain, which was modelled with a complex-valued frequency w' = w + 1j*eps
            eps:      Constant that was added to the frequency during modelling w' = w + 1j*eps.
            taperlen: After applying a gain in t-p domain the highest negative and postitibe time sample have a strong amplitude difference which can lead to fft artefacts. This can be reduced by using a cosine taper.
       
        Outputs:
            array: Wavefield in the w-p domain with real-valued frequencies w.
        
        """
        if eps == 0:
            return array
        
        # Before transforming to the t-p domain the imaginary part of the Nyquist frequency element is deleted
        if self.verbose == 1:
            print('\n')
            print('Correct_eps_p_w:')
            print('\n'+100*'-'+'\n')
            print('To correct for the complex-valued frequency Correct_eps_p_w transforms the input array to the time domain, '+
                  'applies a gain and transforms the result back to the frequency domain. Since the time signal is real-valued '+
                  'the imaginary part of the Nyquist frequency element should be zero. In this case the imaginary part equals,\n')
            print(array[self.nf-1,:].imag)
            print('\nAfter dividing by its real-part the imaginary part of the Nyquist frequency element equals,\n')
            print(array[self.nf-1,:].imag/array[self.nf-1,:].real)
            print('\nIf each event is sampled on-time the Nyquist frequency element should be real-valued within double-precision. '+
                  'Before applying an inverse Fourier transform WP2TP deletes the imaginary part of the Nyquist frequency element. Also see WP2TP.')
            print('\n')
        array[self.nf-1,:] = array[self.nf-1,:].real
        
        # Transform to the time domain.
        array_tp = np.fft.ifft(array,n=None,axis=0).real
        
        if array.shape[1] != 4:
            gain = self.Gain(eps,taperlen)[0]
        elif array.shape[1] == 4:
            gain = self.Gain(eps,taperlen)[1]
        
        gain = np.fft.ifftshift(gain,0)
        array = np.fft.fft(array_tp*gain,n=None,axis=0)
        
        return array
    
    # Transform wavefield from w-kx-domain to t-x-domain
    def WP2TP(self,array,eps=0,taperlen=0,threshold=None,norm=None,zeropad=0,paddingtaper=None,t1=None):
        """
        array_tx = WKx2TX(array,eps=0,taperlen=0)
        
        Input:
            eps:            Gain factor.
            taperlen:       Length of cosine taper.
            norm:           Choose norm='ortho' to comensate the scaling for automatic scaling by ifft. 
            zeropad:        Set zeropad=1 to mute negative times.
            paddingtaper:   Length of taper in seconds for the zero padding (to avoid sharp edges).
            t1:             Onset time of first arrival in seconds if before time zero (1x1 or 4x1). Only the earliest time will be taken into account.
        
        Output: 
            array_tp: Array in tp, fftsift and gain are applied.
        """
        # Before transforming to the t-p domain the imaginary part of the Nyquist frequency element is deleted
        if self.verbose == 1:
            print('\n')
            print('WP2TP:')
            print('\n'+100*'-'+'\n')
            print('Nyquist frequency element has an imaginary part equal to,\n')
            print(array[self.nf-1,:].imag)
            print('\nAfter dividing by its real-part the imaginary part of the Nyquist frequency element equals,\n')
            print(array[self.nf-1,:].imag/array[self.nf-1,:].real)
            print('\nIf each event is sampled on-time the Nyquist frequency element should be real-valued within double-precision. '+
                  'Before applying an inverse Fourier transform WP2TP deletes the imaginary part of the Nyquist frequency element.')
            print('\n')
        array[self.nf-1,:] = array[self.nf-1,:].real
        
        if threshold != None:
            array[abs(array)>threshold] = threshold
            
        dim = 1
        if array.ndim == 3:
            dim = 2
            
        if array.shape[dim] != 4:
            gain = self.Gain(eps,taperlen)[0]
        elif array.shape[dim] == 4:
            gain = self.Gain(eps,taperlen)[1]
            
        if array.ndim == 3:
            gain = np.tile(gain,(self.nr,1,1)).swapaxes(0,1)
        
        if norm is None:
            array_tp = np.fft.ifft(array,n=None,axis=0).real
        elif norm == 'ortho':
            array_tp = np.fft.ifft(array,n=None,axis=0,norm='ortho').real
            
        if zeropad == 1:
            
            if t1 is None:
                shift = 0
            else:
                shift = int(np.max(abs(t1))/self.dt)
            
            # Construct taper array from minus taperlength to time zero
            if paddingtaper is None:
                paddingtaper = int(self.nt/16)
            else:
                paddingtaper = int(paddingtaper/self.dt)
            tap          = np.zeros_like(array_tp)
            tap          = tap[:paddingtaper+1,:]
            tap_tmp      = np.zeros((paddingtaper+1,1))
            tap_tmp[:,0] = np.cos(np.linspace(-np.pi/2,0,paddingtaper+1))**2
            reps         = int(tap.size/tap.shape[0])
            tap_tmp      = np.tile(tap_tmp,reps)
            tap_tmp      = np.reshape(tap_tmp,(tap_tmp.size,1))
            tap          = np.reshape(tap_tmp,tap.shape)
            
            # Apply taper to avoid that zero-padding introduces a strong amplitude jump
            index = self.nt-shift
            array_tp[index-paddingtaper:index,:] = array_tp[index-paddingtaper:index,:]*tap[:-1,:]
            
            # Set times before min(t1,0) equal to zero.
            nf = int(self.nt/2)+1
            array_tp[nf-1:index-paddingtaper,:] = 0
            
            # Construct taper array from latest time minus taperlength to latest time
            # Here I restrict the taperlength to 1/16 of nt to avoid significant scaling of the amplitudes at positive times
            paddingtaper = int(self.nt/16)
            tap          = np.zeros_like(array_tp)
            tap          = tap[:paddingtaper+1,:]
            tap_tmp      = np.zeros((paddingtaper+1,1))
            tap_tmp[:,0] = np.cos(np.linspace(-np.pi/2,0,paddingtaper+1))**2
            reps         = int(tap.size/tap.shape[0])
            tap_tmp      = np.tile(tap_tmp,reps)
            tap_tmp      = np.reshape(tap_tmp,(tap_tmp.size,1))
            tap          = np.reshape(tap_tmp,tap.shape)
            
            # Apply taper to avoid that zero-padding introduces a strong amplitude jump
            array_tp[nf-2-paddingtaper:nf-2,:] = array_tp[nf-2-paddingtaper:nf-2,:]*tap[-2::-1,:]
            array_tp[nf-2,:] = 0
                

        array_tp = np.fft.fftshift(array_tp,0)
        array_tp = array_tp*gain

        return array_tp
    
    # Martix multiplication of each w-p-component of an elastic wavefield (with PP,PS,SP,SS)
    def My_dot(self,B,C):
        """
        A = My_dot(B,C)
        
        Input:
            B: Array of dimensions nf x 4
            C: Array of dimensions nf x 4
            
        Output:
            A: Matrix product between B and C for every frequency.
            
        """
        dim0 = max(B.shape[0],C.shape[0])
        dim1 = max(B.shape[1],C.shape[1])
        A = np.zeros((dim0,dim1),dtype=complex)
        A[:,0] = B[:,0]*C[:,0] + B[:,1]*C[:,2]
        A[:,1] = B[:,0]*C[:,1] + B[:,1]*C[:,3]
        A[:,2] = B[:,2]*C[:,0] + B[:,3]*C[:,2]
        A[:,3] = B[:,2]*C[:,1] + B[:,3]*C[:,3]
        return A     
    
    # Multiple martix multiplication of each w-p-component of an elastic wavefield (with PP,PS,SP,SS)
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
            A: Array of dimensions (nf x 4)
            
        Output:
            Ainv: For every w-kx element sort A in a 2x2 matrix, invert the 2x2 matrix and sort the result in an array like A (nf x nk x 4)
        """
        Ainv = A.copy()
        Ainv[:,0] =  A[:,3]
        Ainv[:,1] = -A[:,1]
        Ainv[:,2] = -A[:,2]
        Ainv[:,3] =  A[:,0]
        det = np.zeros((A.shape[0],1),dtype=complex)
        det[:,0] = A[:,0]*A[:,3] - A[:,1]*A[:,2]
        det = np.tile(det,(1,4))
        Ainv = Ainv/det
        return Ainv
    
    def My_T(self,A):
        """
        AT = My_T(A)
        
        Input:
            A: Array of dimensions (nf x 4)
            
        Output:
            AT: For every w element sort A in a 2x2 matrix, transpose the 2x2 matrix and sort the result in an array like A (nf x 4)
        """
        AT = A.copy()
        AT[:,1] = A[:,2]
        AT[:,2] = A[:,1]
        return AT
    
    # Plot wavefield
    def Plot(self,field,t=0,w=0,tx=0,tvec=0,xvec=0,wvec=0,cmap=1,title='',kill_NaN=1,vmin=None,vmax=None):
        """
        Plot(field,t=0,w=0,tx=0,tvec=0,xvec=0,wvec=0,cmap=1,title="",kill_NaN=1,vmin=None,vmax=None)
        
        Plot a (nt x 1) or (nt x 4) or (nt x nr x 4) wavefield. 
        
        Inputs:
            field:    Wavefield (nt x 1) or (nt x 4) or (nt x nr x 4).  
            t:        Set tx=1 if the wavefield is in the t-p domain (should be real-valued).
            w:        Set wkx=1 if the wavefield is in the w-p domain (can be complex-valued). field will be fftshifted.
            tx:       Set tx=1 if the wavefield is in the t-x domain (should be real-valued).
            tvec:     Set tvec=1 to get a time axis.
            xvec:     Set xvec=1 to get a space axis.
            wvec:     Set wvec=1 to get a frequency axis.
            cmap:     Set cmap=1 to get a colorbar.
            title:    Set a title by inserting a string.
            kill_NaN: Set kill_NaN=1 to set NaNs to zero and infs to high values.
            vmin:     (optional) Set a clipping minimum.
            vmax:     (optional) Set a clipping maximum.
            
        Outputs:
            Option 1: If field has dimensions nt x 1 a line plot is plotted.
            Option 2: If field has dimensions nt  x 4 a image with four subplots is plotted.
            Option 3: If field has dimensions nt  x nr x 4 a image with four subplots is plotted.
            
        """
        
        if kill_NaN == 1:
            field[np.isnan(field)==True] = 0 
        
        xlab = ''
        ylab = ''
        
        # Time domain: Determine clipping
        if t == 1:
            if vmin is None:
                myvmax = field.max()
                myvmin = -myvmax    
            else:
                myvmax = vmax
                myvmin = vmin
        
        # Frequency domain: Determine clipping, apply fftshift, take absolute value
        if w == 1:
            field = np.abs(field)
            field = np.fft.fftshift(field,0)
            if vmin is None:
                myvmin = field.min()    
                myvmax = field.max()  
            else:
                myvmax = vmax
                myvmin = vmin
                
        # Time space : Determine clipping        
        if tx == 1:
            if vmin is None:
                myvmax = field.max()
                myvmin = -myvmax    
            else:
                myvmax = vmax
                myvmin = vmin
                
        if tvec == 1:
            vec = self.Tvec()[0]
            xlab = 'Time (s)'
        else:
            vec = np.zeros((self.nt,1))
            vec[:,0] = np.arange(0,self.nt)
        
        if wvec == 1:
            vec = self.Wvec()[0]
            xlab = 'Frequency $\omega$ ($\mathrm{s}^{-1}$)'
        
        # Plot one elastic component
        if field.shape[1] == 1:
            
            fig = plt.figure()
            plt.plot(vec,field[:,0])
            plt.ylim(myvmin,myvmax)
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.title(title)
            
        # Plot four elastic components    
        elif field.shape[1] == 4:
            
            fig = plt.figure()
            ax0 = plt.subplot2grid((4, 1), (0, 0),colspan=1)
            ax1 = plt.subplot2grid((4, 1), (1, 0),colspan=1)
            ax2 = plt.subplot2grid((4, 1), (2, 0),colspan=1)
            ax3 = plt.subplot2grid((4, 1), (3, 0),colspan=1)
            #ax4 = plt.subplot2grid((2, 3), (0, 2),colspan=1,rowspan=3)
            
            im0 = ax0.plot(vec,field[:,0])
            im1 = ax1.plot(vec,field[:,1])
            im2 = ax2.plot(vec,field[:,2])
            im3 = ax3.plot(vec,field[:,3])
            
            ax0.set_ylim([myvmin,myvmax])
            ax1.set_ylim([myvmin,myvmax])
            ax2.set_ylim([myvmin,myvmax])
            ax3.set_ylim([myvmin,myvmax])
            
            ax0.set_xlabel(xlab)
            ax1.set_xlabel(xlab)
            ax2.set_xlabel(xlab)
            ax3.set_xlabel(xlab)
            
            ax0.set_ylabel(ylab+'(PP)')
            ax1.set_ylabel(ylab+'(PS)')
            ax2.set_ylabel(ylab+'(SP)')
            ax3.set_ylabel(ylab+'(SS)')
            
            plt.suptitle(title)
            
         # Plot four elastic components    
        elif field.ndim == 2 and field.shape[1] == self.nr:
            
            ex1min = 0
            ex1max = self.nt
            ex2min = 0
            ex2max = self.nr
            
            if tvec == 1:
                vec = self.Tvec()[0]
                ex1min = vec.max()
                ex1max = vec.min()
            
            if xvec == 1:
                vec = self.Xvec()[0]
                ex2min = vec.min()
                ex2max = vec.max()
                xlab = 'Offset (m)'
                ylab = 'Time (s)'
            
            fig = plt.figure()
            ax0 = plt.subplot2grid((1, 1), (0, 0),colspan=1)
            
            im0 = ax0.imshow(field,cmap='seismic',extent=[ex2min,ex2max,ex1min,ex1max], aspect=np.abs(ex2max/ex1max), vmin=myvmin, vmax=myvmax)
            
            ax0.set_xlabel(xlab)
            ax0.set_ylabel(ylab)
            
            if cmap == 1:
                fig.colorbar(im0, ax=ax0)
            
            plt.suptitle(title)
            
        # Plot four elastic components    
        elif field.ndim == 3:
            
            ex1min = 0
            ex1max = self.nt
            ex2min = 0
            ex2max = self.nr
            
            if tvec == 1:
                vec = self.Tvec()[0]
                ex1min = vec.max()
                ex1max = vec.min()
            
            if xvec == 1:
                vec = self.Xvec()[0]
                ex2min = vec.min()
                ex2max = vec.max()
                xlab = 'Offset (m)'
                ylab = 'Time (s)'
            
            fig = plt.figure() 
            ax0 = plt.subplot2grid((2, 2), (0, 0),colspan=1)
            ax1 = plt.subplot2grid((2, 2), (0, 1),colspan=1)
            ax2 = plt.subplot2grid((2, 2), (1, 0),colspan=1)
            ax3 = plt.subplot2grid((2, 2), (1, 1),colspan=1)
            
            im0 = ax0.imshow(field[:,:,0],cmap='seismic',extent=[ex2min,ex2max,ex1min,ex1max], aspect=np.abs(ex2max/ex1max), vmin=myvmin, vmax=myvmax)
            im1 = ax1.imshow(field[:,:,1],cmap='seismic',extent=[ex2min,ex2max,ex1min,ex1max], aspect=np.abs(ex2max/ex1max), vmin=myvmin, vmax=myvmax)
            im2 = ax2.imshow(field[:,:,2],cmap='seismic',extent=[ex2min,ex2max,ex1min,ex1max], aspect=np.abs(ex2max/ex1max), vmin=myvmin, vmax=myvmax)
            im3 = ax3.imshow(field[:,:,3],cmap='seismic',extent=[ex2min,ex2max,ex1min,ex1max], aspect=np.abs(ex2max/ex1max), vmin=myvmin, vmax=myvmax)
            
            ax0.set_xlabel(xlab)
            ax1.set_xlabel(xlab)
            ax2.set_xlabel(xlab)
            ax3.set_xlabel(xlab)
            
            ax0.set_ylabel(ylab+'(PP)')
            ax1.set_ylabel(ylab+'(PS)')
            ax2.set_ylabel(ylab+'(SP)')
            ax3.set_ylabel(ylab+'(SS)')
            
            if cmap == 1:
                fig.colorbar(im0, ax=ax0)
                fig.colorbar(im1, ax=ax1)
                fig.colorbar(im2, ax=ax2)
                fig.colorbar(im3, ax=ax3)
            
            plt.suptitle(title)
            
        return fig
        
