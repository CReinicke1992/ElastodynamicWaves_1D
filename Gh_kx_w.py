#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:08:54 2017

@author: christianreini
"""

from Marchenko_kx_w import Marchenko_kx_w
import numpy as np

class Gh_kx_w(Marchenko_kx_w):
    """
    Gh_kx_w
    
    Compute the homogeneous Green's function between two depth levels in a layered medium using a single-sided representation, for multiple frequencies and offsets.
    
    Homogeneous Green's functions are acausal. Therefore a complex-valued frequency will attenuate one wrap-around but boost the opposite wrap around. 
    
    Variables:
        nt:    Number of time/frequency samples
        dt:    Duration per time sample in seconds
        nr:    Number of space/wavenumber samples
        dx:    Distance per space samples in metres
        dzvec: List or array with the thickness of each layer
        cpvec: List or array with the P-wave veclocity in each layer
        csvec: List or array with the S-wave veclocity in each layer
        rovec: List or array with the density of each layer
        zS:    Source depth level
        zR:    Receiver depth level
        
    Data sorting: 
        nt x nr x 4
        
    Vectorised computation
    """
    
    # The class Homogeneous_G_kx_w computes the homogeneous Green's function 
    # between two depth levels in a layered medium using a single-sided
    # representation. Hence focusing functions are required. Hence, all 
    # properties of the class Marchenko_kx_w are inheritted.
    def __init__(self,nt,dt,nr,dx,dzvec,cpvec,csvec,rovec):
        Marchenko_kx_w.__init__(self,nt,dt,nr,dx,dzvec,cpvec,csvec,rovec)
        self.zS   = None
        self.zR   = None
        #self.N    = len(self.dzvec) 
        #self.zvec = np.cumsum(self.dzvec).tolist()
        
#    # Insert a layer in the model    
#    def Insert_layer(self,z0):
#        """
#        Insert_layer(z0)
#        
#        Insert a depth level if it does not exist yet.
#        
#        Input:
#            z0: Depth
#        """
#        
#        # Depth vector
#        z = np.cumsum(self.dzvec)
#        
#        # Vector of depths smaller or equal to z0
#        L = z[z<=z0] 
#        
#        # Case1: z0 smaller than self.dzvec[0]
#        if L.size == 0:
#            dzvec = np.array(self.dzvec)
#            self.dzvec = np.hstack((z0,dzvec[0]-z0,dzvec[1:])).tolist()
#            
#            cpvec = np.array(self.cpvec)
#            self.cpvec = np.hstack((cpvec[0],cpvec)).tolist()
#            csvec = np.array(self.csvec)
#            self.csvec = np.hstack((csvec[0],csvec)).tolist()
#            rovec = np.array(self.rovec)
#            self.rovec = np.hstack((rovec[0],rovec)).tolist()
#            return
#        
#        # Case2: z0 coincides with an element of z = np.cumsum(self.dzvec)
#        elif L[-1] == z0:
#            return
#        
#        # Case 3: z0 is larger than z[-1] = = np.cumsum(self.dzvec)[-1]
#        elif L.size == z.size:
#            dzvec = np.array(self.dzvec)
#            self.dzvec = np.hstack((dzvec,z0-z[-1])).tolist()
#            
#            cpvec = np.array(self.cpvec)
#            self.cpvec = np.hstack((cpvec,cpvec[-1])).tolist()
#            csvec = np.array(self.csvec)
#            self.csvec = np.hstack((csvec,csvec[-1])).tolist()
#            rovec = np.array(self.rovec)
#            self.rovec = np.hstack((rovec,rovec[-1])).tolist()
#            return
#            
#        # Case 4: z0 is between z[0] and z[-1] AND does not coincide with any element of z
#        
#        b = L[-1] 
#        ind = z.tolist().index(b)
#        
#        dzvec = np.array(self.dzvec)
#        self.dzvec = np.hstack((dzvec[:ind+1],z0-b,z[ind+1]-z0,dzvec[ind+2:])).tolist()
#        
#        # Parameters
#        cpvec = np.array(self.cpvec)
#        self.cpvec = np.hstack((cpvec[:ind+1],cpvec[ind+1],cpvec[ind+1],cpvec[ind+2:])).tolist()
#        csvec = np.array(self.csvec)
#        self.csvec = np.hstack((csvec[:ind+1],csvec[ind+1],csvec[ind+1],csvec[ind+2:])).tolist()
#        rovec = np.array(self.rovec)
#        self.rovec = np.hstack((rovec[:ind+1],rovec[ind+1],rovec[ind+1],rovec[ind+2:])).tolist()
#        
#        return
#    
#    # Remove a layer from the model
#    def Remove_layer(self,z0):
#        """
#        Remove_layer(z0)
#        
#        Remove a depth level if it was introduced by Insert_layer.
#        
#        Input:
#            z0: Depth
#        """
#        
#        #if len(self.dzvec) == self.N:
#        if z0 in self.zvec:
#            return
#        
#        ind = np.cumsum(self.dzvec).tolist().index(z0) + 1
#        
#        if ind == len(self.dzvec):
#            del self.dzvec[ind-1]
#            del self.cpvec[ind-1]
#            del self.csvec[ind-1]
#            del self.rovec[ind-1]
#            return
#        
#        self.dzvec[ind-1] = self.dzvec[ind-1] + self.dzvec[ind]
#        del self.dzvec[ind]
#        del self.cpvec[ind]
#        del self.csvec[ind]
#        del self.rovec[ind]
#        
#        return
    
    # Compute homogeneous Green's functions G1
    def G1(self,zS,mul=1,conv=1,RPfull=None):
        """
        GPPfull,GPMfull,GMPfull,GMMfull = G1(zS,mul=1,conv=1)
        
        The homogeneous Green's functions are computed via the Marchenko equations. Hence, the focusing functions and the full-medium's reflection response are used.
        
         Inputs:
            zS:       Virtual source depth.
            mul:      Set mul=1 to include internal multiples.
            conv:     Set conv=1 to include P/S conversions.
            RP:       (Optional) Reflection response in the F-Kx domain with a real-valued frequency w' = w + 1j*0
            
        Outputs:
            GPPfull: Green's function G-plus-plus between the focal depth zS and the surface. Real-valued frequencies. All frequencies and wavenumbers (nt x nr x 4).
            GPMfull: Green's function G-plus-minus between the focal depth zS and the surface. Real-valued frequencies. All frequencies and wavenumbers (nt x nr x 4).
            GMPfull: Green's function G-minus-plus between the focal depth zS and the surface. Real-valued frequencies. Positive and the highest negative frequencies / wavenumbers (nf x nk x 4).
            GMMfull: Green's function G-minus-minus between the focal depth zS and the surface. Real-valued frequencies. Positive and the highest negative frequencies / wavenumbers (nf x nk x 4).
            RPfull:  Reflection response of the full medium.
        """
        
        print('Computing Greens function G1 ...')
            
        if zS != self.zS:
            self.zS = zS
      
        F1P,F1M = self.FocusFunc1_mod(self.zS,mul,conv,eps=0,sort=0)[2:4]
        
        # Compute reflection/transmission response of  medium for positive frequencies and wavenumbers
        if RPfull is None:         
            RP = self.Layercel_kx_w(mul=mul,conv=conv,eps=0,sort=0)[0]
            RPfull = self.Sort_kx_w(RP)
        
        # Number of frequency samples
        nf = int(self.nt/2)+1
        nk = int(self.nr/2)+1
        
        # Complex conjugation of R
        RPc         = RPfull.conj()     # Complex conjugate R
        RPc[:,1:,:] = RPc[:,-1:0:-1,:]  # Reverse kx
        RPc         = RPc[:nf,:nk,:]    # Extract positive w-kx elements
        
        # 1st Marchenko equation: GMP
        GMP = self.My_dot(RP,F1P) - F1M
        GMP = np.nan_to_num(GMP)
        
        # 2nd Marchenko equation: GPP
        GPP = - self.My_dot(RPc,F1M) + F1P
        GPP = np.nan_to_num(GPP)
        
        # Apply symmetry of homogeneous Green's function
        
        # GPM with reversed kx axis
        GPM_nkx = - GMP.conj()
        
        # GMM with reversed kx axis
        GMM_nkx = - GPP.conj()
        
        # Get negative frequencies / wavenumbers
        GPPfull,GPMfull,GMPfull,GMMfull = self.Sort_kx_w(GPP,GPM_nkx,GMP,GMM_nkx)
        
        # Reverse kx
        GPMfull[:,1:,:] = GPMfull[:,-1:0:-1,:]
        GMMfull[:,1:,:] = GMMfull[:,-1:0:-1,:]
        
        return GPPfull,GPMfull,GMPfull,GMMfull,RP
    
    def G2(self,zR,zS,mul=1,conv=1,G1=None,initials=[]):
        """
        GPPfull,GPMfull,GMPfull,GMMfull = G2(zS,mul=1,conv=1,eps=None,RPfull=None)
        
        The homogeneous Green's functions are computed via the Marchenko equations. Hence, the focusing functions and the full-medium's reflection response are used.
        
         Inputs:
            zR:   Virtual receiver depth.
            zS:   Virtual source depth.
            mul:  Set mul=1 to include internal multiples.
            conv: Set conv=1 to include P/S conversions.
            G1:   (Optional) Homogeneous Green's function between virtual source and surface in the F-Kx domain with a real-valued frequency w' = w + 1j*0
            
        Outputs:
            GPPfull: Green's function G-plus-plus between the focal depth zS and the surface. Real-valued frequencies. All frequencies and wavenumbers (nt x nr x 4).
            GPMfull: Green's function G-plus-minus between the focal depth zS and the surface. Real-valued frequencies. All frequencies and wavenumbers (nt x nr x 4).
            GMPfull: Green's function G-minus-plus between the focal depth zS and the surface. Real-valued frequencies. Positive and the highest negative frequencies / wavenumbers (nf x nk x 4).
            GMMfull: Green's function G-minus-minus between the focal depth zS and the surface. Real-valued frequencies. Positive and the highest negative frequencies / wavenumbers (nf x nk x 4).
        """
        print('Computing Greens function G2 ...')
        
        # Number of frequency samples
        nf = int(self.nt/2)+1
        nk = int(self.nr/2)+1
        
        if zS != self.zS or zR != self.zR:
            self.zS = zS
            self.zR = zR    
        
        if G1 == None:
            GPPfull,GPMfull,GMPfull,GMMfull = self.G1(self.zS,mul=1,conv=1)[:4]
        
            G1PP = GPPfull[0:nf,0:nk,:]
            G1PM = GPMfull[0:nf,0:nk,:]
            G1MP = GMPfull[0:nf,0:nk,:]
            G1MM = GMMfull[0:nf,0:nk,:]
            G1 = [G1PP,G1PM,G1MP,G1MM]
            
        # Reuse G1 if it was computed previously
        else:
            G1PP,G1PM,G1MP,G1MM = G1
        
        
        F1P,F1M,initials = self.FocusFunc1_mod(self.zR,mul,conv,eps=0,sort=1,initials=initials)[2:5]
        
        # Extract negative kx values
        F1P_nkx = F1P.copy()                # Copy F1P
        F1P_nkx[:,1:,:] = F1P[:,-1:0:-1,:]  # Reverse kx
        F1P_nkx = F1P_nkx[:nf,:nk,:]        # Extract positive w-kx
        
        F1M_nkx = F1M.copy()                # Copy F1M
        F1M_nkx[:,1:,:] = F1M[:,-1:0:-1,:]  # Reverse kx
        F1M_nkx = F1M_nkx[:nf,:nk,:]        # Extract positive w-kx
        
        F1P_nkx = F1P_nkx[:nf,:nk,:]
        F1M_nkx = F1M_nkx[:nf,:nk,:]
        
        # G2MP = FPt G1MP - FMt G1PP
        G2MP = self.My_dot(self.My_T(F1P_nkx),G1MP) - self.My_dot(self.My_T(F1M_nkx),G1PP)
        G2MP = np.nan_to_num(G2MP)
        
        # G2MM = FPt G1MM - FMt G1PM
        G2MM = self.My_dot(self.My_T(F1P_nkx),G1MM) - self.My_dot(self.My_T(F1M_nkx),G1PM)
        G2MM = np.nan_to_num(G2MM)
        
        # Symmetry of homogeneous Green's fucntion
        G2PM_nkx = -G2MP.conj()
        G2PP_nkx = -G2MM.conj()
        
        # Construct full wavefield
        G2PPfull,G2PMfull,G2MPfull,G2MMfull = self.Sort_kx_w(G2PP_nkx,G2PM_nkx,G2MP,G2MM)
    
        G2PPfull[:,1:,:] = G2PPfull[:,-1:0:-1,:] # Reverse kx
        G2PMfull[:,1:,:] = G2PMfull[:,-1:0:-1,:] # Reverse kx
        
        return G2PPfull,G2PMfull,G2MPfull,G2MMfull,G1,initials
    
    # Double-sided homogeneous Greens function
    def Gh(self,zR,zS,mul=1,conv=1,GzS=None,initials=[]):
        """  
        GPPfull,GPMfull,GMPfull,GMMfull = Gh(zS,mul=1,conv=1)
        
        The homogeneous Green's functions are computed via interferometry. Hence, the Green's functions recorded at the domain's enclosing boundary are used.
        
         Inputs:
            zR:   Virtual receiver depth.
            zS:   Virtual source depth.
            mul:  Set mul=1 to include internal multiples.
            conv: Set conv=1 to include P/S conversions.
            
        Outputs:
            GPPfull: Homogeneous Green's function G-plus-plus between source depth zS and receiver depth zR. Real-valued frequencies. All frequencies and wavenumbers (nt x nr x 4).
            GPMfull: Homogeneous Green's function G-plus-plus between source depth zS and receiver depth zR. Real-valued frequencies. All frequencies and wavenumbers (nt x nr x 4).
            GMPfull: Homogeneous Green's function G-plus-plus between source depth zS and receiver depth zR. Real-valued frequencies. Positive and the highest negative frequencies / wavenumbers (nf x nk x 4).
            GMMfull: Homogeneous Green's function G-plus-plus between source depth zS and receiver depth zR. Real-valued frequencies. Positive and the highest negative frequencies / wavenumbers (nf x nk x 4).
        """
        print('Computing double-sided representation of homogeneous Greens function Gh ...')
        
        # Number of frequency samples
        nf = int(self.nt/2)+1
        nk = int(self.nr/2)+1
            
        if zS != self.zS or zR != self.zR:
            self.zS = zS
            self.zR = zR   
        
        # Model all required Green's functions with source at zS
        
        if GzS is None:
            
            # Gives wavefields of the shape (nt,nr,4)
            Gs,Gb = self.Gz2bound(self.zS,mul,conv,eps=0)[0:2]
            
            # Extract Green's function with source inside the medium
            GMPzS,GMMzS = Gs[2:4]
            GPMzS,GPPzS = Gb[2:4]
            
            # Extract positive frequencies and wavenumbers
            GMPzS = GMPzS[:nf,:nk,:]
            GMMzS = GMMzS[:nf,:nk,:]
            GPMzS = GPMzS[:nf,:nk,:]
            GPPzS = GPPzS[:nf,:nk,:]
            
            GzS = [GMPzS,GMMzS,GPMzS,GPPzS]
            
        else:
            # Reuse GzS
            GMPzS,GMMzS,GPMzS,GPPzS = GzS
        
        # Model all required Green's functions with source at zR
        
        # Gives wavefields of the shape (nt,nr,4)
        Gs,Gb,initials = self.Gz2bound(self.zR,mul,conv,eps=0,initials=initials)[0:3]
        
        # Extract Green's function with source inside the medium
        GMPzR,GMMzR = Gs[2:4]
        GPMzR,GPPzR = Gb[2:4]
        
        # Extract positive frequencies and wavenumbers
        GMPzR = GMPzR[:nf,:nk,:]
        GMMzR = GMMzR[:nf,:nk,:]
        GPMzR = GPMzR[:nf,:nk,:]
        GPPzR = GPPzR[:nf,:nk,:]
        
        # Compute dagger of GzR
        GMPzR = self.My_T(GMPzR).conj()
        GMMzR = self.My_T(GMMzR).conj()
        GPMzR = self.My_T(GPMzR).conj()
        GPPzR = self.My_T(GPPzR).conj()
        
        # Compute homogeneous Green's function
        GhPP =   self.My_dot(GMPzR,GMPzS) + self.My_dot(GPPzR,GPPzS) 
        GhPM =   self.My_dot(GMPzR,GMMzS) + self.My_dot(GPPzR,GPMzS) 
        GhMP = - self.My_dot(GMMzR,GMPzS) - self.My_dot(GPMzR,GPPzS) 
        GhMM = - self.My_dot(GMMzR,GMMzS) - self.My_dot(GPMzR,GPMzS) 
        
        
        # G2MP = FPt G1MP - FMt G1PP
        # G2MM = FMt G1MPc - FPt G1PPc
        
        GhPPfull,GhPMfull,GhMPfull,GhMMfull = self.Sort_kx_w(GhPP,GhPM,GhMP,GhMM)
        
        return GhPPfull,GhPMfull,GhMPfull,GhMMfull,GzS,initials
        
        
        