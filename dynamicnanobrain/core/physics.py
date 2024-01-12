#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:57:06 2021

@author: dwinge
"""
import numpy as np
import pandas as pd


def square_pulse(t, tlims, amplitude):
    tmp = 0.
    for ttuple in tlims:
        tmp += float((t >= ttuple[0])*(t < ttuple[1]))
    return tmp*amplitude

def constant(t, amplitude) :
    return amplitude


class Device:
    # Class variable, shared by all instances
    keys = ['Rinh','Rexc','RLED','Rstore','Cinh','Cexc','CLED','Cstore','Cgate','Vt','m','I_Vt','vt','Lg','AB','CB']
    units = ['Ohm']*4 + ['F']*4 + ['F/cm'] + ['V','dim. less','nA','cm/s','um','uA','1/uA']
    kT = 0.02585
    
    def __init__(self, path_to_file) :
        # Read in the specified parameter file
        self.p_dict = {}
        self.p_units = {}
        self.read_parameter_file(path_to_file,self.p_dict, self.p_units)
        # Transistor linear slope in nA/V
        self.linslope=self.p_dict['Cgate']*self.p_dict['vt']*1e9 # nA/V
        self.gammas = self.calc_gammas()
        self.A = self.calc_A()        
        
    def read_parameter_file(self, path_to_file, p_dict, p_units) :
        # Read file and then loop through and assign to dict
        params = np.loadtxt(path_to_file)
        for k, key in enumerate(self.keys):
            p_dict[key] = params[k]
            p_units[key] = self.units[k]

    def set_parameter(self,key,value) :
        # HERE GAMMAS SHOULD BE UPDATED!
        self.p_dict[key]=value
        self.gammas = self.calc_gammas()
        self.A = self.calc_A()
    
    def print_parameter(self, key):
        print(f'The parameter {key}={self.p_dict[key]} {self.p_units[key]}')
              
    # Supply transistor functionality, the method transistorIV can be ported 
    def Id_sub(self,Vg,Vt,mask) :
        return self.p_dict['I_Vt']*np.exp((Vg-Vt[mask])/self.p_dict['m']/self.kT)

    def Id_sat(self,Vg,Vt,mask) :
        # Invert the mask for this one
        return self.p_dict['I_Vt'] + self.linslope*(Vg-Vt[mask==False])
    
    def Id_sub_0(self,Vg,Vt) :
        return self.p_dict['I_Vt']*np.exp((Vg-Vt)/self.p_dict['m']/self.kT)

    def Id_sat_0(self,Vg,Vt) :
        return self.p_dict['I_Vt'] + self.linslope*(Vg-Vt)

    def transistorIV(self,Vg,Vt_vec=None) :
        """Reads gate voltage and calculated transistor current based on 
        parameters in p_dict. Can take a vector of individual threshold currents 
        to introduce fluctuations in the system. Returns current in nA."""

        if Vt_vec is None :
            Vt = self.p_dict['Vt'] 
            #return np.piecewise(Vg, [Vg<Vt, Vg>=Vt], [self.Id_sub_0, self.Id_sat_0],Vt) 
        return np.heaviside(Vg-Vt,0.5)*self.Id_sat_0(Vg,Vt)+np.heaviside(Vt-Vg,0.5)*self.Id_sub_0(Vg,Vt)
        #else :
        #    Vt = Vt_vec
        #    # This should work even when Vt is an array
        #    return np.piecewise(Vg, [Vg<Vt, Vg>=Vt], [self.Id_sub, self.Id_sat],Vt,Vg<Vt)   
    
    def transistorIVold(self,Vg,Vt_vec=None) :
        
        if Vt_vec is None :
            Vt = self.p_dict['Vt'] 
            return np.piecewise(Vg, [Vg<Vt, Vg>=Vt], [self.Id_sub_0, self.Id_sat_0],Vt) 
            #return np.heaviside(Vg-Vt,0.5)*self.Id_sat_0(Vg,Vt)+np.heaviside(Vt-Vg,0.5)*self.Id_sub_0(Vg,Vt)
        else :
            Vt = Vt_vec
            # This should work even when Vt is an array
            return np.piecewise(Vg, [Vg<Vt, Vg>=Vt], [self.Id_sub, self.Id_sat],Vt,Vg<Vt)   
    
    def transistorIV_example (self, Vstart=-0.5, Vend=1.0) :
        # Generate a sample transistor IV, generates data in uA
        NV = 200
        
        Vgate = np.linspace(Vstart, Vend, NV)
        I = self.transistorIV(Vgate)*1e-3
        data = np.array([Vgate, I])
        
        df = pd.DataFrame(data.transpose(),
                          columns=['Vgate','Current'])
        
        return df

    def calc_Cmem(self,Cstore=None) :
        # Sum the memory and gate capacitance, convert Lg in um to cm
        if Cstore is None :
            Cmem = self.p_dict['Cstore'] + self.p_dict['Cgate']*self.p_dict['Lg']*1e-4 
        else :
            Cmem = Cstore + self.p_dict['Cgate']*self.p_dict['Lg']*1e-4  
        return Cmem

    # Supply the gammas needed for time stepping
    def calc_gammas(self,Rstore=None,Cstore=None) :
        # Sum the memory and gate capacitance, convert Lg in um to cm
        if Cstore is not None:
            Cmem = self.calc_Cmem(Cstore)
        else :
            Cmem = self.calc_Cmem()
        # System frequencies
        g11 = 1e-9/self.p_dict['Cinh']/self.p_dict['Rinh'] # ns^-1 # GHz
        g22 = 1e-9/self.p_dict['Cexc']/self.p_dict['Rexc'] # ns^-1 # GHz
        g13 = 1e-9/Cmem/self.p_dict['Rinh'] # ns^-1 # GHz
        g23 = 1e-9/Cmem/self.p_dict['Rexc'] # ns^-1 # GHz
        if Rstore is not None :
            g33 = 1e-9/Cmem/Rstore
        else :
            g33 = 1e-9/Cmem/self.p_dict['Rstore'] # ns^-1 # GHz
        gled = 1e-9/self.p_dict['CLED']/self.p_dict['RLED'] # ns^-1 # GHz

        return np.array([g11,g22,g13,g23,g33,gled])

    def calc_tau_gate(self) :
        # System frequencies
        g33 = 1e-9/self.calc_Cmem()/self.p_dict['Rstore'] # ns^-1 # GHz
        # Modifying this as it is g33 that matter for a system in equilibrium
        return g33**-1

    # System matrix setup
    def A_mat(self,g11,g22,g13,g23,g33) :
        gsum = g13+g23+g33
        A = np.array([[-g11, 0, g11],
                      [0, -g22, g22],
                      [g13, g23, -gsum]])
    
        return A
    
    def calc_A(self,Rstore=None,Cstore=None) :
        # Calcualte A for the special case of a distribution of Rstore and Cstore, 
        # or for the object self.gammas
        if (Rstore is not None) and (Cstore is not None) :
            new_gammas = self.calc_gammas(Rstore,Cstore)
            return self.A_mat(*new_gammas[:-1])
        elif Rstore is not None :
            new_gammas = self.calc_gammas(Rstore)
            return self.A_mat(*new_gammas[:-1])
        else :
            return self.A_mat(*self.gammas[:-1])    

    # Transfer function calculations
    def setup_gain (self,gammas) :
        # Setup matrix
        A = self.A_mat(*gammas[:-1])
        #print('System matrtix A=\n',A)   
        # Find the eigenvalues
        v, w = np.linalg.eig(A)
    
        # Add our LED system as a pole as well
        gled = gammas[-1]
        v_add = np.concatenate((v,np.array([-gled])))
          
        return v_add
    
    def gain(self, s, eigvals, gammas, vsat=1e7, Vt_prime=0.) :
        # This function needs to change if Vt_prime is not zero
        if Vt_prime > 0. :
            print('Values of Vt_prime not equal zero not supported')
            return 0
        
        # Sort the input
        g11, g22,g13,g23,g33,gled = gammas
        # Get the necessary capacitance values
        Cexc = self.p_dict['Cexc'] 
        Cinh = self.p_dict['Cinh'] 
        Cgate = self.p_dict['Cgate']
        
        # Denoninator of expression (shared between G11, G12)
        denom = np.ones(len(s),dtype=complex)
        for l in eigvals :
            denom *= s-l

        # Constant prefactor (shared between G11, G12)
        prefactor = gled*vsat*Cgate
        # gammas are in inverse ns so units have to be adjusted
        # (in the scope of the total expression)
        prefactor *= 1e-9 # ns to s
        
        # Specifying the actual terms
        G11 = g23*(s+g22)/Cexc * prefactor/denom
        G12 = -g13*(s+g11)/Cinh * prefactor/denom  
        
        return G11, G12
    
    def unity_coupling_coefficient(self, eta_handle, s=1e-3) :
        # Get the unity couping coefficient from the system
        sample_s = np.array([s])
        # Setup the gain function with system gammas
        gammas = self.gammas
        eigvals = self.setup_gain(gammas)
        # Sample the gain function
        G11_at_s, _ = self.gain(sample_s,eigvals,gammas)
        # Add also the LED efficiency
        eta_max = max(self.eta_example(eta_handle)['eta, IQE'])
        print(f'Found max eta of {eta_max}')
    
        return (eta_max*G11_at_s.real[0])**-1
        
    def inverse_gain_coefficient(self, eta_handle, Vthres) :
        # Remember nA
        Rsum=self.p_dict['Rstore']+self.p_dict['Rexc']
        max_Vg = Vthres*self.p_dict['Rstore']/Rsum
        Iexc = Vthres/Rsum*1e9 # nA
        Isd = self.transistorIV(max_Vg) 
        Iout = eta_handle(Isd)*Isd
        return Iexc/Iout, Iexc
        
    # LED efficiency physics as well
    def ABC(self,I,AB,CB) :
        # ABC model adapted to currents
        eta = I/(AB + I + CB*I**2)
        return eta
        
    def eta_ABC(self,I) :
        # calculate efficiency from ABC model
        # ABC model is in uA, so multiply I by 1e-3
        return self.ABC(I*1e-3,self.p_dict['AB'],self.p_dict['CB']) 
    
    def eta_unity(self, I) :
        #return np.ones_like(np.atleast_1d(I))
        return 1.0
    
    def eta_example(self, handle) :
        
        I = 10**np.linspace(-2,2,num=50) # in uA
        eta = handle(I*1e3) # ABC expects nA
        if len(np.atleast_1d(eta)) < len(I) :
            # eta is a single value
            eta = eta*np.ones_like(I)
            
        data = np.array([I, eta])
        
        df = pd.DataFrame(data.transpose(),
                          columns=['Current (uA)','eta, IQE'])
        
        return df
    
    def setupLED(self, path_to_LED) :

        VLED, ILED = np.loadtxt(path_to_LED,comments='%',unpack=True)
  
        # Interpolate for I(VLED)
        from scipy.interpolate import interp1d
        # from uA to nA here:
        intp_iv_LED = interp1d(VLED,ILED*1e3,kind='cubic',fill_value='extrapolate')

        # Now calculate the diff resistance
        vspace = np.linspace(0.6,1.9,301) # This part is hard-coded
        dv = vspace[1]-vspace[0]
        delta_I = intp_iv_LED(vspace[1:])-intp_iv_LED(vspace[:-1])
        dI_dV = delta_I/dv

        # Invert and interpolate to get the the differential resistance
        intp_dR_LED = interp1d(vspace[1:],dI_dV**-1,kind='cubic',fill_value='extrapolate')
    
        return intp_dR_LED, intp_iv_LED