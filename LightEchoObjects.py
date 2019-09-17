#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:25:37 2019

@author: elizabethjohnson
"""

import numpy as np
import math 
import matplotlib.pylab as plt
from sb import S_tableclass
from pylab import * 
from matplotlib import rc
import collections
import multiprocessing

import GetThetaDistributionFromTychoCasA as GetThetas

class LightEcho:
    
    ## instance variables
    def __init__(self):
        self.theta = 90  ## scattering angle 
        self.D = 163000  ## distance to the LMC in lyrs 
        self.c = 1       ## speed of light
    
    
    
    
    ## methods 

    def ellipse_location(self,theta,t):
    
        theta_rad = np.radians(theta)
        
        ## from solving quadratic eq on paper
        r = (np.cos(theta_rad) + 1)/((np.sin(theta_rad))**2 / (self.c*t))
        
        x = r*np.sin(theta_rad) ## trig
        
        z = r*np.cos(theta_rad) ## trig 
        
        
        return r,z,x
    
    def brightness_dim_factor(self,theta,t):
        
        need = self.ellipse_location(theta,t) ## approx. how much extra it travels
        
        dim = 1/need[0]**2 * 1/(self.D - need[1])**2  ## approx just D 
        #print('Theta= ', theta, ' z= ',need[1],' r= ',need[0], ' x= ', need[2])
        return dim 
    
    def magnitude_compared_to_known(self,theta,t):
        
        dim_candidate = self.brightness_dim_factor(theta,t)
        dim_known = self.brightness_dim_factor(45,400) ## known parameters
        
        factor_difference = dim_known/dim_candidate
        mag_difference = math.log(factor_difference,100**(1/5))
        
        mag_candidate = 22.5 + mag_difference ## known one is 22.5 
        
        return mag_candidate
    
    def carry_out_for_t_vals(self,theta,tmin,tmax,step):
        
        mags = []
        for t in range(tmin,tmax,step):
            mags.append(self.magnitude_compared_to_known(theta,t))
            
        return mags 
        
    def plot_mags_vs_age(self,theta,tmin,tmax,step):
        
        mags = self.carry_out_for_t_vals(theta,tmin,tmax,step)
        t_vals = np.arange(tmin,tmax,step)
        
        
        plt.plot(t_vals,mags,label=r'$\theta =$%s$^o$' % np.str(theta))
        
        
    def plot_many_thetas(self,theta1,theta2,theta3,tmin,tmax,step):
        
        plt.figure(figsize=(6,6))
        plt.rc('axes', linewidth=2)
        self.plot_mags_vs_age(theta1,tmin,tmax,step)
        self.plot_mags_vs_age(theta2,tmin,tmax,step)
        self.plot_mags_vs_age(theta3,tmin,tmax,step)
        plt.xlabel('Age',fontsize=18)
        plt.ylabel('Vega Magnitude / arcsec^2',fontsize=18)
        plt.tight_layout()
        plt.legend()
        
        
    ########################################################    
        
    def get_params_for_obs_angle(self,obsangle,t):
        
        obsangle_rad = np.radians(obsangle)
        #x = np.tan(obsangle_rad)*(self.D) ## this is the approximation 
        x = (-1 + np.sqrt(1 + 4*(np.tan(obsangle_rad))**2/(2*self.c*t)*(self.D + self.c*t/2)))/(
                np.tan(obsangle_rad)/(self.c*t)) ## solution to quadratic... real 
        
        ## use light echo equation 
        
        z = x**2/(2*self.c*t) - self.c*t/2
        
        r = np.sqrt(x**2 + z**2)
        
        theta = np.degrees(np.arccos(z/r))
        
        #dim = 1/r**2 * 1/self.D**2
        dim = self.brightness_dim_factor(theta,t) ### using my previous code 
        
        dim_known = self.brightness_dim_factor(45,400) ## known parameters
        
        factor_difference = dim_known/dim
        mag_difference = math.log(factor_difference,100**(1/5))
        
        mag_candidate = 22.5 + mag_difference ## known one is 22.5 
        
        return mag_candidate, z, r, x, factor_difference, theta
        

    def keeping_obs_angle_constant(self,obsangle,tmin,tmax,step):
        
        mags = []
        factors = []
        for t in range(tmin,tmax,step):
            mags.append(self.get_params_for_obs_angle(obsangle,t)[0])
            factors.append(self.get_params_for_obs_angle(obsangle,t)[4])
            
        return mags, factors
    
    def plot_mags_vs_age_obsangle(self,obsangle,tmin,tmax,step):
        
        mags = self.keeping_obs_angle_constant(obsangle,tmin,tmax,step)[0]
        t_vals = np.arange(tmin,tmax,step)
        
        plt.plot(t_vals,mags,label=r'Obs Angle =%s$^o$' % obsangle)
        
    def plot_many_obs_angles(self,obsangle1,obsangle2,obsangle3,obsangle4,
                             obsangle5,obsangle6,obsangle7,tmin,tmax,step):
        
        plt.figure(figsize=(6,6))
        plt.rc('axes', linewidth=2)
        self.plot_mags_vs_age_obsangle(obsangle1,tmin,tmax,step)
        self.plot_mags_vs_age_obsangle(obsangle2,tmin,tmax,step)
        self.plot_mags_vs_age_obsangle(obsangle3,tmin,tmax,step)
        self.plot_mags_vs_age_obsangle(obsangle4,tmin,tmax,step)
        self.plot_mags_vs_age_obsangle(obsangle5,tmin,tmax,step)
        self.plot_mags_vs_age_obsangle(obsangle6,tmin,tmax,step)
        self.plot_mags_vs_age_obsangle(obsangle7,tmin,tmax,step)
        plt.xlabel('Age',fontsize=18)
        plt.ylabel('Vega Magnitude / arcsec^2',fontsize=18)
        plt.tight_layout()
        plt.legend()
        
    #############################################################
    
    def scattering_function(self,obsangle,tmin,tmax,step):
        
        S = {}
        wmin={'LMCavg':4000.0,'LMC2':4000.0,'SMCbar':4000.0,'MWG':3500.0}
        wmax={'LMCavg':7999.0,'LMC2':7999.0,'SMCbar':7999.0,'MWG':9999.0}
        lamb = 7000
        
        Sval_of_known = 9.354e-23
        
        ## the scattering function stuff
        dusttype = 'LMCavg'
        S[dusttype]=S_tableclass()
        dustfilename =  S[dusttype].getdustfilename(dusttype)
        #print('Loading dust properties...')
        S[dusttype].loadtable(dustfilename)
        #if lamb<wmin[dusttype] or lamb>wmax[dusttype]: continue 
        ##########
        
        Svals = []
        mags = []
        factors = []
        for t in range(tmin,tmax,step):
            vals = self.get_params_for_obs_angle(obsangle,t)
            z = vals[1]
            r = vals[2]
            x = vals[3]
            theta = np.degrees(np.arccos(z/r))
            
            Sval = S[dusttype].S_cm2(theta,lamb) 
            Svals.append(Sval)
            
            factor_difference = Sval_of_known/Sval
            factors.append(factor_difference)
            
            mag_difference = math.log(factor_difference,100**(1/5))
        
            mag_candidate = 22.5 + mag_difference ## known one is 22.5 
            mags.append(mag_candidate)
        
        return mags,factors
    
    def plot_scattering_function_for_obsangles(self,obsangle1,obsangle2,obsangle3,obsangle4,
                             obsangle5,obsangle6,obsangle7,tmin,tmax,step):
        tvals = np.arange(tmin,tmax,step)
    
        plt.figure(figsize=(6,6))
        plt.rc('axes', linewidth=2)
        plt.plot(tvals,self.scattering_function(obsangle1,tmin,tmax,step)[0],
                 label=r'Obs Angle =%s$^o$' % obsangle1)
        plt.plot(tvals,self.scattering_function(obsangle2,tmin,tmax,step)[0],
                 label=r'Obs Angle =%s$^o$' % obsangle2)
        plt.plot(tvals,self.scattering_function(obsangle3,tmin,tmax,step)[0],
                 label=r'Obs Angle =%s$^o$' % obsangle3)
        plt.plot(tvals,self.scattering_function(obsangle4,tmin,tmax,step)[0],
                 label=r'Obs Angle =%s$^o$' % obsangle4)
        plt.plot(tvals,self.scattering_function(obsangle5,tmin,tmax,step)[0],
                 label=r'Obs Angle =%s$^o$' % obsangle5)
        plt.plot(tvals,self.scattering_function(obsangle6,tmin,tmax,step)[0],
                 label=r'Obs Angle =%s$^o$' % obsangle6)
        plt.plot(tvals,self.scattering_function(obsangle7,tmin,tmax,step)[0],
                 label=r'Obs Angle =%s$^o$' % obsangle7)
        
        plt.xlabel('Age',fontsize=18)
        plt.ylabel('Vega Magnitude / arcsec^2',fontsize=18)
        plt.tight_layout()
        plt.legend()
       
    ##############################################################
    ########### MULTIPLY THE FACTORS ############################
    
    def both_factors(self,obsangle,tmin,tmax,step):
        
        dist_factors = self.keeping_obs_angle_constant(obsangle,tmin,tmax,step)[1]
        S_factors = self.scattering_function(obsangle,tmin,tmax,step)[1]
        
        factors_combined = [dist_factors[i] * S_factors[i] for i in range(0,len(dist_factors))]


        mag_difference = [math.log(i,100**(1/5)) for i in factors_combined]
        
        mags = [22.5 + i for i in mag_difference]
        
        return mags, factors_combined 
    
    def plot_both_factors_for_obsangles(self,obsangle1,obsangle2,obsangle3,obsangle4,
                             obsangle5,obsangle6,obsangle7,tmin,tmax,step):
        
        tvals = np.arange(tmin,tmax,step)
        
        plt.figure(figsize=(6,6))
        plt.rc('axes', linewidth=2)
        plt.plot(tvals,self.both_factors(obsangle1,tmin,tmax,step)[0], color='k')
                 #label='Candidate1')
        plt.plot(tvals,self.both_factors(obsangle2,tmin,tmax,step)[0], color='k')
                 #label='Candidate 2')
        plt.plot(tvals,self.both_factors(obsangle3,tmin,tmax,step)[0], color='k')
                 #label='Candidate 3')
        plt.plot(tvals,self.both_factors(obsangle4,tmin,tmax,step)[0], color='k')
                 #label='Candidate 4')
        plt.plot(tvals,self.both_factors(obsangle5,tmin,tmax,step)[0], color='k')
                 #label='Candidate 5')
        plt.plot(tvals,self.both_factors(obsangle6,tmin,tmax,step)[0], color='k')
                 #label='Candidate 6')
        plt.plot(tvals,self.both_factors(obsangle7,tmin,tmax,step)[0], color='k',
                label='Candidates')
        plt.axvline(x=10000,linestyle='--',color='b',label='Candidates Published \nMinimum Age')
        plt.axhline(y=23.7,color='r',label='Approx. LE Surface \nBrightness')
        plt.xlabel('Age',fontsize=18)
        plt.ylabel(r'Surface Brightness (mag/arcsec$^2$)',fontsize=18)
        plt.tight_layout()
        plt.legend(loc='lower right',prop={'size': 15})
        plt.show()
        
    def plot_factors_for_obsangles(self,obsangle1,obsangle2,obsangle3,obsangle4,
                             obsangle5,obsangle6,obsangle7,tmin,tmax,step):
            
       tvals = np.arange(tmin,tmax,step)
        
       plt.figure(figsize=(6,6))
       plt.rc('axes', linewidth=2)
       plt.plot(tvals,self.both_factors(obsangle1,tmin,tmax,step)[1], color='k')
                 #label='Candidate1')
       plt.plot(tvals,self.both_factors(obsangle2,tmin,tmax,step)[1], color='k')
                 #label='Candidate 2')
       plt.plot(tvals,self.both_factors(obsangle3,tmin,tmax,step)[1], color='k')
                 #label='Candidate 3')
       plt.plot(tvals,self.both_factors(obsangle4,tmin,tmax,step)[1], color='k')
                 #label='Candidate 4')
       plt.plot(tvals,self.both_factors(obsangle5,tmin,tmax,step)[1], color='k')
                 #label='Candidate 5')
       plt.plot(tvals,self.both_factors(obsangle6,tmin,tmax,step)[1], color='k')
                 #label='Candidate 6')
       plt.plot(tvals,self.both_factors(obsangle7,tmin,tmax,step)[1], color='k',
                label='Candidates')
       plt.axvline(x=10000,linestyle='--',label='Candidates Published \nMinimum Age')
       #plt.axhline(y=24,color='r',label='Approx. LE Surface \nBrightness')
       plt.xlabel('Age',fontsize=18)
       plt.ylabel('Factor Difference from Brightest \nKnown LEs in LMC',fontsize=18)
       plt.tight_layout()
       plt.legend(loc='upper right',prop={'size': 15})
       
    def Scat_function_value(self,t,theta):
        S = {}
        lamb = 7000
        
        Sval_of_known = 9.354e-23
        ## the scattering function stuff
        dusttype = 'LMCavg'
        S[dusttype]=S_tableclass()
        dustfilename =  S[dusttype].getdustfilename(dusttype)
        #print('Loading dust properties...')
        S[dusttype].loadtable(dustfilename)
        #if lamb<wmin[dusttype] or lamb>wmax[dusttype]: continue 
        ##########
        
        #mag_candidate, z, r, x, factor_difference = self.get_params_for_obs_angle(obsangle,t)
        
        #theta = np.degrees(np.arccos(z/r))
        
        Sval = S[dusttype].S_cm2(theta,lamb)
        
        return Sval
        
    def get_SB_vs_obsangle_for_given_age(self,t,obsanglemin,obsanglemax,step):
        
        S = {}
        wmin={'LMCavg':4000.0,'LMC2':4000.0,'SMCbar':4000.0,'MWG':3500.0}
        wmax={'LMCavg':7999.0,'LMC2':7999.0,'SMCbar':7999.0,'MWG':9999.0}
        lamb = 7000
        
        Sval_of_known = 9.354e-23
        
        ## the scattering function stuff
        dusttype = 'LMCavg'
        S[dusttype]=S_tableclass()
        dustfilename =  S[dusttype].getdustfilename(dusttype)
        #print('Loading dust properties...')
        S[dusttype].loadtable(dustfilename)
        #if lamb<wmin[dusttype] or lamb>wmax[dusttype]: continue 
        ##########
        
        ### hold some values, especially for sanity checks 
        Svals = []
        factors = []
        mag_vals = []
        Sfactor = []
        thetas = []
        zvals = []
        dist_factors = []
        
        obsangles = np.arange(obsanglemin,obsanglemax,step)
        for obsangle in obsangles:
            
            ### get the factor from the 1/r^2 relationship (WRITTEN ABOVE)
            mag_candidate, z, r, x, factor_difference, theta = self.get_params_for_obs_angle(obsangle,t)
            zvals.append(z)
            dist_factors.append(factor_difference)
            
            theta = np.degrees(np.arccos(z/r))
            thetas.append(theta)
            
            #### FROM THE ORIGINAL SCRIPT TO CALCULATE THE S VALUE 
            Sval = S[dusttype].S_cm2(theta,lamb)
            Svals.append(Sval)
            
            Sval_factor_difference = Sval_of_known/Sval
            Sfactor.append(Sval_factor_difference)
            
            full_factor_difference = factor_difference * Sval_factor_difference
            factors.append(full_factor_difference)
            
            mag_difference = math.log(full_factor_difference,100**(1/5))
        
            mag = 22.5 + mag_difference ## 22.5 was the actual event 
            mag_vals.append(mag)
       
        return mag_vals, Sfactor, dist_factors, factors, thetas, zvals, Svals 
        
    ### THE PLOT OF SB VS OBSANGLES FOR DIFFERENT AGES 
    def plot_SB_vs_obsangle_for_given_age(self,t1,t2,t3,t4,t5,t6,t7,obsanglemin,obsanglemax,step): 
        
        obsangles = np.arange(obsanglemin,obsanglemax,step)
        
        plt.figure(figsize=(6,6))
        plt.rc('axes',linewidth=2)
        plt.plot(obsangles,self.get_SB_vs_obsangle_for_given_age(t1,obsanglemin,obsanglemax,step)[0],
                 color='r',label='%s yrs' % t1)
        plt.plot(obsangles,self.get_SB_vs_obsangle_for_given_age(t2,obsanglemin,obsanglemax,step)[0],
                 color='orange',label='%s yrs' % t2)
        plt.plot(obsangles,self.get_SB_vs_obsangle_for_given_age(t3,obsanglemin,obsanglemax,step)[0],
                 color='y',label='%s yrs' % t3)
        plt.plot(obsangles,self.get_SB_vs_obsangle_for_given_age(t4,obsanglemin,obsanglemax,step)[0],
                 color='g',label='%s yrs' % t4)
        plt.plot(obsangles,self.get_SB_vs_obsangle_for_given_age(t5,obsanglemin,obsanglemax,step)[0],
                 color='b',label='%s yrs' % t5)
        plt.plot(obsangles,self.get_SB_vs_obsangle_for_given_age(t6,obsanglemin,obsanglemax,step)[0],
                 color='purple',label='%s yrs' % t6)
        plt.plot(obsangles,self.get_SB_vs_obsangle_for_given_age(t7,obsanglemin,obsanglemax,step)[0],
                 color='m',label='%s yrs' % t7)
        plt.axhline(y=23.3,color='k',linestyle='--',label='Approx. LE Surface \nBrightness')
        plt.xlabel('Observed Angular Separation',fontsize=18)
        plt.ylabel(r'Surface Brightness (mag/arcsec$^2$)',fontsize=18)
        plt.tight_layout()
        plt.legend(loc='lower right',prop={'size': 15})
        
    #### THE PLOT OF Z DIMENSION WITH OBS ANGLE ####################    
    def plot_z_vs_obsangle_for_given_age(self,t1,t2,t3,t4,t5,t6,t7,obsanglemin,obsanglemax,step): 
        
        obsangles = np.arange(obsanglemin,obsanglemax,step)
        
        plt.figure(figsize=(6,6))
        plt.rc('axes',linewidth=2)
        plt.plot(obsangles,self.get_SB_vs_obsangle_for_given_age(t1,obsanglemin,obsanglemax,step)[5],
                 color='r',label='%s yrs' % t1)
        plt.plot(obsangles,self.get_SB_vs_obsangle_for_given_age(t2,obsanglemin,obsanglemax,step)[5],
                 color='orange',label='%s yrs' % t2)
        plt.plot(obsangles,self.get_SB_vs_obsangle_for_given_age(t3,obsanglemin,obsanglemax,step)[5],
                 color='y',label='%s yrs' % t3)
        plt.plot(obsangles,self.get_SB_vs_obsangle_for_given_age(t4,obsanglemin,obsanglemax,step)[5],
                 color='g',label='%s yrs' % t4)
        plt.plot(obsangles,self.get_SB_vs_obsangle_for_given_age(t5,obsanglemin,obsanglemax,step)[5],
                 color='b',label='%s yrs' % t5)
        plt.plot(obsangles,self.get_SB_vs_obsangle_for_given_age(t6,obsanglemin,obsanglemax,step)[5],
                 color='purple',label='%s yrs' % t6)
        plt.plot(obsangles,self.get_SB_vs_obsangle_for_given_age(t7,obsanglemin,obsanglemax,step)[5],
                 color='m',label='%s yrs' % t7)
        #plt.axhline(y=1630,linestyle='--',color='k',label='Edge of LMC')
        #plt.axhline(y=-1630,linestyle='--',color='k')
        plt.axhspan(-1630,1630,color='skyblue',alpha=0.5,label='Thickness of LMC')
        #plt.axhline(y=2000,linestyle='-.',color='k',label='Edge of 30 Dor')
        plt.xlabel('Observed Angular Separation',fontsize=18)
        plt.ylabel('z (lyrs)',fontsize=18)
        plt.tight_layout()
        plt.legend(loc='upper left',prop={'size': 15})
        
        
    ######## A SANITY CHECK PLOT ###############    
    def plot_SB_factors_for_given_age(self,t,obsanglemin,obsanglemax,step):
        
        obsangles = np.arange(obsanglemin,obsanglemax,step)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        
        #plt.figure(figsize=(6,6))
        #plt.rc('axes',linewidth=2)
        ax1.plot(obsangles,self.get_SB_vs_obsangle_for_given_age(t,obsanglemin,obsanglemax,step)[3],
                 color='r',label='%s yrs' % t)
        ax1.set_xlabel('Observational Angle',fontsize=18)
        
        ax2 = ax1.twiny()
        
        ax2.plot(self.get_SB_vs_obsangle_for_given_age(t,obsanglemin,obsanglemax,step)[4],
                 self.get_SB_vs_obsangle_for_given_age(t,obsanglemin,obsanglemax,step)[3],
                 color='r',label='%s yrs' % t)
        
        xticksneeded = np.array([obsangles[0],obsangles[10],obsangles[20],
                                 obsangles[30],obsangles[40],obsangles[50]])
        ax2.set_xticks(xticksneeded)
        
        ## round the values 
        thetavalsneeded = [np.round(np.degrees(np.arccos(self.get_params_for_obs_angle(xticksneeded[i],t)[1]/
                            self.get_params_for_obs_angle(xticksneeded[i],t)[2])),1) for i in range(0, len(xticksneeded))] ## how to get theta
        
        ax2.set_xticklabels(thetavalsneeded)
        #ax2.set_xlabel('Theta',fontsize=18)
        ax1.set_ylabel('Factors Multiplied Together',fontsize=18)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xlabel('Theta')
        
        print(thetavalsneeded)
    
        #ax2.plot(self.get_SB_vs_obsangle_for_given_age(t,obsanglemin,obsanglemax,step)[4],
        #         self.get_SB_vs_obsangle_for_given_age(t,obsanglemin,obsanglemax,step)[3],
        #         color='r')
        #ax2.set_xlabel('Theta')
        
        
        plt.tight_layout()
        plt.legend(loc='upper right',prop={'size': 15})
        

        
    def make_area_prob_plot(self):
        
        #plt.axvspan(0, 0.49475, alpha=0.2, color='gray',label='400')
        plt.axvspan(0, 0.49475, alpha=0.2, color='gray',label='600')
        plt.axvspan(0, 0.4435, alpha=0.2, color='gray',label='800')
        #plt.axvspan(0, 0.4267, alpha=0.2, color='gray',label='200')
        plt.axvspan(0, 0.3665, alpha=0.2, color='gray',label='1000')
        plt.axvspan(0, 0.245, alpha=0.2, color='gray',label='1200')
        
        plt.axhspan(500,1200, alpha=0.2, color='gray', label='Ages')
        plt.axhspan(500,1000, alpha=0.2, color='gray', label='Ages')
        plt.axhspan(500,800, alpha=0.2, color='gray', label='Ages')
        plt.axhspan(500,600, alpha=0.2, color='gray', label='Ages')
        
        plt.xlim(-0.1,1.1)
        plt.ylim(0,2000)
        plt.xlabel('Observed Angular Separation')
        plt.ylabel('Age')
        
    #def make_better_area_prob_plot(self):
        
        #fill([0,0])
        
    ######## MAKE A CONTOUR PLOT WITH AGE AND ANG SEP AS INDEP VARS #######
    def contour_age_ang_sep_surface_brightness(self,minAge,maxAge,stepAge,minAng,maxAng,stepAng):
        
        x = np.linspace(minAge,maxAge,5000)
        y = np.linspace(minAng,maxAng,5000)
        X,Y = np.meshgrid(x,y)
        
        #Z = np.empty_like(X)
        #for lengthIter in range(0,len(X)):
        #    for xIter in range(0,len(X[0])):
        #        Z[lengthIter][xIter] = self.get_params_for_obs_angle(Y[lengthIter][0],X[lengthIter][xIter])[0] ## the 0 is to get the mag
        
        Zzvals = np.empty_like(X)
        for lengthIter in range(0,len(X)):
            for xIter in range(0,len(X[0])):
                Zzvals[lengthIter][xIter] = self.get_params_for_obs_angle(Y[lengthIter][0],X[lengthIter][xIter])[1]
        
        #Zthetavals = np.empty_like(X)
        #for lengthIter in range(0,len(X)):
        #    for xIter in range(0,len(X[0])):
        #        Zthetavals[lengthIter][xIter] = self.get_params_for_obs_angle(Y[lengthIter][0],X[lengthIter][xIter])[5] ## this is for theta 
        
        
        #points = []
        #values = []
        #for i in range(0,len(x)):
        #    item = [x[i], y[i]]
        #    points.append(item)
        #    values.append(self.get_params_for_obs_angle(y[i],x[i])[0])
        #points = np.array(points)
        #values = np.array(values)
        #print(points)
        #grid_z0 = griddata(points, values, (X, Y), method='nearest')
        
        
        plt.figure()
        plt.rc('text', usetex=True)
        #plt.imshow(grid_z0.T, extent=[minAge,maxAge,minAng,maxAng], origin='lower',interpolation='none',aspect='auto')
        #plt.title('Nearest')
        
        
        plt.imshow(Zzvals,extent=[minAge,maxAge,minAng,maxAng], vmin=-8000, vmax=8000, origin='lower',interpolation='none',aspect='auto',cmap='RdBu') 
        # vmin=-8000, vmax=8000,
        
        
        cb = plt.colorbar()
        cb.set_label(r"z (lyrs)", fontsize=20,fontweight='bold')
        #cb.set_label(r"Surface Brightness (mag/arcsec$^2$)", fontsize=20,fontweight='bold')
        #SUGGESTION = plt.contour(X,Y,Z,levels=[23.7,23.7+0.5,23.7+1.5],colors=['k','k','k'],alpha=0.5,linewidths=[3,2,1])
        
        
        
        CS = plt.contour(X,Y,Zzvals,levels=[3*-815,-1630,-815,0,815,1630,3*815],colors=['black','black',
                          'g','g','g','black','black'],linewidths=[0.5,1,2,2,2,1,0.5])
        plt.clabel(CS, fmt = '%2.1d', colors = 'k', fontsize=14) #contour line labels
        #plt.colorbar()
        
        #THETAS = plt.contour(X,Y,Zthetavals,levels=[45,75,90,100],colors=['skyblue','skyblue','skyblue','skyblue'],
        #                     linewidths=[1,2,3,2])
        #plt.clabel(THETAS, fmt = '%2.1d', colors = 'k', fontsize=14) #contour line labels
        
        
        
        plt.plot([10000],[0.85],marker='o',color='k') ## candidate 2
        plt.plot([12500],[1.49],marker='o',color='k') ## candidate 3
        plt.plot([12500],[1.44],marker='o',color='k') ## candidate 4
        plt.plot([25000],[1.92],marker='o',color='k') ## candidate 5
        plt.plot([25000],[1.18],marker='o',color='k') ## candidate 6
        
        
        #### If you want error bars on the ages 
        #plt.errorbar([10000],[0.85],xerr=3000,marker='o',color='white') ## candidate 2
        #plt.errorbar([12500],[1.49],xerr=3000,marker='o',color='white') ## candidate 3
        #plt.errorbar([12500],[1.44],xerr=3000,marker='o',color='white') ## candidate 4
        #plt.errorbar([25000],[1.921],xerr=5000,marker='o',color='white') ## candidate 5
        #plt.errorbar([25000],[1.18],xerr=5000,marker='o',color='white') ## candidate 6
        
        
        plt.xlabel('Age',fontsize=20,fontweight='bold')
        plt.ylabel('Angular Separation',fontsize=20,fontweight='bold')
        plt.ylim(0,2)
        plt.xscale('log')
        plt.show()
        
        
    def plot_Scat_function_candidates(self):
        
        allThetas = GetThetas.returnallThetas()
        
        angles = np.arange(0,179,0.5)
        
        val90 = self.Scat_function_value(500,90)
        SvalsNorm = [self.Scat_function_value(500,i)/val90 for i in angles]
        
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(angles,SvalsNorm,color='g')
        plt.axvline(x = 153.6, color='b')
        plt.axvline(x = 143.1, color='b')
        plt.axvline(x = 144.4, color='b')
        plt.axvline(x = 155.7, color='b')
        plt.axvline(x = 164, color='b', label='Candidates')
        
        plt.axvline(x = 48, linestyle='dashed', color='r',label=r'Brightnest Known SNR LE Example')
        
        #plt.axvspan(30,110,color='k',alpha=0.25,label='Observed LEs from Literature')
        plt.hist(allThetas)
        
        plt.xlabel(r'$\theta$', fontsize=20)
        plt.ylabel(r'Scattering Efficiency (Norm. to 90$^o$)',fontsize=20)
        
        plt.legend(loc='upper right',fontsize=16)
        plt.yscale('log')
        
    def plot_z_vs_Ang_Sep_for_ages(self,obsanglemin, obsanglemax, step):
        
        obsangles = np.arange(obsanglemin,obsanglemax,step)
        ages = np.arange(200,5000,100)
        alphas = np.linspace(0,1,len(ages))
        
        plt.figure(figsize=(6,6))
        plt.rc('axes',linewidth=2)
        for i in range(0,len(ages)):
            plt.plot(obsangles,self.get_SB_vs_obsangle_for_given_age(ages[i],obsanglemin,obsanglemax,step)[5],
                 color='b',alpha = alphas[i])
        
        plt.axhspan(-1630,1630,color='r',alpha=0.5,label='Thickness of LMC')
        plt.ylabel('z (ly)',fontsize=20,fontweight='bold')
        plt.xlabel('Angular Separation',fontsize=20,fontweight='bold')
        plt.legend(fontsize=18)
        plt.show()
        
        
       
        

