import numpy as np
import timeit
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import lambertw
import math
import seaborn as sns
import pickle
import time


user_params = {'lambda': 3, 'gamma': 2,'hetero':1.6}
capacity = 45
Tstep = 1
#unusual = {"unusual":True,'day':10,"capacity": 38,"read":'Trinitytt.npy','ctrlprice':1.55,'regulatesTime':400,'regulateeTime':480,
#		   "dropstime":360,"dropetime":480}
#1.02557013 327.8359478  371.82177488
unusual = {"unusual":False,'day':10,"capacity": 38,"read":'Trinitytt.npy','ctrlprice':2.63,'regulatesTime':415,'regulateeTime':557,
		   "dropstime":360,"dropetime":480,"FTC": 0.05,"AR":0.0}
storeTT = {'flag':False,"ttfilename":'Trinitylumptt'}

deltaP = 0.05
RBTD = 100
Plot = True
seed = 333
verbose = True
 # Policy: False, uniform, personalizedCap, personalizedNocap
numOfusers = 7500 
numOfdays = 100

def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

class Travelers():
    # user parameters
    # user accounts
    # predicted departure times
    # update trip intentions
    # wihtin day mobility decisions
    # sell and buy
    # compute user account 
    # distance is asummed to be 16miles
    def __init__(self,_numOfusers,_user_params,_allowance,_allocation,_scenario,_hoursInA_Day = 12,_Tstep = 1,
                _fftt=24,_dist=18,_choiceInterval = 30,_seed=5843,_unusual=False,_CV=True,_numOfdays=50):

        self.AR = _allocation['AR']
        self.ARway = _allocation['way']
        self.FTCs = _allocation['FTCs']
        self.FTCb = _allocation['FTCb'] # fixed transaction fees
        self.PTCs = _allocation['PTCs']
        self.PTCb = _allocation['PTCb'] # proportional transaction fees  

        self.users = np.arange(_numOfusers)
        self.numOfusers = _numOfusers
        self.hoursInA_Day = _hoursInA_Day
        self.Tstep = _Tstep
        self.fftt = _fftt
        self.dist = _dist
        self.mpg = 23
        self.fuelprice = 4
        self.ptfare = 2
        self.ptspeed = 25
        self.pttt = self.dist/self.ptspeed*60 # in minutes
        self.ptheadway = 10

        self.choiceInterval = _choiceInterval
        self.user_params = _user_params
        self.seed = _seed
        self.allowance = _allowance
        self.scenario = _scenario
        self.unusual = _unusual
        self.CV = _CV
        self.numOfdays = _numOfdays
        self.decaying = _allocation['Decaying']
        if self.CV:
            self.NT_sysutil = np.load("./output/Bottleneck/NT/NT_sysutil.npy")
        self.userCV = np.zeros(self.numOfusers)

        # initialize user accounts
        self.userAccounts = np.zeros(self.numOfusers)+self.AR*self.hoursInA_Day*60
        self.distribution = np.zeros(self.numOfusers)

    def interpolatePDT(self):
        x = np.array([390, 405, 420, 435, 450, 465, 480, 495, 510, 525 ,540, 555], dtype=float)
        # from https://link.springer.com/article/10.1007/s11116-016-9750-2
        y = np.array([0.06, 0.09, 0.095, 0.098, 0.098, 0.11, 0.095, 0.085, 0.08, 0.07,0.059,0.061 ])
        from scipy.interpolate import interp1d
        f = interp1d(x, y)
        return f

    def generatePDT(self):
        np.random.seed(seed=self.seed)
        # rejection sampling
        def batch_sample(function, num_samples, xmin=390, xmax=555, ymax=0.15, batch=1000):
            samples = []
            while len(samples) < num_samples:
                x = np.random.uniform(low=xmin, high=xmax, size=batch)
                y = np.random.uniform(low=0, high=ymax, size=batch)
                samples += x[y < function(x)].tolist()
            return samples[:num_samples]
        f = self.interpolatePDT()
        samps = batch_sample(f, self.numOfusers)
        return np.array(samps).astype(int)

    # generate travelers parameters (vot, sde, sdl, mu, epsilon, income)
    # and trip intentions (desired arrival time, choice set)
    def newvot(self,cov = 1.6):
        np.random.seed(seed=self.seed)
        if cov == 0.2:
            newvot = np.random.lognormal(-2.2,0.2,self.numOfusers)
        elif cov == 0.9:
            newvot = np.random.lognormal(-2.2,0.78,self.numOfusers)
        return newvot

    
    def generate_params(self):
        np.random.seed(seed=self.seed)
        self.betant = 20
        # self.sde = np.random.lognormal(-2.5,1.33,self.numOfusers)#np.random.lognormal(-2.7,1.28,self.numOfusers)
        # self.sde = self.sde + 0.04 # right shifted lognormal
        # # self.sde = np.random.normal(np.mean(np.log(self.sde)),np.std(np.log(self.sde))/1.5,self.numOfusers)
        # # self.sde = np.exp(self.sde)
        # self.vot = self.sde*np.exp(0.4)#np.exp(0.75)
        # self.sdl = self.sde*np.exp(0.9)#np.exp(1.38)
        # self.sde = self.sde*1.5
        # self.vot = self.vot*1.2
        # self.sdl = self.sdl*1.3
        self.vot = np.exp(np.random.normal(5.104019892828459, 1.1311644092299618,self.numOfusers))/8/60/3
        annualincome = self.vot*3*60*8* 260
        if self.user_params['hetero'] != 1.6:
            newvot = self.newvot(cov = self.user_params['hetero'])
            self.vot[np.argsort(annualincome)] = newvot[np.argsort(newvot)]
        ratiomu = np.random.triangular(0.1,0.5,1,self.numOfusers)
        self.sde = self.vot*ratiomu #self.vot/np.exp(0.3)
        ratioeta = np.random.triangular(1,2,3,self.numOfusers)
        self.sdl = self.vot*ratioeta#self.vot*np.exp(0.7)
        self.waiting = self.vot*3


        self.mu = np.random.lognormal(-0.8,0.47,self.numOfusers)# avg mu as 0.7, co.v as 0.5
        #np.random.lognormal(-0.12,0.5,self.numOfusers) # avg mu as 1, cov. as 0.5 #np.random.lognormal(-0.8,1.3,self.numOfusers)
        # -0.12,0.5 c.o.v as 0.5; -0.29,0.81 c.o.v as 1; -0.8,1.3 c.o.v. as 2
        # self.mu = np.random.normal(np.mean(np.log(self.mu)),np.std(np.log(self.mu))/1.5,self.numOfusers)
        # self.mu = np.exp(self.mu)
        self.mu = np.maximum(self.mu,0.005)
        self.user_eps = np.zeros((self.numOfusers,int(self.hoursInA_Day*60/self.Tstep)+1)) # one extra for no travel
        for i in range(self.numOfusers):
            self.user_eps[i,:] = np.random.gumbel(-0.57721 / self.mu[i], 1.0 /self.mu[i], (1,int(self.hoursInA_Day*60/self.Tstep)+1)) # one extra for no travel
        
        self.I = np.maximum((annualincome/260)-0.6*7.25*8,0.4*7.25*8)
        print("annual income ginit",gini(annualincome),"remaining I gini",gini(self.I))	

        self.desiredArrival = self.generatePDT()+self.fftt
        # generate predicted travel
        if self.unusual['read'] and self.unusual['unusual']:
            self.predictedTT = np.load(self.unusual['read'])
            self.actualTT =  np.load(self.unusual['read'])
            # self.desiredDeparture = np.load(self.unusual['departure'])
        else:
            self.predictedTT = np.array([self.fftt]*self.hoursInA_Day*60)
            self.actualTT = np.array([self.fftt]*self.hoursInA_Day*60)
        # generate desired departure time
        self.desiredDeparture = self.desiredArrival-self.fftt
        self.DepartureLowerBD = self.desiredDeparture-self.choiceInterval
        self.DepartureUpperBD = self.desiredDeparture+self.choiceInterval
        self.predayDeparture= np.zeros(self.numOfusers,dtype=int) + self.desiredDeparture

    def update_TC(self,FTCs,FTCb,PTCs,PTCb):
        self.FTCs = FTCs
        self.FTCb = FTCb
        self.PTCs = PTCs
        self.PTCb = PTCb

    def user_optimization(self,distribution, x,Th, I,sys_util_t,user,utileps):
        # Th is the product of toll and price
        # calculate logsum, which is slow
        numOfdist = len(distribution)
        numOfcost = len(Th)
        a_exp = 2*np.repeat(distribution,numOfcost).reshape(numOfdist,numOfcost)
        # cost is positive
        cost = np.tile(Th,(numOfdist,1))
        income_c = self.user_params['lambda']*np.log(self.user_params['gamma']+I-cost+a_exp)
        sys_util_c = -cost+income_c+a_exp
        sys_util = sys_util_t+sys_util_c
        util = sys_util+utileps
        idx = np.argmax(util,axis=1)
        MUI = 1+self.user_params['lambda']/(self.user_params['gamma']+I-Th[idx]+distribution)
        a_index = (np.abs(MUI-x)).argmin()
        # obj = 1/self.mu[user]*np.log(np.sum(np.exp(self.mu[user]*sys_util),axis=1))-x*distribution
        # a_index = np.argmax(obj)
        return a_index

    def calculate_swcv(self,_mu,s,n,choicen,sys_util_nt,sys_util_t, cost,dailyincome):
        # obsolete, need to update
        # s: allowance
        user_eps = np.random.gumbel(-0.57721 /_mu , 1.0 /_mu, (n,choicen))
        #ans[i] = optimize.bisect(welfare_diff,-200, 1+dailyincome+s-np.min(c),args=(s,sys_util_nt,sys_util_t,dailyincome,user_eps,c))
        a = 1
        b = self.user_params['lambda']
        num_s = len(s)
        s = np.array(s)
        s_exp = np.repeat(s,n*choicen).reshape(n*num_s,choicen)
        n = num_s*n
        user_eps = np.tile(user_eps,(num_s,1))

        util_nt = np.tile(sys_util_nt,(n,1))+user_eps
        max_util_nt = np.max(util_nt,axis=1)
        sys_util_cp = np.tile(sys_util_t-cost,(n,1))+user_eps+s_exp
        cost = np.tile(cost,(n,1))

        util_cp = np.tile(sys_util_t,(n,1))-cost+user_eps+s_exp+self.user_params['lambda']*np.log(self.user_params['gamma']+dailyincome-cost+s_exp)
        max_cp_idx = np.argmax(util_cp,axis=1)
        c = sys_util_cp[np.arange(n),max_cp_idx]-max_util_nt-1-dailyincome+cost[np.arange(n),max_cp_idx]-s_exp[np.arange(n),max_cp_idx]

        cv1 = np.max(sys_util_cp,axis=1)-np.max(util_nt-self.user_params['lambda']*np.log(self.user_params['gamma']+dailyincome),axis=1)
        x = b/a*lambertw((a/b)*np.exp(-c/b))
        cv2 = 1+dailyincome-cost[np.arange(n),max_cp_idx]-x+s_exp[np.arange(n),max_cp_idx]
        cv = np.where(np.max(-c/b)>300,cv1,cv2)
        ans_avg = np.average(cv.reshape(num_s,int(n/num_s)),axis = 1)[0]

        return ans_avg
    # perturbP: in order to calculate elasticity (only support for no toll scenario)
    def elasticity_util(self,_predictedTT, _beginTime, _totTime,_toll,RR,price,day,perturbP=0,perturbF=0,tollelast=False,fuelelast=True,perturbT = (420,481)):
        np.random.seed(seed=self.seed)

        predayDeparture = np.zeros(self.numOfusers)
        # aggregate by time interval
        _predictedTT = np.mean(_predictedTT.reshape(-1,self.Tstep),axis=1)
        # perturb toll

        toll = np.zeros(len(_toll))
        toll[:] = _toll[:]
        if tollelast:
            toll[int(perturbT[0]):int(perturbT[1])] = toll[int(perturbT[0]):int(perturbT[1])] + perturbP
        else:
            toll[int(perturbT[0]):int(perturbT[1])] = toll[int(perturbT[0]):int(perturbT[1])] 
        origtoll = toll
        toll = np.mean(toll.reshape(-1,self.Tstep),axis=1)

        sysutil_arr = np.zeros((self.numOfusers, self.choiceInterval*2+1+1)) # one additional slot for no travel
        fuelcost = self.dist/self.mpg*self.fuelprice
        for user in self.users:
            s1 = self.DepartureLowerBD[user]
            s2 = self.DepartureUpperBD[user]
            tstar = self.desiredArrival[user]
            vot = self.vot[user]
            sde = self.sde[user]
            sdl = self.sdl[user]
            vow = self.waiting[user]
            I = self.I[user]

            prev_allowance = self.distribution[user]
            # handle budget constraint
            if self.ARway == 'continuous':
                overBudget = np.where((origtoll-x[user,:])*pb+self.FTCb+fuelcost>I/2+prev_allowance)[0]
            else:
                overBudget = np.where((origtoll-self.userAccounts[user])*price+fuelcost>I/2+prev_allowance)[0]
            possibleDepartureTimes = np.arange(s1, s2 + 1, self.Tstep)
            possibleDepartureTimes = (possibleDepartureTimes/self.Tstep).astype(int)

            if len(overBudget)>0:
                overBudget = (overBudget/self.Tstep).astype(int)	
                #print((overBudget/self.Tstep).astype(int),possibleDepartureTimes)
                #print("intersect: ",np.intersect1d((overBudget/self.Tstep).astype(int),possibleDepartureTimes))
                #print(possibleDepartureTimes[~np.in1d(possibleDepartureTimes,(overBudget/self.Tstep))])
                originlen = len(possibleDepartureTimes)
                possibleDepartureTimes = possibleDepartureTimes[~np.in1d(possibleDepartureTimes,(overBudget/self.Tstep))]
                if len(possibleDepartureTimes) < originlen and len(possibleDepartureTimes)>0:
                    self.numOfbindingI += 1
                    self.bindingI.append(user)
                if len(possibleDepartureTimes) == 0 : 
                    firstOne = overBudget[0]
                    lastOne = overBudget[-1]
                    shift = s2-firstOne+1
                    s1 = max(s1-shift,0)
                    s2 = min(firstOne-1,self.hoursInA_Day*60)

                    possibleDepartureTimes = np.arange(s1, s2 + 1, self.Tstep)
                    possibleDepartureTimes = (possibleDepartureTimes/self.Tstep).astype(int) # round into time interval
            utileps = np.zeros(1+len(possibleDepartureTimes))
            utileps[0] = self.user_eps[user,-1] # add random term of no travel
            utileps[1:] = self.user_eps[user,possibleDepartureTimes] 
            

            tautilde = np.zeros(1+len(possibleDepartureTimes))
            tautilde[0] = self.pttt
            tautilde[1:] = _predictedTT[possibleDepartureTimes] # add tt of no travel
            Th = np.zeros(1+len(possibleDepartureTimes))
            Th[0] = self.ptfare
            Th[1:] = toll[possibleDepartureTimes]
            possibleDepartureTimes = possibleDepartureTimes * self.Tstep
            if self.ARway == "continuous":
                possibleAB = x[user,possibleDepartureTimes]
            SDE = np.zeros(1+len(possibleDepartureTimes))
            SDE[1:] = np.maximum(0,tstar-(possibleDepartureTimes+tautilde[1:]+self.Tstep/2)) # use middle point of time interval to calculate SDE
            SDL = np.zeros(1+len(possibleDepartureTimes))
            SDL[1:] = np.maximum(0,(possibleDepartureTimes+tautilde[1:]+self.Tstep/2)-tstar)# use middle point of time interval to calculate SDL
            ASC = np.zeros(len(utileps))
            W = np.zeros(1+len(possibleDepartureTimes)) # waiting time
            W[0] = 1/2*self.ptheadway
            # ASC[0] = self.betant
            sysutil_t = ASC+(-2*vot *tautilde  - sde * SDE - sdl * SDL-2*vow*W) # for all day, double travel time but not sde and sdl
            ch = np.zeros(len(utileps))
            # allowance
            if not self.allowance['policy']:
                self.distribution[user] = 0
            
            buy = Th*price
            sell = np.zeros_like(Th)
            ch = buy-sell-prev_allowance
            if not fuelelast:
                ch[1:] += fuelcost
            else:
                # print(possibleDepartureTimes,int(perturbT[0])<=possibleDepartureTimes)
                ch[1:] += fuelcost+perturbF
                # ch[1:] = np.where((int(perturbT[0])<=possibleDepartureTimes) & (possibleDepartureTimes <=int(perturbT[1])),ch[1:] +fuelcost+perturbF,ch[1:] +fuelcost)
            ch = ch*2 # double for all day
            sysutil =( sysutil_t+self.user_params['lambda']*np.log(self.user_params['gamma']+I-ch)+I-ch)
            util = sysutil+utileps

            if np.argmax(util) == 0: # choose no travel
                predayDeparture[user] = -1
            else:
                departuretime = possibleDepartureTimes[np.argmax(util)-1]+_beginTime
                departuretime = int(np.random.choice(np.arange(departuretime,departuretime+self.Tstep),1)[0])
                predayDeparture[user] = departuretime
        
        return predayDeparture
    # make preday choice according to attributes (travel time, schedule delay, toll)
    # check if budget constraint is satisfied
    # allowance distribution has three cases: no distribution, uniform distribution
    # and personalized distribution (which itself has two cases: cap and no cap)
    # no distribution and uniform distribution can be handled easily without additional work
    # personalization requires additional user level optimization to determine allowance
    def update_choice(self, _predictedTT, _beginTime, _totTime,_toll,RR,price,day):
        np.random.seed(seed=self.seed)


        self.todaySchedule = {i: [] for i in range(_beginTime, _totTime)}
        self.ptshare = []
        self.bindingI = []
        self.predayEps = np.zeros(self.numOfusers)
        self.numOfbindingI = 0

        pb = price*(1+self.PTCb)
        ps = price*(1-self.PTCs)
        if self.ARway == 'continuous':
            # predict today's account balances
            FW = self.AR*self.hoursInA_Day*60
            x = np.zeros((self.numOfusers, self.hoursInA_Day*60))
            x[:,0] = self.userAccounts
            td = np.where(self.predayDeparture!=-1, np.mod(self.predayDeparture,self.hoursInA_Day*60),-self.hoursInA_Day*60)
            Td = np.where(td!=-self.hoursInA_Day*60,_toll[td-td%self.Tstep], td)
            for t in range(self.hoursInA_Day*60-1):
                td = np.where(td!=-self.hoursInA_Day*60, np.where(td<t,td+self.hoursInA_Day*60,td)  ,td) 
                profitAtT = np.zeros(self.numOfusers)
                mask_cansell = td != t
                FA = np.where(td!=-self.hoursInA_Day*60,  np.minimum((td-t)*self.AR,FW), 0)
                if self.decaying:
                    profitAtT = x[:,t]*ps-(ps*(x[:,t])**2/(2*self.hoursInA_Day*60*self.AR))-self.FTCs-np.where(Td>FA,(Td-FA)*pb+self.FTCb,0)
                else:
                    profitAtT = x[:,t]*ps-self.FTCs-np.where(Td>FA,(Td-FA)*pb+self.FTCb,0)
                profitAtT[~mask_cansell] = 0.0

                mask_positiveprofit = profitAtT>1e-10
                mask_needbuy = Td>=FA
                mask_needbuynext = Td>=np.maximum(FA-self.AR,0)
                mask_FW = np.abs(x[:,t]-FW)<1e-10
                mask_sellnow = (mask_cansell&mask_positiveprofit)&(mask_needbuy | mask_FW|mask_needbuynext)
                x[mask_sellnow,t] = 0
                x[:,t+1] = np.maximum(FW,x[:,t]+self.AR)


        # aggregate by time interval
        _predictedTT = np.mean(_predictedTT.reshape(-1,self.Tstep),axis=1)
        origtoll = _toll
        _toll = np.mean(_toll.reshape(-1,self.Tstep),axis=1)
        
        sysutil_arr = np.zeros((self.numOfusers, self.choiceInterval*2+1+1)) # one additional slot for no travel
        th_arr = np.zeros((self.numOfusers, self.choiceInterval*2+1+1)) # one additional slot for no travel
        utileps_arr = np.zeros((self.numOfusers, self.choiceInterval*2+1+1)) # one additional slot for no travel
        sysutilt_arr = np.zeros((self.numOfusers, self.choiceInterval*2+1+1))

        fuelcost = self.dist/self.mpg*self.fuelprice
        for user in self.users:
            s1 = self.DepartureLowerBD[user]
            s2 = self.DepartureUpperBD[user]
            tstar = self.desiredArrival[user]
            vot = self.vot[user]
            sde = self.sde[user]
            sdl = self.sdl[user]
            vow = self.waiting[user]
            I = self.I[user]

            prev_allowance = self.distribution[user]
            # handle budget constraint
            if self.ARway == 'continuous':
                R = np.where(x[user,:]>=origtoll, (FW-origtoll)*ps-self.FTCs , (FW-x[user,:])*ps-self.FTCs-((origtoll-x[user,:])*pb+self.FTCb))
                overBudget = np.where(-R+fuelcost>I/2+prev_allowance)[0]
            else:
                overBudget = np.where((origtoll-self.userAccounts[user])*price+fuelcost>I/2+prev_allowance)[0]
            possibleDepartureTimes = np.arange(s1, s2 + 1, self.Tstep)
            possibleDepartureTimes = (possibleDepartureTimes/self.Tstep).astype(int)

            if len(overBudget)>0:
                overBudget = (overBudget/self.Tstep).astype(int)	
                #print((overBudget/self.Tstep).astype(int),possibleDepartureTimes)
                #print("intersect: ",np.intersect1d((overBudget/self.Tstep).astype(int),possibleDepartureTimes))
                #print(possibleDepartureTimes[~np.in1d(possibleDepartureTimes,(overBudget/self.Tstep))])
                originlen = len(possibleDepartureTimes)
                possibleDepartureTimes = possibleDepartureTimes[~np.in1d(possibleDepartureTimes,(overBudget/self.Tstep))]
                if len(possibleDepartureTimes) < originlen and len(possibleDepartureTimes)>0:
                    self.numOfbindingI += 1
                    self.bindingI.append(user)
                if len(possibleDepartureTimes) == 0 : 
                    firstOne = overBudget[0]
                    lastOne = overBudget[-1]
                    shift = s2-firstOne+1
                    s1 = max(s1-shift,0)
                    s2 = min(firstOne-1,self.hoursInA_Day*60)

                    possibleDepartureTimes = np.arange(s1, s2 + 1, self.Tstep)
                    possibleDepartureTimes = (possibleDepartureTimes/self.Tstep).astype(int) # round into time interval
            utileps = np.zeros(1+len(possibleDepartureTimes))
            utileps[0] = self.user_eps[user,-1] # add random term of no travel
            utileps[1:] = self.user_eps[user,possibleDepartureTimes] 
            

            tautilde = np.zeros(1+len(possibleDepartureTimes))
            tautilde[0] = self.pttt
            tautilde[1:] = _predictedTT[possibleDepartureTimes] # add tt of no travel
            Th = np.zeros(1+len(possibleDepartureTimes))
            Th[0] = self.ptfare
            Th[1:] = _toll[possibleDepartureTimes]
            possibleDepartureTimes = possibleDepartureTimes * self.Tstep
            if self.ARway == "continuous":
                possibleAB = x[user,possibleDepartureTimes]
            SDE = np.zeros(1+len(possibleDepartureTimes))
            SDE[1:] = np.maximum(0,tstar-(possibleDepartureTimes+tautilde[1:]+self.Tstep/2)) # use middle point of time interval to calculate SDE
            SDL = np.zeros(1+len(possibleDepartureTimes))
            SDL[1:] = np.maximum(0,(possibleDepartureTimes+tautilde[1:]+self.Tstep/2)-tstar)# use middle point of time interval to calculate SDL
            ASC = np.zeros(len(utileps))
            W = np.zeros(1+len(possibleDepartureTimes)) # waiting time
            W[0] = 1/2*self.ptheadway
            # ASC[0] = self.betant # no more no travel
            sysutil_t = ASC+(-2*vot *tautilde  - sde * SDE - sdl * SDL-2*vow*W) # for all day, double travel time but not sde and sdl
            ch = np.zeros(len(utileps))
            
            if self.scenario == 'Trinity':
                if self.ARway == 'lumpsum':
                    buy = np.maximum(Th-self.userAccounts[user], 0)*price
                    sell = np.maximum(self.userAccounts[user] - Th, 0)*price
                    ch[:] = buy-sell
                    ch[0] = Th[0]-np.maximum(self.userAccounts[user], 0)*price
                elif self.ARway == 'continuous':
                    # calculate opportunity cost
                    tempTh = Th[1:] # exclude the first element corresponding to no travel
                    if self.decaying:
                        ch[1:] = -np.where(possibleAB>=tempTh, (FW-tempTh)*ps-(ps*self.AR*((FW-tempTh)/self.AR)**2/(2*self.hoursInA_Day*60))-self.FTCs,
                                    np.maximum(-(tempTh-possibleAB)*pb-self.FTCb+(FW-possibleAB)*ps-(ps*self.AR*((FW-possibleAB)/self.AR)**2/(2*self.hoursInA_Day*60))-self.FTCs,0))
                        ch[0] = Th[0]-np.maximum((FW)*ps-(ps*self.AR*((FW)/self.AR)**2/(2*self.hoursInA_Day*60))-self.FTCs,0)
                    else:
                        #ch = np.where(possibleAB>=Th, Th*ps-self.FTCs,np.maximum((Th-possibleAB)*pb+self.FTCb+possibleAB*ps-self.FTCs,0))
                        # consider opportunity benefit and cost
                        ch[1:] = -np.where(possibleAB>=tempTh, (FW-tempTh)*ps-self.FTCs , (FW-possibleAB)*ps-self.FTCs-((tempTh-possibleAB)*pb+self.FTCb))
                        ch[0] = Th[0]-np.maximum((FW)*ps-self.FTCs,0)
                    
            else:
                buy = Th*price
                sell = np.zeros_like(Th)
                ch = buy-sell-prev_allowance
            # add fuel price
            ch[1:] += fuelcost
            ch = ch*2 # double for all day
            ch_woallowance = (ch/2+prev_allowance)*2
            # allowance
            if not self.allowance['policy']:
                self.distribution[user] = 0
            elif self.allowance['policy'] == 'uniform':
                self.distribution[user] = RR/self.numOfusers
            elif self.allowance['policy'] =='personalization':
                income_c = self.user_params['lambda']*np.log(self.user_params['gamma']+I-ch_woallowance)
                idx = np.argmax(sysutil_t+income_c-ch_woallowance+utileps)
                if self.user_params['lambda']/(self.user_params['gamma']+I-ch_woallowance[idx])+1<=self.allowance['ctrl']:
                    self.distribution[user] = 0
                else:
                    distribution = np.linspace(max(max(ch_woallowance-I),0),max(max(ch_woallowance-I),0)+30,num=300)
                    a_idx = self.user_optimization(distribution, self.allowance['ctrl'],ch_woallowance, I,sysutil_t,user,utileps)
                    self.distribution[user] = min(distribution[a_idx],self.allowance['cap'])

            sysutil =( sysutil_t+self.user_params['lambda']*np.log(self.user_params['gamma']+I-ch)+I-ch)
            util = sysutil+utileps
            # th_arr[user,:] = (ch/2+prev_allowance)*2
            # utileps_arr[user,:] = utileps
            # sysutilt_arr[user,:] = 	sysutil_t
            # calculate user cv
            if self.CV:
                if day == self.numOfdays - 1:
                    self.userCV[user] =  self.calculate_swcv(self.mu[user],[0],max(int(500/self.mu[user]),500),len(possibleDepartureTimes)+1,self.NT_sysutil[user,:], sysutil_t, ch,self.I[user])
                else:
                    self.userCV[user] = 0
        
            if np.argmax(util) == 0: # choose no travel
                self.ptshare.append(user)
                self.predayDeparture[user] = -1
            else:
                departuretime = possibleDepartureTimes[np.argmax(util)-1]+_beginTime
                departuretime = int(np.random.choice(np.arange(departuretime,departuretime+self.Tstep),1)[0])
                self.todaySchedule[departuretime].append({'user':user,'tstar':tstar,'departure':departuretime})
                self.predayDeparture[user] = departuretime
            # if user == 7459:
                # print("user 7559",sysutil,util,possibleDepartureTimes,I,ch,Th,prev_allowance,fuelcost,utileps)
            self.predayEps[user] = utileps[np.argmax(util)]
        np.save("./output/Bottleneck/NT/th_arr",th_arr)
        np.save("./output/Bottleneck/NT/utileps_arr",utileps_arr)
        np.save("./output/Bottleneck/NT/sysutilt_arr",sysutilt_arr)
        if self.scenario == 'NT':
            sysutil_arr[user,:] = sysutil
            np.save("./output/Bottleneck/NT/NT_sysutil",sysutil_arr)


    def withinday_choice(self, _predictedTT, _beginTime, _totTime,_toll,RR,price,_t,_originalAtt,_unusual):
        np.random.seed(seed=self.seed)

        todaySchedule = {i: [] for i in range(_beginTime, _totTime)}

        if self.ARway == 'continuous':
            regulatePrice = price
            regulateAR = self.AR
            regulateCs = self.PTCs
            regulateCb = self.PTCb
            regulateTCs = self.FTCs
            regulateTCb = self.FTCb
            regulateStartTime = int(_unusual['regulatesTime'])
            regulateEndTime = int(_unusual['regulateeTime'])

            originalAR = _originalAtt['AR']
            originalmarketPrice = _originalAtt['price']
            originalCs = _originalAtt['PTCs']
            originalCb = _originalAtt['PTCb']
            originalTCs = _originalAtt['FTCs']
            originalTCb = _originalAtt['FTCb']

            FW = originalAR*self.hoursInA_Day*60

            pb = np.array([originalmarketPrice*(1+originalCb)]*self.hoursInA_Day*60)
            pb[regulateStartTime:regulateEndTime] = regulatePrice*(1+regulateCb)
            ps = np.array([originalmarketPrice*(1-originalCs)]*self.hoursInA_Day*60)
            ps[regulateStartTime:regulateEndTime] = regulatePrice*(1-regulateCs)
            TCs = np.array([originalTCs]*self.hoursInA_Day*60)
            TCs[regulateStartTime:regulateEndTime] = regulateTCs
            TCb = np.array([originalTCb]*self.hoursInA_Day*60)
            TCb[regulateStartTime:regulateEndTime] = regulateTCb
            

            # predict today's account balances
            currt = np.mod(_t,self.hoursInA_Day*60)
            x = np.zeros((self.numOfusers, self.hoursInA_Day*60))
            x[:,currt] = self.userAccounts
            td = np.where(self.predayDeparture!=-1, np.mod(self.predayDeparture,self.hoursInA_Day*60),-self.hoursInA_Day*60)
            Td = np.where(td!=-self.hoursInA_Day*60,_toll[td-td%self.Tstep], td)
            for t in range(self.hoursInA_Day*60-currt-1):
                t = t + currt
                td = np.where(td!=-self.hoursInA_Day*60, np.where(td<t,td+self.hoursInA_Day*60,td)  ,td) 
                profitAtT = np.zeros(self.numOfusers)
                mask_cansell = td != t

                if t <= regulateStartTime:
                    S = x[:,t]*ps[t]-TCs[t]
                    mask_tdlessStart = td<regulateStartTime
                    mask_tdbwt = (td>=regulateStartTime)&(td <= regulateEndTime)
                    mask_tdgreaterEnd = td>regulateEndTime
                    futureallocation = np.zeros(self.numOfusers)
                    futureallocationNextT = np.zeros(self.numOfusers)
                    futureallocation[mask_tdlessStart] = (td[mask_tdlessStart] -t)*originalAR
                    futureallocationNextT[mask_tdlessStart] = futureallocation[mask_tdlessStart]-originalAR
                    futureallocation[mask_tdbwt] = (regulateStartTime-t)*originalAR+(td[mask_tdbwt]-regulateStartTime)*regulateAR
                    futureallocationNextT[mask_tdbwt] = futureallocation[mask_tdbwt]-regulateAR
                    futureallocation[mask_tdgreaterEnd] = (regulateStartTime-t)*originalAR+(regulateEndTime-regulateStartTime)*regulateAR + (td[mask_tdgreaterEnd]-regulateEndTime)*originalAR
                    futureallocationNextT[mask_tdgreaterEnd] = futureallocation[mask_tdgreaterEnd]-originalAR

                    futureallocation = np.minimum(futureallocation,FW)
                    futureallocationNextT = np.minimum(futureallocationNextT,FW)
                if (t <= regulateEndTime) and (t>=regulateStartTime):
                    S = x[:,t]*ps[t]-TCs[t]
                    futureallocation = np.where(td<=regulateEndTime,(td-t)*regulateAR,np.minimum((regulateEndTime-t)*regulateAR+(td-regulateEndTime)*originalAR,FW))
                    futureallocationNextT = np.where(td<=regulateEndTime,np.maximum((td-t-1)*regulateAR,0),np.minimum(np.maximum(regulateEndTime-t-1,0)*regulateAR+np.where(t+1>regulateEndTime,(td-(t+1))*originalAR,(td-regulateEndTime)*originalAR),FW))

                if t > regulateEndTime:
                    S = x[:,t]*ps[t]-TCs[t]
                    futureallocation = np.minimum((td-t)*originalAR,FW)
                    futureallocationNextT = np.minimum((td-t-1)*originalAR,FW)
                futureallocation = np.where(td!=-self.hoursInA_Day*60,  futureallocation, 0)
                futureallocationNextT = np.where(td!=-self.hoursInA_Day*60,  futureallocationNextT, 0)

                profitAtT = S-np.where(Td> futureallocation,
                                    (Td-futureallocation)*np.where((t <= regulateEndTime) and (t>=regulateStartTime),regulatePrice*(1+regulateCb)+regulateTCb,originalmarketPrice*(1+originalCb)+originalTCb),0.0)
                profitAtT[~mask_cansell] = 0.0

                mask_profit = profitAtT>1e-10
                mask_buy = Td>futureallocation
                mask_buyNextT = Td>futureallocationNextT
                mask_toll0 = Td<1e-10
                mask_FW = np.abs(x[:,t]-FW)<1e-10

                mask_buysell = (mask_cansell&mask_profit)&(mask_buy&~mask_toll0)
                mask_sell1 = (mask_cansell&mask_profit)&(~mask_buy&mask_FW)
                mask_sell2 = (mask_cansell&mask_profit)&(~mask_buy&mask_buyNextT) # or sell when toll is equal to future allocation
                x[(mask_buysell|mask_sell1)|mask_sell2,t] = 0
                if (regulateStartTime<=t) and (t <= regulateEndTime-1):
                        x[:,t+1] = np.minimum(FW,x[:,t]+regulateAR)
                else:
                        x[:,t+1] = np.minimum(FW,x[:,t]+originalAR)

        # aggregate by time interval
        _predictedTT = np.mean(_predictedTT.reshape(-1,self.Tstep),axis=1)
        origtoll = _toll
        _toll = np.mean(_toll.reshape(-1,self.Tstep),axis=1)
        p = np.array([_originalAtt['price']]*self.hoursInA_Day*60)
        p[int(_unusual['regulatesTime']):int(_unusual['regulateeTime'])] = price
        fuelcost = self.dist/self.mpg*self.fuelprice
        convertT = np.mod(_t, self.hoursInA_Day*60)
        for user in self.users:
            if self.predayDeparture[user]< _t:
                todaySchedule[self.predayDeparture[user]].append({'user':user,'tstar':self.desiredArrival[user],'departure':self.predayDeparture[user]})
                continue
            # only if user has preday departure later than now
            s1 = np.maximum(self.DepartureLowerBD[user],convertT)
            s2 = self.DepartureUpperBD[user]
            tstar = self.desiredArrival[user]
            vot = self.vot[user]
            sde = self.sde[user]
            sdl = self.sdl[user]
            vow = self.waiting[user]
            I = self.I[user]
            prev_allowance = self.distribution[user]
            # handle budget constraint (TODO: update burdge constraint handle if _t as a begining point still violates budget constraint)
            if self.ARway == 'continuous':
                R = np.where(x[user,:]>=origtoll, (FW-origtoll)*ps-self.FTCs , (FW-x[user,:])*ps-self.FTCs-((origtoll-x[user,:])*pb+self.FTCb))
                overBudget = np.where(-R+fuelcost>I/2+prev_allowance)[0]
            else:
                overBudget = np.where((origtoll-self.userAccounts[user])*price+fuelcost>I/2+prev_allowance)[0]
            if len(overBudget)>0:
                overBudget = (overBudget/self.Tstep).astype(int)	
                originlen = len(possibleDepartureTimes)
                possibleDepartureTimes = possibleDepartureTimes[~np.in1d(possibleDepartureTimes,(overBudget/self.Tstep))]
                if len(possibleDepartureTimes) < originlen and len(possibleDepartureTimes)>0:
                    self.numOfbindingI += 1
                    self.bindingI.append(user)
                if len(possibleDepartureTimes) == 0 : 
                    firstOne = overBudget[0]
                    lastOne = overBudget[-1]
                    shift = s2-firstOne+1
                    s1 = max(s1-shift,0)
                    s2 = min(firstOne-1,self.hoursInA_Day*60)

                    possibleDepartureTimes = np.arange(s1, s2 + 1, self.Tstep)
                    possibleDepartureTimes = (possibleDepartureTimes/self.Tstep).astype(int) 

                
            utileps = np.zeros(1+len(possibleDepartureTimes))
            utileps[0] = self.user_eps[user,-1] # add random term of no travel
            utileps[1:] = self.user_eps[user,possibleDepartureTimes] 			


            tautilde = np.zeros(1+len(possibleDepartureTimes))
            tautilde[0] = self.pttt
            tautilde[1:] = _predictedTT[possibleDepartureTimes] # add tt of no travel

            Th = np.zeros(1+len(possibleDepartureTimes))
            Th[0] = self.ptfare
            Th[1:] = _toll[possibleDepartureTimes]
            possibleDepartureTimes = possibleDepartureTimes * self.Tstep

            if self.ARway == "continuous":
                possibleAB = x[user,possibleDepartureTimes]
            SDE = np.zeros(1+len(possibleDepartureTimes))
            SDE[1:] = np.maximum(0,tstar-(possibleDepartureTimes+tautilde[1:]+self.Tstep/2)) # use middle point of time interval to calculate SDE
            SDL = np.zeros(1+len(possibleDepartureTimes))
            SDL[1:] = np.maximum(0,(possibleDepartureTimes+tautilde[1:]+self.Tstep/2)-tstar)# use middle point of time interval to calculate SDL
            ASC = np.zeros(len(utileps))
            W = np.zeros(1+len(possibleDepartureTimes)) # waiting time
            W[0] = 1/2*self.ptheadway

            sysutil_t = ASC+(-2*vot *tautilde  - sde * SDE - sdl * SDL-2*vow*W) # for all day, double travel time but not sde and sdl
            ch = np.zeros(len(utileps))


            if self.scenario == 'Trinity':
                if self.ARway == 'lumpsum':
                    buy = np.maximum(Th-self.userAccounts[user], 0)*price
                    sell = np.maximum(self.userAccounts[user] - Th, 0)*price
                    ch[:] = buy-sell
                    ch[0] = Th[0]-np.maximum(self.userAccounts[user], 0)*price
                elif self.ARway == 'continuous':
                    # calculate opportunity cost
                    tempTh = Th[1:] # exclude the first element corresponding to no travel
                    if self.decaying:
                        ch[1:] = -np.where(possibleAB>=tempTh, (FW-tempTh)*ps-(ps*self.AR*((FW-tempTh)/self.AR)**2/(2*self.hoursInA_Day*60))-self.FTCs,
                                    np.maximum(-(tempTh-possibleAB)*pb-self.FTCb+(FW-possibleAB)*ps-(ps*self.AR*((FW-possibleAB)/self.AR)**2/(2*self.hoursInA_Day*60))-self.FTCs,0))
                        ch[0] = Th[0]-np.maximum((FW)*ps-(ps*self.AR*((FW)/self.AR)**2/(2*self.hoursInA_Day*60))-self.FTCs,0)
                    else:
                        #ch = np.where(possibleAB>=Th, Th*ps-self.FTCs,np.maximum((Th-possibleAB)*pb+self.FTCb+possibleAB*ps-self.FTCs,0))
                        # consider opportunity benefit and cost
                        ch[1:] = -np.where(possibleAB>=tempTh, (FW-tempTh)*ps-self.FTCs , (FW-possibleAB)*ps-self.FTCs-((tempTh-possibleAB)*pb+self.FTCb))
                        ch[0] = Th[0]-np.maximum((FW)*ps-self.FTCs,0)
                    
            else:
                buy = Th*price
                sell = np.zeros_like(Th)
                ch = buy-sell-prev_allowance

            # add fuel price
            ch[1:] += fuelcost
            ch = ch*2 # double for all day
            ch_woallowance = (ch/2+prev_allowance)*2
            # allowance
            if not self.allowance['policy']:
                self.distribution[user] = 0
            elif self.allowance['policy'] == 'uniform':
                self.distribution[user] = RR/self.numOfusers
            elif self.allowance['policy'] =='personalization':
                income_c = self.user_params['lambda']*np.log(self.user_params['gamma']+I-ch_woallowance)
                idx = np.argmax(sysutil_t+income_c-ch_woallowance+utileps)
                if self.user_params['lambda']/(self.user_params['gamma']+I-ch_woallowance[idx])+1<=self.allowance['ctrl']:
                    self.distribution[user] = 0
                else:
                    distribution = np.linspace(max(max(ch_woallowance-I),0),max(max(ch_woallowance-I),0)+30,num=300)
                    a_idx = self.user_optimization(distribution, self.allowance['ctrl'],ch_woallowance, I,sysutil_t,user,utileps)
                    self.distribution[user] = min(distribution[a_idx],self.allowance['cap'])

            sysutil =( sysutil_t+self.user_params['lambda']*np.log(self.user_params['gamma']+I-ch)+I-ch)
            util = sysutil+utileps
            if np.argmax(util) == 0: # choose no travel
                self.ptshare.append(user)
                self.predayDeparture[user] = -1
            else:
                departuretime = possibleDepartureTimes[np.argmax(util)-1]+_beginTime
                departuretime = int(np.random.choice(np.arange(departuretime,departuretime+self.Tstep),1)[0])
                self.todaySchedule[departuretime].append({'user':user,'tstar':tstar,'departure':departuretime})
                self.predayDeparture[user] = departuretime
                todaySchedule[departuretime].append({'user':user,'tstar':tstar,'departure':departuretime})
            # if user == 7459:
                # print("user 7559",sysutil,util,possibleDepartureTimes,I,ch,Th,prev_allowance,fuelcost,utileps)
            self.predayEps[user] = utileps[np.argmax(util)]
        self.todaySchedule = todaySchedule
    def get_numOfbindingI(self):
        return self.numOfbindingI

    # realize selling and buying behavior
    def sell_and_buy(self,_t,_currToll,_toll,_price,_totTime):
        FW = self.hoursInA_Day*60*self.AR
        userBuy = np.zeros_like(self.userAccounts)
        userSell = np.zeros_like(userBuy)
        p = _price
        departureTime  = self.predayDeparture.copy()
        mask_cansell = np.where(departureTime!=-1,departureTime!=_t,True)
        departureTime = np.where(departureTime!=-1,np.where(departureTime < _t, departureTime + self.hoursInA_Day * 60, departureTime) , departureTime)
        
        if self.ARway == 'lumpsum':
            userBuy[~mask_cansell] = np.maximum((_currToll-self.userAccounts)[~mask_cansell],0)
            # no need to calculate profit if allocation is lump-sum
            # as selling will be automated at the end of day
            mask_sellnow = False
            if _t == _totTime-1:
                mask_sellnow = (mask_cansell&(self.userAccounts>0))
                userSell[mask_sellnow] = self.userAccounts[mask_sellnow]
            # update user accounts for lump-sum allocation
            # lump-sum allocation 
            mask_donothing = ~(mask_sellnow | ~mask_cansell)
            self.userAccounts[mask_sellnow] = 0
            self.userAccounts[~mask_cansell] = np.maximum((self.userAccounts-_currToll)[~mask_cansell],0)
            #self.userAccounts[mask_donothing] = np.maximum(self.userAccounts[mask_donothing]+AR,FW)
            if _t == _totTime-1:
                self.userAccounts[:] = FW

        elif self.ARway == 'continuous':
            # get buying tokens
            userBuy[~mask_cansell] = np.where(_currToll>self.userAccounts[~mask_cansell], _currToll-self.userAccounts[~mask_cansell], 0.0)

            FA = np.where(departureTime!=-1, np.minimum((departureTime-_t)*self.AR,FW),0)
            B = np.where(departureTime!=-1,(_toll[np.mod(departureTime,self.hoursInA_Day*60)]-FA), departureTime)

            if self.decaying:
                S = self.userAccounts*p*(1-self.PTCs)-(p*(1-self.PTCs)*self.AR*(self.userAccounts/self.AR)**2/(2*self.hoursInA_Day*60))-self.FTCs
            else:
                S = self.userAccounts*p*(1-self.PTCs)-self.FTCs

            profit = S-np.where(B>0, B*p*(1+self.PTCb)+self.FTCb, 0)

            mask_positiveprofit = profit>1e-10
            mask_needbuy = B>=0
            mask_needbuynext = (B+self.AR)>0
            mask_FW = np.abs(self.userAccounts-FW)<1e-10
            mask_sellnow = (mask_cansell&mask_positiveprofit)&(mask_needbuy | mask_FW|mask_needbuynext)
            userSell[mask_sellnow] = self.userAccounts[mask_sellnow]

            #### update accounts
            currTime = np.mod(_t,self.hoursInA_Day*60) # range from 0 to hoursInA_Day*60
            # handle selling
            self.userAccounts[mask_sellnow] = self.AR # sell all and get new allocation

            # handle paying toll and buying
            self.userAccounts[~mask_cansell] = np.maximum((self.userAccounts-_currToll)[~mask_cansell],0)
            self.userAccounts[~mask_cansell] = np.minimum(self.userAccounts[~mask_cansell]+self.AR,FW) # add new allocation and cap it at FW
            # handle do nothing (expire oldest tokens if reach maximum life time and get new allocation)
            mask_donothing = ~(mask_sellnow | ~mask_cansell)
            self.userAccounts[mask_donothing] = np.minimum(self.userAccounts[mask_donothing]+self.AR,FW)
            # print("individual 2 profit",profit[2],S[2],B[2]*p*(2+self.PTCb)+self.FTCb,self.userAccounts[2],FA[2],departureTime[2],_toll[np.mod(departureTime[2],self.hoursInA_Day*60)])

        return [userBuy, userSell]
    
    # sell and buy function only for unusual event day and if allocation is continuous
    def special_sell_and_buy(self,_t,_currToll,_toll,_price,_beginTime,_totTime,_originalAtt,_unusual):

        userBuy = np.zeros_like(self.userAccounts)
        userSell = np.zeros_like(userBuy)
        departureTime  = self.predayDeparture.copy()
        mask_cansell = np.where(departureTime!=-1,departureTime!=_t,True)
        departureTime = np.where(departureTime!=-1,np.where(departureTime < _t, departureTime + self.hoursInA_Day * 60, departureTime) , departureTime)
        userBuy[~mask_cansell] = np.where(_currToll>self.userAccounts[~mask_cansell], _currToll-self.userAccounts[~mask_cansell], 0.0)


        P = _price
        regulatePrice = _unusual['ctrlprice']
        regulateAR = _unusual['AR']
        regulateCs = self.PTCs
        regulateCb = self.PTCb
        regulateTCs = _unusual['FTC']
        regulateTCb = _unusual['FTC']
        regulateStartTime = int(_unusual['regulatesTime'])
        regulateEndTime = int(_unusual['regulateeTime'])

        originalAR = _originalAtt['AR']
        originalmarketPrice = _originalAtt['price']
        originalCs = _originalAtt['PTCs']
        originalCb = _originalAtt['PTCb']
        originalTCs = _originalAtt['FTCs']
        originalTCb = _originalAtt['FTCb']

        FW = originalAR*self.hoursInA_Day*60

        pb = np.array([originalmarketPrice*(1+originalCb)]*self.numOfusers)
        pb[(departureTime>=_beginTime+regulateStartTime)&(departureTime<=_beginTime+regulateEndTime)] = regulatePrice*(1+regulateCb)
        TCb = np.array([originalTCb]*self.numOfusers)
        TCb[(departureTime>=_beginTime+regulateStartTime)&(departureTime<=_beginTime+regulateEndTime)] = regulateTCb
        
        TCs = self.FTCs
        Cs = self.PTCs
        ps = P*(1-Cs)
        profit = np.zeros(self.numOfusers)
        _user_toll = _toll[np.mod(departureTime,self.hoursInA_Day*60)]
        if _t < _beginTime+regulateStartTime:
            mask_tdlessStart = departureTime<regulateStartTime
            mask_tdbwt = (departureTime>=regulateStartTime) & (departureTime <= regulateEndTime)
            mask_tdgreaterEnd = departureTime>regulateEndTime
            futureallocation = np.zeros(self.numOfusers)
            futureallocationNextT = np.zeros(self.numOfusers)
            futureallocation[mask_tdlessStart] = (departureTime[mask_tdlessStart]-_t)*originalAR
            futureallocationNextT[mask_tdlessStart] = futureallocation[mask_tdlessStart]-originalAR
            futureallocation[mask_tdbwt] = (regulateStartTime-_t)*originalAR+(departureTime[mask_tdbwt]-regulateStartTime)*regulateAR
            futureallocationNextT[mask_tdbwt] = futureallocation[mask_tdbwt]-regulateAR
            futureallocation[mask_tdgreaterEnd] = (regulateStartTime-_t)*originalAR+(regulateEndTime-regulateStartTime)*regulateAR + (departureTime[mask_tdgreaterEnd]-regulateEndTime)*originalAR
            futureallocationNextT[mask_tdgreaterEnd] = futureallocation[mask_tdgreaterEnd]-originalAR

            futureallocation = np.minimum(futureallocation,FW)
            futureallocationNextT = np.minimum(futureallocationNextT,FW)

        if (_t >= _beginTime+regulateStartTime) & (_t<=_beginTime+regulateEndTime):
            futureallocation = np.where(departureTime<=_beginTime+regulateEndTime,(departureTime-_t)*regulateAR,np.minimum((_beginTime+regulateEndTime-_t)*regulateAR+(departureTime-(_beginTime+regulateEndTime))*originalAR,FW))
            futureallocationNextT = np.where(departureTime<=_beginTime+regulateEndTime,np.maximum((departureTime-_t-1)*regulateAR,0),np.minimum(np.maximum(_beginTime+regulateEndTime-_t-1,0)*regulateAR+np.where(_t+1>_beginTime+regulateEndTime,(departureTime-(_t+1))*originalAR,(departureTime-(_beginTime+regulateEndTime))*originalAR),FW))
        else:
            futureallocation = np.minimum((departureTime-_t)*originalAR,FW)
            futureallocationNextT = np.minimum((departureTime-_t-1)*originalAR,FW)

        futureallocation = np.where(departureTime!=-1, futureallocation,0)
        futureallocationNextT = np.where(departureTime!=-1, futureallocationNextT,0)
        profit = self.userAccounts*ps-TCs-np.where(_user_toll>futureallocation,(_user_toll-futureallocation)*pb+TCb,0.0)


        profit[~mask_cansell] = 0.0
        mask_profit = profit>1e-10
        mask_buy = _user_toll>futureallocation
        mask_buyNextT = _user_toll> futureallocationNextT
        mask_toll0 = _user_toll<1e-10
        mask_FW = np.abs(self.userAccounts-FW)<1e-10
        mask_buysell = (mask_cansell&mask_profit)&(mask_buy&~mask_toll0)
        mask_sell1 = (mask_cansell&mask_profit)&(~mask_buy&mask_FW) # sell at FW
        mask_sell2 = (mask_cansell&mask_profit)&(~mask_buy&mask_buyNextT) # or sell when toll is equal to future allocation
        mask_sellnow = (mask_buysell|(mask_sell1|mask_sell2))

        userSell[mask_sellnow] = self.userAccounts[mask_sellnow]

        #### update accounts
        # handle selling
        self.userAccounts[mask_sellnow] = self.AR # sell all and get new allocation

        # handle paying toll and buying
        self.userAccounts[~mask_cansell] = np.maximum((self.userAccounts-_currToll)[~mask_cansell],0)
        self.userAccounts[~mask_cansell] = np.minimum(self.userAccounts[~mask_cansell]+self.AR,FW) # add new allocation and cap it at FW
        # handle do nothing (expire oldest tokens if reach maximum life time and get new allocation)
        mask_donothing = ~(mask_sellnow | ~mask_cansell)
        self.userAccounts[mask_donothing] = np.minimum(self.userAccounts[mask_donothing]+self.AR,FW)
        # print("individual 2 profit",profit[2],S[2],B[2]*p*(2+self.PTCb)+self.FTCb,self.userAccounts[2],FA[2],departureTime[2],_toll[np.mod(departureTime[2],self.hoursInA_Day*60)])
        return [userBuy,userSell]

    # compute future user accounts:
    def update_account(self):
        return

    def update_arrival(self,actualArrival):
        self.actualArrival = actualArrival

        
    # perform day to day learning
    def d2d(self,actualTT):
        self.actualTT = actualTT
        self.predictedTT = 0.9*self.predictedTT+0.1*self.actualTT 

class Regulator():
    # regulator account balance
    def __init__(self,marketPrice=1,RBTD = 100, deltaP = 0.05):
        self.RR = 0
        self.tollCollected = 0
        self.allowanceDistributed = 0
        self.marketPrice = marketPrice
        self.RBTD = 100
        self.deltaP = 0.05
    # update regulator account
    def update_balance(self,userToll,userReceive):
        # userToll: regulator revenue
        # userReceive: regulator cost
        self.tollCollected = np.sum(userToll)
        self.allowanceDistributed = np.sum(userReceive)
        self.RR = self.tollCollected-self.allowanceDistributed

    def update_price(self):
        if self.RR > self.RBTD:
            self.marketPrice += self.deltaP
        elif self.RR < -self.RBTD:
            self.marketPrice -= self.deltaP


class Simulation():
    # simulate one day

    def __init__(self, _user_params,_allocation ,_scenario='NT',_allowance=False,_numOfdays=50,_numOfusers=7500,_Tstep=1,_hoursInA_Day=12,_fftt=24,_capacity = 42,
                _marketPrice = 1,_RBTD = 100, _deltaP=0.05,_Plot = False,_seed=5843,_verbose = False,_unusual=False,_storeTT=False,_CV=True,save_dfname='CPresult.csv' , toll_type ="normal"):
        self.numOfdays = _numOfdays
        self.hoursInA_Day = _hoursInA_Day
        self.numOfusers = _numOfusers
        self.allowance = _allowance
        self.save_dfname = save_dfname
        self.currday = 0
        self.fftt = _fftt
        self.user_params = _user_params
        self.Tstep = _Tstep
        self.capacity = _capacity
        self.scenario = _scenario
        self.FTCs = _allocation['FTCs']
        self.FTCb = _allocation['FTCb']
        self.PTCs = _allocation['PTCs']
        self.PTCb = _allocation['PTCb']
        self.Plot = _Plot
        self.verbose = _verbose
        self.unusual = _unusual
        self.storeTT = _storeTT
        self.CV = _CV
        self.AR = _allocation['AR']
        self.toll_type = toll_type
        self.decaying = _allocation['Decaying']
        self.usertradedf = pd.DataFrame({'buy': np.zeros(self.hoursInA_Day*60),'sell': np.zeros(self.hoursInA_Day*60)}) # record user amount of trade behaviors
        self.tokentradedf = pd.DataFrame({'buy': np.zeros(self.hoursInA_Day*60),'sell': np.zeros(self.hoursInA_Day*60)}) # record average token amount of trade behaviors

        self.flowdf = pd.DataFrame({'departure':np.zeros(self.numOfdays*self.numOfusers),'arrival':np.zeros(self.numOfdays*self.numOfusers),
            'user':np.tile(np.arange(self.numOfusers),self.numOfdays)})

        self.users = Travelers(self.numOfusers,_user_params=self.user_params,_allocation=_allocation,_fftt=_fftt,
                            _hoursInA_Day=_hoursInA_Day,_Tstep=self.Tstep,_allowance=self.allowance,
                            _scenario = self.scenario,_seed=_seed,_unusual=self.unusual,_CV = _CV,_numOfdays = _numOfdays)
        self.users.generate_params()

        self.regulator = Regulator(_marketPrice,_RBTD,_deltaP)
        self.pricevec = []
        self.swvec = []
        self.flowconvergevec = []
        self.ptsharevec = []
        self.originalAtt = {}

        self.presellAfterdep = np.zeros(self.numOfusers,dtype=int)



    def steptoll_fxn(self,x,step1=3,step2=6,step3=10,step4=6,step5=3,x_step1 = 250,x_step2 = 310,x_step3=360,x_step4=460,x_step5=510,x_step6=570):
        steptoll = (x>=x_step1)*(x<x_step2)*step1 + (x>=x_step2)*(x<x_step3)*step2 + (x>=x_step3)*(x<x_step4)*step3 + \
            (x>=x_step4)*(x<x_step5)*step4 + (x>=x_step5)*(x<x_step6)*step5 + (x>=x_step6)*(x<x_step1)*0
        return steptoll

    def bimodal(self,x,mu1,sigma1,A1):
        def custgauss(x,mu,sigma,A):
            return A*np.exp(-(x-mu)**2/2/sigma**2)
        return custgauss(x,mu1*60,sigma1,A1)

    def simulate(self, tollparams):
        # create time of day toll 
        self.tollparams=tollparams
        timeofday = np.arange(self.hoursInA_Day*60)
        if self.scenario == 'NT':
            self.toll = np.array([0]*len(timeofday))
        elif self.scenario == 'CP' or self.scenario == 'Trinity':
            if self.toll_type == 'step':
                self.toll = np.maximum(self.steptoll_fxn(timeofday, *tollparams),0)#np.repeat(np.maximum(bimodal(O['departuretime'].values[np.arange(0,hoursInA_Day*60,Tstep)], *params),0),Tstep)
                self.toll = np.around(self.toll,2)
            elif self.toll_type == 'normal':
                self.toll = np.repeat(np.maximum(self.bimodal(timeofday[np.arange(0,self.hoursInA_Day*60,self.Tstep)], *tollparams),0),self.Tstep)
                self.toll = np.around(self.toll,2)

        for day in range(self.numOfdays):
            self.simulateOneday(day=day)

    def simulateOneday(self,day=0):
        self.currday = day
        beginTime = day*self.hoursInA_Day*60
        totTime =  (day+1)*self.hoursInA_Day*60
        self.users.update_choice(self.users.predictedTT,beginTime,totTime,self.toll,self.regulator.tollCollected,self.regulator.marketPrice,self.currday)
        self.numOfundesiredTrading = np.zeros(self.numOfusers)
        sellTime = np.zeros(self.numOfusers)

        departureQueue = []
        actualArrival = np.zeros(self.numOfusers)
        userSell = np.zeros(self.numOfusers)
        userBuy = np.zeros(self.numOfusers)
        userBuytc = np.zeros(self.numOfusers)
        userSelltc = np.zeros(self.numOfusers)
        userToll = np.zeros(self.numOfusers)
        actualTT = np.zeros(self.hoursInA_Day*60)
        numDepart = np.zeros(self.hoursInA_Day*60)
        buyvec = np.zeros(totTime-beginTime) # buy user amount
        sellvec = np.zeros(totTime-beginTime) # sell user amount
        buyamount = np.zeros(totTime-beginTime) # average buy token amount
        sellamount = np.zeros(totTime-beginTime) # average sell token amount

        if self.unusual['unusual'] and self.currday == self.unusual['day']:
            self.originalAtt['price'] = self.regulator.marketPrice
            self.originalAtt['FTCb'] = self.users.FTCb
            self.originalAtt['FTCs'] = self.users.FTCs
            self.originalAtt['PTCb'] = self.users.PTCb
            self.originalAtt['PTCs'] = self.users.PTCs
            self.originalAtt['AR'] = self.users.AR
            
        for t in range(beginTime,totTime):
            ####### for unusual event
            if self.unusual['unusual'] and self.currday == self.unusual['day']:
                if (t>=self.unusual['dropstime']+beginTime) & (t<=self.unusual['dropetime']+beginTime):
                    outputcounter = self.unusual['capacity']
                else:
                    outputcounter = self.capacity
                if (t>=self.unusual['regulatesTime']+beginTime) & (t<=self.unusual['regulateeTime']+beginTime):
                    self.regulator.marketPrice = self.unusual['ctrlprice']
                    if self.users.ARway == 'continuous':
                        self.users.update_TC(self.unusual['FTC'],self.unusual['FTC'],self.users.PTCs,self.users.PTCb)
                        self.users.AR = self.unusual['AR']
                else:
                    self.regulator.marketPrice = self.originalAtt['price']
                    if self.users.ARway == 'continuous':
                        self.users.update_TC(self.originalAtt['FTCs'],self.originalAtt['FTCb'],self.users.PTCs,self.users.PTCb)
                        self.users.AR = self.originalAtt['AR']
                if t == self.unusual['regulatesTime']+beginTime:
                    self.users.withinday_choice(self.users.predictedTT,beginTime,totTime,self.toll,self.regulator.tollCollected,self.regulator.marketPrice,t,self.originalAtt,self.unusual)
            else:
                outputcounter = self.capacity 
            #############
            departureNow = self.users.todaySchedule[t]
            currToll = self.toll[np.mod(t-t%self.Tstep,self.hoursInA_Day*60)]
            for car in departureNow:
                userToll[car['user']] = currToll*self.regulator.marketPrice
                if outputcounter != 0:
                    if len(departureQueue) == 0:
                        actualArrival[car['user']] = t+self.fftt
                        actualTT[np.mod(car['departure'],self.hoursInA_Day*60)] += t+self.fftt-car['departure']
                        numDepart[np.mod(car['departure'],self.hoursInA_Day*60)] += 1
                    else:
                        actualArrival[departureQueue[0]['user']] =  t+self.fftt
                        actualTT[np.mod(departureQueue[0]['departure'],self.hoursInA_Day*60)] += t+self.fftt-departureQueue[0]['departure']
                        numDepart[np.mod(departureQueue[0]['departure'],self.hoursInA_Day*60)] += 1
                        departureQueue = departureQueue[1:]
                        departureQueue.append(car)
                    outputcounter = outputcounter - 1
                else:
                    departureQueue.append(car)
            # if number of people departing now less than output capacity, we continue to dissipate people in queue
            while outputcounter != 0:
                if len(departureQueue) != 0:
                    actualArrival[departureQueue[0]['user']] = t+self.fftt
                    actualTT[np.mod(departureQueue[0]['departure'],self.hoursInA_Day*60)] += t+self.fftt-departureQueue[0]['departure']
                    numDepart[np.mod(departureQueue[0]['departure'],self.hoursInA_Day*60)] += 1
                    departureQueue = departureQueue[1:]
                outputcounter = outputcounter - 1
            if self.scenario == 'Trinity':
                if (self.unusual['unusual'] & (self.currday == self.unusual['day'])) & (self.users.ARway=='continuous'):
                    tempuserBuy, tempuserSell = self.users.special_sell_and_buy(t,currToll,self.toll,self.regulator.marketPrice,beginTime,totTime,self.originalAtt,self.unusual)
                else:
                    tempuserBuy, tempuserSell = self.users.sell_and_buy(t,currToll,self.toll,self.regulator.marketPrice,totTime)
                buy_user_amount= int(np.count_nonzero(tempuserBuy))
                sell_user_amount= int(np.count_nonzero(tempuserSell))
                buyvec[t-beginTime] = buy_user_amount
                sellvec[t-beginTime] = sell_user_amount
                sellamount[t-beginTime] = np.sum(tempuserSell*self.regulator.marketPrice)/sell_user_amount
                buyamount[t-beginTime] = np.sum(tempuserBuy*self.regulator.marketPrice)/buy_user_amount

                self.numOfundesiredTrading = np.where(((userSell >1e-6)|(self.presellAfterdep)) & (tempuserBuy>1e-6),1,self.numOfundesiredTrading)
                sellTime[np.where(tempuserSell>1e-6)[0]] = t
                # userBuy += np.where(tempuserBuy>1e-6 ,tempuserBuy*self.regulator.marketPrice*(1+self.PTCb)+self.FTCb,0)
                userBuy += np.where(tempuserBuy>1e-6, tempuserBuy*self.regulator.marketPrice*1, 0)
                userBuytc += np.where(tempuserBuy>1e-6, tempuserBuy*self.regulator.marketPrice*self.PTCb+self.FTCb, 0)
                if self.decaying:
                    userSell += np.where(tempuserSell>1e-6,tempuserSell*self.regulator.marketPrice*(1)-
                                                            (self.regulator.marketPrice*(1)*self.AR*(tempuserSell/self.AR)**2/(2*self.hoursInA_Day*60)),0)
                    userSelltc += np.where(tempuserSell>1e-6,tempuserSell*self.regulator.marketPrice*(-self.PTCs)-
                                                            (self.regulator.marketPrice*(-self.PTCs)*self.AR*(tempuserSell/self.AR)**2/(2*self.hoursInA_Day*60))-self.FTCs,0)
                else:
                    userSell += np.where(tempuserSell>1e-6, tempuserSell*self.regulator.marketPrice*(1), 0)
                    userSelltc += np.where(tempuserSell>1e-6, tempuserSell*self.regulator.marketPrice*(-self.PTCs)-self.FTCs, 0)
            # regulator updates balance

        self.usertradedf['buy'] = buyvec
        self.usertradedf['sell'] = sellvec
        self.tokentradedf['sell'] = sellamount
        self.tokentradedf['buy'] = buyamount

        actualTT = np.divide(actualTT, numDepart, out=np.zeros_like(actualTT)+self.fftt, where=numDepart!=0)
        self.users.update_arrival(actualArrival)
        self.flowdf.iloc[self.currday*self.numOfusers:(self.currday+1)*self.numOfusers,0] = np.maximum(self.users.predayDeparture-beginTime,-1)
        self.flowdf.iloc[self.currday*self.numOfusers:(self.currday+1)*self.numOfusers,1] = np.maximum(actualArrival-beginTime,-1)
        if self.currday>=1:
            mask1 = self.flowdf.iloc[self.currday*self.numOfusers:(self.currday+1)*self.numOfusers,0].values>0
            mask2 = self.flowdf.iloc[(self.currday-1)*self.numOfusers:(self.currday)*self.numOfusers,0].values>0
            self.flowconvergevec.append(np.linalg.norm(self.flowdf.iloc[self.currday*self.numOfusers:(self.currday+1)*self.numOfusers,0].values[mask1&mask2]-self.flowdf.iloc[(self.currday-1)*self.numOfusers:(self.currday)*self.numOfusers,0].values[mask1&mask2]))

        # day to day learning
        # update regulator account balance at the end of day
        self.userSell = userSell
        self.userSelltc = userSelltc
        self.userBuytc = userBuytc
        self.userToll = userToll
        if self.scenario == 'Trinity':
            self.userBuy = userBuy
            self.regulator.update_balance(self.userBuy+self.userBuytc,self.userSell+self.userSelltc)
        else:
            self.userBuy = userToll
            self.regulator.update_balance(userToll,self.users.distribution)
        self.pricevec.append(self.regulator.marketPrice)
        self.ptsharevec.append(len(self.users.ptshare))
        self.swvec.append(self.calculate_sw())

        if self.unusual['unusual'] and self.currday == self.unusual['day']:
            self.users.d2d(self.users.predictedTT)
            self.regulator.marketPrice = self.originalAtt['price']
        else:
            self.users.d2d(actualTT)
            if self.scenario == 'Trinity':
                self.regulator.update_price()	

        self.presellAfterdep = sellTime>self.users.predayDeparture
        # print("number of users buying: ",np.count_nonzero(userBuy)," number of users selling: ", np.count_nonzero(userSell),'RR',self.regulator.RR,'# undesired',np.sum(self.numOfundesiredTrading),
            # 'price', self.regulator.marketPrice,'buying amount',np.sum(userBuy*self.regulator.marketPrice),'selling amount',np.sum(userSell*self.regulator.marketPrice))
        # print(self.numOfundesiredTrading)
    def calculate_sw(self):
        
        TT = np.where(self.users.predayDeparture!=-1,self.users.actualArrival-self.users.predayDeparture,self.users.pttt) 
        SDE = np.where(self.users.predayDeparture!=-1,np.maximum(0,self.users.desiredArrival+self.currday*self.hoursInA_Day*60-self.users.actualArrival),0)
        SDL = np.where(self.users.predayDeparture!=-1,np.maximum(0,self.users.actualArrival-(self.users.desiredArrival+self.currday*self.hoursInA_Day*60)),0)
        allowance = self.users.distribution
        # either car fuel cost or transit fare
        fuelcost = np.where(self.users.predayDeparture!=-1,self.users.dist/self.users.mpg*self.users.fuelprice ,self.users.ptfare)
        ASC = np.zeros(self.numOfusers)
        ptwaitingtime = np.where(self.users.predayDeparture!=-1,0 ,self.users.ptheadway)
        util = ASC+(-2*self.users.vot*TT-self.users.sde*SDE-self.users.sdl*SDL-self.users.waiting*ptwaitingtime
            +self.user_params['lambda']*np.log(self.user_params['gamma']+self.users.I-2*self.userBuy+2*self.userSell+2*allowance-2*fuelcost)
            +self.users.I-2*self.userBuy+2*self.userSell+2*allowance-2*fuelcost)+self.users.predayEps
        NTMUI = 1+self.user_params['lambda']/(self.user_params['gamma']+self.users.I)
        if self.scenario == 'NT':
            obj = np.sum(util)
        else:
            NT_util = np.load("./output/Bottleneck/NT/NT_util.npy")
            # userBenefits = (util-NT_util)/NTMUI
            userBenefits = (util-NT_util)
            obj = np.sum(userBenefits)+ 2*self.regulator.RR
        return obj

    def metrics(self):
        TT = np.where(self.users.predayDeparture!=-1,self.users.actualArrival-self.users.predayDeparture,self.users.pttt)
        SDE = np.where(self.users.predayDeparture!=-1,np.maximum(0,self.users.desiredArrival+self.currday*self.hoursInA_Day*60-self.users.actualArrival),0)
        SDL = np.where(self.users.predayDeparture!=-1,np.maximum(0,self.users.actualArrival-(self.users.desiredArrival+self.currday*self.hoursInA_Day*60)),0)
        allowance = self.users.distribution
        ASC = np.zeros(self.numOfusers)
        fuelcost = np.where(self.users.predayDeparture!=-1,self.users.dist/self.users.mpg*self.users.fuelprice ,self.users.ptfare)
        ptwaitingtime = np.where(self.users.predayDeparture!=-1,0 ,self.users.ptheadway)
        TTv = self.users.vot*TT
        SDEv = self.users.sde*SDE
        SDLv = self.users.sdl*SDL
        Wv = self.users.waiting*ptwaitingtime
        nonlinearInc =  self.user_params['lambda']*np.log(self.user_params['gamma']+self.users.I-2*self.userBuy-2*fuelcost+2*self.userSell+2*allowance)
        linearInc = self.users.I-2*self.userBuy-2*fuelcost+2*self.userSell+2*allowance
        sysutil = ASC+(-2*TTv-SDEv-SDLv-Wv
            +nonlinearInc
            +linearInc)
        util = sysutil+self.users.predayEps
        NTMUI = 1+self.user_params['lambda']/(self.user_params['gamma']+self.users.I)
        print("numOfbindingI: ",self.users.get_numOfbindingI(),	"number of pt travel: ",len(self.users.ptshare))
        # print("pt users: ", self.users.ptshare)
        # print("avg income of no travel", np.average(self.users.I[self.users.notravel]),np.quantile(self.users.I[self.users.notravel],0.8),np.quantile(self.users.I[self.users.notravel],0.2))
        ifptshare = np.full(self.numOfusers, False)
        ifbindingI = np.full(self.numOfusers, False)
        ifptshare[self.users.ptshare] = True	
        ifbindingI[self.users.bindingI] = True
        print("average tt: ",np.average(TT[TT>=self.fftt]),"average sde: ", np.average(SDE[TT>=self.fftt]),"average SDL: ", np.average(SDL[TT>=self.fftt]),"max TT: ",max(TT))
        print("average car tt: ",np.average(TT[self.users.predayDeparture!=-1]),"average car sde: ", np.average(SDE[self.users.predayDeparture!=-1]),
            "average SDL: ", np.average(SDL[self.users.predayDeparture!=-1]),"max TT: ",max(TT))
        self.flowdf['tt'] = np.where(self.flowdf.departure!=-1, self.flowdf.arrival-self.flowdf.departure, self.users.pttt)
        if self.scenario =="NT":
            main_log_dir = "./output/Bottleneck/NT/"
        if 	self.scenario =="Trinity":
            if self.toll_type == "step":
                main_log_dir = "./output/Bottleneck/Trinity_step/"
            if self.toll_type == "normal":
                main_log_dir = "./output/Bottleneck/Trinity_normal/"
        np.save((main_log_dir+"swvec.npy"), np.array(self.swvec))
        np.save((main_log_dir+"pricevec.npy"), np.array(self.pricevec))
        self.flowdf.to_csv(main_log_dir+"flowdf.csv")
        self.tokentradedf.to_csv(main_log_dir+"tokentradedf.csv")
        self.usertradedf.to_csv(main_log_dir+"usertradedf.csv")

        #####
        def save_obj(obj, name):
            with open(name + '.pkl', 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        def load_obj(name ):
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)

        #### plot ###
        if self.scenario == 'NT':
            np.save("./output/Bottleneck/NT/NT_util",util)
            np.save("./output/Bottleneck/NT/NT_utiltt",TT)
            np.save("./output/Bottleneck/NT/NT_utilsde",SDE)
            np.save("./output/Bottleneck/NT/NT_utilsdl",SDL)
            print("nan user id:", np.where(np.isnan(util)))
            df = pd.DataFrame({'betavot':self.users.vot,'dailyincome':self.users.I, 'ifptshare':ifptshare,'TTv':TTv,'SDEv':SDEv,'SDLv':SDLv,'Wv':Wv,'nonlinearInc':nonlinearInc,'linearInc':linearInc,
                                'arrival':np.mod(self.users.actualArrival,self.hoursInA_Day*60),'departure':np.where(self.users.predayDeparture!=-1,np.mod(self.users.predayDeparture,self.hoursInA_Day*60),self.users.predayDeparture)})
            qt1 = df[df['dailyincome']<=df.dailyincome.quantile(0.25)]
            qt2 = df[(df['dailyincome']>df.dailyincome.quantile(0.25))&(df['dailyincome']<=df.dailyincome.quantile(0.5))]
            qt3 = df[(df['dailyincome']>df.dailyincome.quantile(0.5))&(df['dailyincome']<=df.dailyincome.quantile(0.75))]
            qt4 = df[(df['dailyincome']>df.dailyincome.quantile(0.75))&(df['dailyincome']<=df.dailyincome.quantile(0.9))]
            qt5 = df[(df['dailyincome']>df.dailyincome.quantile(0.9))]
            data_to_plot = [qt1,qt2,qt3,qt4,qt5]
            folder = "Plot/Bottleneck/NT/"
            if self.Plot:
                print(df.dailyincome.quantile(0.25),df.dailyincome.quantile(0.5))
                print("ifptshare by VOT",[np.sum(i['ifptshare']) for i in data_to_plot] )
                print([len(i) for i in data_to_plot])
                self.plot_flow(self.flowdf,figname='NTflow',toll=False,folder=folder,day=20)
                self.plot_toll(folder=folder)
                # self.plot_flow(self.flowdf,figname='NTflow2',toll=False,folder=folder,day=1)
                self.plot_tt(self.flowdf,figname='NTtt',toll=False,flow=True,histbins =5,folder=folder,day=20)
                self.plot_cumflow(self.flowdf,figname='NTcumflow',toll=False,folder=folder,day=20)
                self.plot_swconvergence(np.array(self.swvec)/self.numOfusers, folder = folder)
                self.plot_priceconvergence(np.array(self.pricevec),folder = folder)
                self.plot_ptshare_number(np.array(self.ptsharevec), folder = folder)
                self.plot_ttconvergence(self.flowdf, folder = folder)

            if self.verbose:
                dicttosave = {"Tollcollected": len(self.users.ptshare)*2*2/self.numOfusers,"ntgini": gini(self.users.I),
                        "avgtt":np.average(TT),"avgsde":np.average(SDE),"avgsdl":np.average(SDL),"RR": 2*self.regulator.RR,
                        'numUndesired':np.sum(self.numOfundesiredTrading),'dataframe':df,
                        "avgcartt":np.average(TT[self.users.predayDeparture!=-1]),"avgcarsde": np.average(SDE[self.users.predayDeparture!=-1]),"avgcarsdl": np.average(SDL[self.users.predayDeparture!=-1]),
                        "flowdf":self.flowdf,"numOfbindingI":self.users.get_numOfbindingI(),"numberOfPT":len(self.users.ptshare)}
                save_obj(dicttosave, self.save_dfname)

                return 	{"ntgini": gini(self.users.I),"avgtt":np.average(TT[TT>=10]),"avgsde":np.average(SDE[TT>=10]),"avgsdl":np.average(SDL[TT>=10]),"sw":self.swvec}
        else:
            if self.scenario == "Trinity":
                if  self.toll_type == "normal":
                    folder = "./Plot/Bottleneck/Trinity_normal/"
                if self.toll_type == "step":
                    folder = "./Plot/Bottleneck/Trinity_step/"
            else: 
                folder = "./Plot/Bottleneck/CP/"
            NT_util = np.load("./output/Bottleneck/NT/NT_util.npy")
            NT_utiltt = np.load("./output/Bottleneck/NT/NT_utiltt.npy")
            NT_utilsdl = np.load("./output/Bottleneck/NT/NT_utilsdl.npy")
            NT_utilsde = np.load("./output/Bottleneck/NT/NT_utilsde.npy")
            userBenefits = (util-NT_util)
            if self.scenario == 'CP':
                if not self.allowance['policy']:
                    np.save("./output/Bottleneck/CP/CPuserw", userBenefits)
                else:
                    CPuserw = np.load("./output/Bottleneck/CP/CPuserw.npy")
            
            print("nan user id:", np.where(np.isnan(userBenefits)))
            obj = np.sum(userBenefits)+ 2*self.regulator.RR
            if math.isnan(obj):
                obj = -math.inf
            if self.scenario == 'Trinity' and np.abs(self.regulator.RR)>500:
                obj = -math.inf
            if self.CV:
                objcv = np.sum(self.users.userCV) + 2*self.regulator.RR
                print("objcv",objcv)
            else:
                objcv = 0
            print("welfare: ",obj,"MCE: ", np.sum((util-NT_util)/NTMUI)+ 2*self.regulator.RR,'full allowance: ', 2*np.sum(allowance),'full RR: ',2*self.regulator.RR,'full Toll revenue: ', 2*np.sum(self.userBuy))
            print("number of buying: ", np.count_nonzero(self.userBuy),"number of selling: ", np.count_nonzero(self.userSell),"number of undesired: ", np.sum(self.numOfundesiredTrading))
            print("original gini: ", gini(self.users.I), "current gini: ", gini(self.users.I+userBenefits))
            print("NT util gini: ", gini(NT_util), "current util gini: ", gini(util))	
            if self.storeTT['flag']:
                np.save(self.storeTT['ttfilename'],self.users.predictedTT)

            #print("price: ",self.pricevec)
            df = pd.DataFrame({'betavot':self.users.vot,'dailyincome':self.users.I,'a':self.users.distribution,'mce':userBenefits/NTMUI,'utilbenefit':userBenefits,'util':util,'tt': TT,'cv':self.users.userCV,
                                'arrival':np.mod(self.users.actualArrival,self.hoursInA_Day*60),'departure':np.where(self.users.predayDeparture!=-1,np.mod(self.users.predayDeparture,self.hoursInA_Day*60),self.users.predayDeparture),
                                'ifptshare':ifptshare,'ifbindingI':ifbindingI,"SDE":SDE,"SDL":SDL,"TT":TT,"CPcost":self.userBuy,"NTtt":NT_utiltt,"NTsde":NT_utilsde,"NTsdl":NT_utilsdl,'PAT':self.users.desiredDeparture,
                                'usersell':self.userSell,'userbuy':self.userBuy,'usernet':self.userSell-self.userBuy,'userToll':self.userToll,'userAllowance':self.users.distribution,'uninet':self.users.distribution-self.userBuy,
                                'TTv':TTv,'SDEv':SDEv,'SDLv':SDLv,'Wv':Wv,'nonlinearInc':nonlinearInc,'linearInc':linearInc
                                })
            df.to_csv(self.save_dfname+'.csv')
            qt1 = df[df['dailyincome']<=df.dailyincome.quantile(0.25)]
            qt2 = df[(df['dailyincome']>df.dailyincome.quantile(0.25))&(df['dailyincome']<=df.dailyincome.quantile(0.5))]
            qt3 = df[(df['dailyincome']>df.dailyincome.quantile(0.5))&(df['dailyincome']<=df.dailyincome.quantile(0.75))]
            qt4 = df[(df['dailyincome']>df.dailyincome.quantile(0.75))&(df['dailyincome']<=df.dailyincome.quantile(0.9))]
            qt5 = df[(df['dailyincome']>df.dailyincome.quantile(0.9))]
            data_to_plot = [qt1,qt2,qt3,qt4,qt5]
            qt1select = qt1[qt1['ifbindingI']==False]
            print("average utilbenefit for qt1 no binding: ",np.average(qt1select['utilbenefit']),np.average(qt1['utilbenefit']),len(qt1),len(qt1select))
            qt1select = qt1[qt1['ifptshare']==False]
            print("average utilbenefit for qt1 traveling: ",np.average(qt1select['utilbenefit']),np.average(qt1['utilbenefit']),len(qt1),len(qt1select))
            print([len(i) for i in data_to_plot])
            print("average mce by I",[np.average(i['mce']) for i in data_to_plot])
            print("average utilbenefit by I",[np.average(i['utilbenefit']) for i in data_to_plot])
            print("average utilbenefit increase percentage by I",[np.average(i['utilbenefit']/i['dailyincome']) for i in data_to_plot])
            print("average SDE by I",[np.average(i['SDE']) for i in data_to_plot])
            print("average SDL by I",[np.average(i['SDL']) for i in data_to_plot])
            print("average TT by I",[np.average(i['TT']) for i in data_to_plot])
            print("average cpcost by I",[np.average(i['CPcost']) for i in data_to_plot])
            print("average buy by I",[np.average(i['userbuy']) for i in data_to_plot])
            print("average sell by I",[np.average(i['usersell']) for i in data_to_plot])
            print("average NTtt by I",[np.average(i['NTtt']) for i in data_to_plot])
            print("average NTsdl by I",[np.average(i['NTsdl']) for i in data_to_plot])
            print("average NTsde by I",[np.average(i['NTsde']) for i in data_to_plot])
            print("average cv by I",[np.average(i['cv']) for i in data_to_plot])
            print("total income share by I",[np.sum(i['dailyincome'])/np.sum(df['dailyincome']) for i in data_to_plot])
            print("total income+benefit share by I",[np.sum(i['utilbenefit']+i['dailyincome'])/np.sum(df['dailyincome']+df['utilbenefit']) for i in data_to_plot])
            print("allowance by I",[np.average(i['a']) for i in data_to_plot])
            print("ifptshare by I",[np.sum(i['ifptshare']) for i in data_to_plot] )
            print("ifbindingI by I",[np.sum(i['ifbindingI']) for i in data_to_plot] )
            print("MUI by I",[np.average(1+self.user_params['lambda']/(self.user_params['gamma']+i['dailyincome'])) for i in data_to_plot])

            print("final market price is: ", self.regulator.marketPrice)
            ##### plot ####
            if self.Plot:
                # self.plot_priceconvergence(self.pricevec,folder = folder)
                self.plot_toll(folder=folder)
                if self.scenario == 'CP' :
                    self.plot_swconvergence(np.array(self.swvec)/self.numOfusers, folder = folder)
                    self.plot_priceconvergence(np.array(self.pricevec), folder = folder)
                    self.plot_ptshare_number(np.array(self.ptsharevec), folder = folder)
                    self.plot_ttconvergence(self.flowdf, folder = folder)
                    self.plot_token_trade_amount(folder = folder)
                    self.plot_user_trade_amount(folder = folder)

                    if self.allowance['policy'] == 'uniform':
                        self.plot_heatmap(df,figname='uniformutilheat',histbins=5,folder= folder)
                        self.plot_indwelfare(df['dailyincome'].values,df['mce'].values-CPuserw,xlim=(50,200),ylim=(-10,40),figname="uniformWimprov.png",folder= folder)
                        self.plot_indallowance(df['dailyincome'].values,df['a'].values,xlim=(50,200),ylim=(0,10),figname = "uniformallowance.png",folder = folder)
                        self.plot_flow(self.flowdf,figname='uniformflow',toll=True,folder= folder)
                        self.plot_cumflow(self.flowdf,figname='uniformcumflow',toll=True,folder= folder,day=10)
                        self.plot_todutil(df,figname='uniformtodutil',toll=True,folder= folder)
                        self.plot_box(df,figname="cpuni_ttbox",y='TT',ylabel='min',folder= folder)
                        self.plot_box(df,figname="cpuni_sdebox",y='SDE',ylabel='min',folder= folder)
                        self.plot_box(df,figname="cpuni_sdlbox",y='SDL',ylabel='min',folder= folder)
                        self.plot_box(df,figname="cpuni_allowbox",y='usersell',ylabel='$',folder= folder)
                        self.plot_box(df,figname="cpuni_buybox",y='userbuy',ylabel='$',folder= folder)
                        self.plot_box(df,figname="cpuni_netbox",y='uninet',ylabel='$',folder= folder)
                        self.plot_box(df,figname="cpuni_userWbox",y='utilbenefit',ylabel='$',folder= folder)
                        self.plot_boxptshare(df,figname="cpuni_PTuserWbox",y='utilbenefit',ylabel='$',folder= folder)
                    elif  self.allowance['policy'] == 'personalization':
                        if self.allowance['cap'] == float("inf"):
                            # self.plot_indwelfare(df['dailyincome'].values, df['mce'].values-CPuserw,xlim=(50,200),ylim=(-10,40),figname="personalizedWimprov.png",folder= folder)
                            # self.plot_indallowance(df['dailyincome'].values,df['a'].values,xlim=(50,200),ylim=(0,10),figname = "personalizedallowance.png",folder = folder)
                            self.plot_flow(self.flowdf,figname='personalizedflow',toll=True,folder= folder,day=10)
                            self.plot_cumflow(self.flowdf,figname='personalizedcumflow',toll=True,folder= folder,day=10)
                            self.plot_boxptshare(df,figname="cpper_PTuserWbox",y='utilbenefit',ylabel='$',folder= folder)
                            self.plot_todutil(df,figname='personalizedtodutil',toll=True,folder= folder)
                        else:
                            # self.plot_indwelfare(df['dailyincome'].values, df['mce'].values-CPuserw,xlim=(50,200),ylim=(-10,40),figname="personalizedCapWimprov.png",folder= folder)
                            # self.plot_indallowance(df['dailyincome'].values,df['a'].values,xlim=(50,200),ylim=(0,10),figname = "personalizedCapallowance.png",folder = folder)
                            self.plot_flow(self.flowdf,figname='personalizedCapflow',toll=True,folder= folder,day=10)
                            self.plot_cumflow(self.flowdf,figname='personalizedCapcumflow',toll=True,folder= folder,day=10)
                            self.plot_boxptshare(df,figname="cpperCap_PTuserWbox",y='utilbenefit',ylabel='$',folder= folder)
                            self.plot_todutil(df,figname='personalizedCaptodutil',toll=True,folder= folder)
                    else:
                        self.plot_heatmap(df,figname='CPutilheat',histbins=5,folder= folder)
                        self.plot_flow(self.flowdf,figname='CPflow',toll=True,folder= folder,day=10)
                        self.plot_todutil(df,figname='CPtodutil',toll=False,folder= folder)
                        self.plot_tt(self.flowdf,figname='CPtt2',toll=False,flow=True,histbins =5,folder= folder,day=10)
                        self.plot_cumflow(self.flowdf,figname='CPcumflow',toll=True,folder= folder,day=10)
                        self.plot_speed(self.flowdf,figname='CPspeed',histbins=5,folder= folder,day=10)
                        self.plot_box(df,figname="cp_ttbox",y='TT',ylabel='min',folder= folder)
                        self.plot_box(df,figname="cp_sdebox",y='SDE',ylabel='min',folder= folder)
                        self.plot_box(df,figname="cp_sdlbox",y='SDL',ylabel='min',folder= folder)
                        self.plot_box(df,figname="cp_allowbox",y='usersell',ylabel='$',folder= folder)
                        self.plot_box(df,figname="cp_buybox",y='userbuy',ylabel='$',folder= folder)
                        self.plot_box(df,figname="cp_userWbox",y='utilbenefit',ylabel='$',folder= folder)
                        self.plot_boxptshare(df,figname="cp_PTuserWbox",y='utilbenefit',ylabel='$',folder= folder)
                else:
                    self.plot_token_trade_amount(folder = folder)
                    self.plot_user_trade_amount(folder = folder)
                    self.plot_heatmap(df,figname='Trinityutilheat',histbins=5,folder= folder)
                    self.plot_swconvergence(np.array(self.swvec)/self.numOfusers, folder = folder)
                    self.plot_priceconvergence(np.array(self.pricevec), folder = folder)
                    self.plot_ptshare_number(np.array(self.ptsharevec), folder = folder)
                    self.plot_ttconvergence(self.flowdf, folder = folder)
                    self.plot_flow(self.flowdf,figname='Trinityflow',toll=True,folder= folder,day=10)
                    self.plot_cumflow(self.flowdf,figname='Trinitycumflow',toll=True,folder= folder,day=10)
                    self.plot_tt(self.flowdf,figname='Trinitytt',toll=False,flow=True,histbins =5,folder= folder,day=1)
                    self.plot_box(df,figname="tri_ttbox",y='TT',ylabel='min',folder= folder)
                    self.plot_box(df,figname="tri_sdebox",y='SDE',ylabel='min',folder= folder)
                    self.plot_box(df,figname="tri_sdlbox",y='SDL',ylabel='min',folder= folder)
                    self.plot_box(df,figname="tri_sellbox",y='usersell',ylabel='$',folder= folder)
                    self.plot_box(df,figname="tri_buybox",y='userbuy',ylabel='$',folder= folder)
                    self.plot_box(df,figname="tri_netbox",y='usernet',ylabel='$',folder= folder)
                    self.plot_box(df,figname="tri_userWbox",y='utilbenefit',ylabel='$',folder= folder)
                    self.plot_boxptshare(df,figname="tri_PTuserWbox",y='utilbenefit',ylabel='$',folder= folder)
            if self.verbose == 'personalizationopt':
                return {'obj':obj, 'RR':self.regulator.RR, 'UB':[np.average(i['utilbenefit']) for i in data_to_plot]}
            if self.verbose:
                dicttosave = {"obj": obj,"objcv":objcv,"Tollcollected": 2*np.sum(self.userBuy),"ntgini": gini(self.users.I),"gini":gini(self.users.I+userBenefits),"grpcv": np.array([np.average(i['cv']) for i in data_to_plot]) ,
                        "grpmce":np.array([np.average(i['mce']) for i in data_to_plot]),"avgtt":np.average(TT),"avgsde":np.average(SDE),"avgsdl":np.average(SDL),"RR": 2*self.regulator.RR,"ginicv":gini(self.users.I+self.users.userCV),
                        "grpa": np.array([np.average(i['a']) for i in data_to_plot]),"dataframe":df,'numUndesired':np.sum(self.numOfundesiredTrading),"sw":self.swvec,"pricevec":self.pricevec,
                        "Buy tc": np.sum(self.userBuytc),"Sell tc": np.sum(self.userSelltc),"Buy":np.sum(self.userBuy),"Sell":np.sum(self.userSell),"ptshare": self.ptsharevec,"flowconvergevec ":self.flowconvergevec,
                        "avgcartt":np.average(TT[self.users.predayDeparture!=-1]),"avgcarsde": np.average(SDE[self.users.predayDeparture!=-1]),"avgcarsdl": np.average(SDL[self.users.predayDeparture!=-1]),
                        "flowdf":self.flowdf,"toll":self.toll,"numOfbindingI":self.users.get_numOfbindingI(),"numberOfPT":len(self.users.ptshare),
                        'allowance': 2*np.sum(allowance),'usertradedf':self.usertradedf, 'tokentradedf':self.tokentradedf}
                save_obj(dicttosave,self.save_dfname)

                return 	{"obj": obj,"objcv":objcv,"Tollcollected": 2*np.sum(self.userBuy),"ntgini": gini(self.users.I),"gini":gini(self.users.I+userBenefits),"grpcv": np.array([np.average(i['cv']) for i in data_to_plot]) ,
                        "grpmce":np.array([np.average(i['mce']) for i in data_to_plot]),"avgtt":np.average(TT),"avgsde":np.average(SDE),"avgsdl":np.average(SDL),"RR": 2*self.regulator.RR,"ginicv":gini(self.users.I+self.users.userCV),
                        "grpa": np.array([np.average(i['a']) for i in data_to_plot]),"dataframe":df,'numUndesired':np.sum(self.numOfundesiredTrading),"sw":self.swvec,"pricevec":self.pricevec,
                        "Buy tc": np.sum(self.userBuytc),"Sell tc": np.sum(self.userSelltc),"Buy":np.sum(self.userBuy),"Sell":np.sum(self.userSell),"ptshare": self.ptsharevec,"flowconvergevec ":self.flowconvergevec,
                        "avgcartt":np.average(TT[self.users.predayDeparture!=-1]),"avgcarsde": np.average(SDE[self.users.predayDeparture!=-1]),"avgcarsdl": np.average(SDL[self.users.predayDeparture!=-1]),
                        "numOfbindingI":self.users.get_numOfbindingI(),"numberOfPT":len(self.users.ptshare)}
            else:
                return obj

    def aggdfbyTbyI(self,predayDeparture):
        df = pd.DataFrame({'departure':np.where(predayDeparture!=-1,np.mod(predayDeparture,self.hoursInA_Day*60),predayDeparture),'dailyincome':self.users.I})
        incinterval = np.zeros(6)
        incinterval[1] = df.dailyincome.quantile(0.25)
        incinterval[2] = df.dailyincome.quantile(0.5)
        incinterval[3] = df.dailyincome.quantile(0.75)
        incinterval[4] = df.dailyincome.quantile(0.9)
        incinterval[5] = math.inf
        df['Irange'] = pd.cut(df["dailyincome"],incinterval)
        
        # only works for step toll
        timeinterval = np.zeros(9)
        timeinterval[0] = -1000
        timeinterval[2:8] = self.tollparams[5:]
        timeinterval[8] = self.hoursInA_Day*60
        df['Trange'] = pd.cut(df["departure"],timeinterval)
        
        res = df.groupby(['Trange','Irange'],as_index=False).departure.count()
        res['departure'].fillna(0,inplace=True)
        res['Trange'] = res['Trange'].astype(str)
        return res
    
    def numbytimebyinc(self,df,tval = '(380.0, 425.0]'):
        tol = np.sum(df[df['Trange']==tval]['departure'])
        byI = []
        for i in pd.unique(df['Irange']):
            byI.append(np.sum(df[(df['Trange']==tval)&(df['Irange']==i)]['departure']))
        return byI
    
    def arcelasticity(self,d1,d2,c1,c2):
        perd = (d2-d1)/((d1+d2)/2)
        perc = (c2-c1)/((c1+c2)/2)
        return perd/perc
    
    def demand_dist(self,predayDeparture1,predayDeparture2,p1,p2,tval='(420.0, 480.0]'):
        print("p1:",p1,"p2:",p2)
        demand1 = self.aggdfbyTbyI(predayDeparture1)
        demand2 = self.aggdfbyTbyI(predayDeparture2)
        numbytval1 = self.numbytimebyinc(demand1,tval = tval)
        numbytval2 = self.numbytimebyinc(demand2,tval = tval)
        elas = []
        for i in range(len(numbytval1)):
            elas.append(self.arcelasticity(numbytval1[i],numbytval2[i],p1,p2))
        print("travel share by I before:", numbytval1)
        print("travel share by I after:", numbytval2)
        print("peak arc elasticity by income: ", elas)
        print("peak arc elasticity tol: ",self.arcelasticity(sum(numbytval1),sum(numbytval2),p1,p2))

    def plot_heatmap(self,df,figname,histbins=5,folder='.'):
        fig, ax = plt.subplots(figsize=(12,9))
        # df = df[df["departure"]>0]
        # df = df.groupby(['user'],as_index=False).mean()
        # df['PAT_range'] = pd.cut(df["PAT"], np.arange(0,725,histbins))
        # df = df.groupby(["PAT_range", "dailyincome"] )['utilbenefit']
        # print(df)
        # Generate some test data
        # df['Iquintile'] = pd.qcut(df['dailyincome'], 5, labels=False)
        # x = df['Iquintile'].values
        # y = df['utilbenefit'].values
        # heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
        # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        # plt.imshow(heatmap.T, extent=extent, origin='lower')
        # df['Iquintile'] = pd.qcut(df['dailyincome'], 5, labels=False)
        # plt.imshow(df.groupby(['Iquintile'],as_index=False).mean()['utilbenefit'].values,cmap='hot')
        from scipy.stats import gaussian_kde
        x = df['dailyincome'].values
        y = df['utilbenefit'].values
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        ax.scatter(x, y, c=z)
        plt.xlim((0,np.percentile(x, 95)))
        plt.ylim((-15,100))
        print("percetange positive benefit: ",np.sum(y>0)/len(y))
        plt.xlabel("Daily Disposable Income",fontsize=15)
        plt.ylabel("Benefits",fontsize=15)
        ax.tick_params(axis='x', labelsize= 15)
        ax.tick_params(axis='y', labelsize= 15)
        fig.savefig(folder+'/'+figname, dpi=fig.dpi)

    def plot_boxptshare(self,df,figname,y,ylabel,folder='.'):
        fig, ax = plt.subplots(figsize=(12,9))
        selectdf = df[df["departure"]<0]
        print(selectdf[['SDE','TT','SDL','usersell','userbuy','userToll','utilbenefit']].describe())
        sns.stripplot(data=selectdf, y=y)
        sns.violinplot(y=y, data=selectdf,cut=0,inner=None, color=".8")
        # ax.set_xticklabels(['Group1','Group2','Group3','Group4','Group5'])
        plt.xlabel("ptfare",fontsize=15)
        plt.ylabel(ylabel,fontsize=15)
        # if y == 'utilbenefit':
            # plt.ylim((-10,50))
        ax.tick_params(axis='x', labelsize= 15)
        ax.tick_params(axis='y', labelsize= 15)
        fig.savefig(folder+'/'+figname, dpi=fig.dpi)

    def plot_box(self,df,figname,y,ylabel,folder='.'):
        fig, ax = plt.subplots(figsize=(12,9))
        incquantile = [0]
        for i in [0.25,0.5,0.75,0.9]:
            incquantile.append(df.dailyincome.quantile(i))
        incquantile.append(math.inf)
        df['grp'] = (pd.cut(df["dailyincome"], incquantile))
        sns.boxplot(data=df, x='grp', y=y)
        # sns.violinplot(x="grp", y=y, data=df,cut=0,inner=None, color=".8")
        ax.set_xticklabels(['Group1','Group2','Group3','Group4','Group5'])
        plt.xlabel("Income group",fontsize=15)
        plt.ylabel(ylabel,fontsize=15)
        if y == 'utilbenefit':
            plt.ylim((-15,100))
        ax.tick_params(axis='x', labelsize= 15)
        ax.tick_params(axis='y', labelsize= 15)
        fig.savefig(folder+'/'+figname, dpi=fig.dpi)

    def plot_todutil(self,df,figname,toll=True,histbins =5,folder="."):
        df = df[df["departure"]>0]
        df_gb = df.groupby(pd.cut(df["departure"], np.arange(0,725,histbins))).mean()
        fig, ax = plt.subplots(figsize=(12,9))
        plt.plot(np.arange(200,720,histbins)/60, df_gb['util'].values[40:144],label='util')
        plt.xlabel("Time (hr)",fontsize=15)
        plt.ylabel("$ (5-min)",fontsize=15)
        plt.legend(fontsize=15)
        ax.tick_params(axis='x', labelsize= 15)
        ax.tick_params(axis='y', labelsize= 15)
        # if toll:
        #     ax2=ax.twinx()
        #     ax2.plot(np.arange(200,720)/60,self.toll[np.arange(200,720)],color='black')
        #     ax2.tick_params(axis='y', labelsize= 15)
        #     ax2.set_ylabel('Toll ($)',fontsize = 15)
        fig.savefig(folder+'/'+figname, dpi=fig.dpi)

    def plot_tt(self,df,figname,toll=False,flow=True,histbins =5,folder=".",day=1):
        df = df[df["departure"]>0]
        df_1 = df[ : 1*self.numOfusers]
        df_1 = df_1.groupby(['user'],as_index=False).mean()
        trips_tt_1 = df_1.groupby(pd.cut(df_1["departure"], np.arange(0,725,histbins))).mean()
        trips_tt_1.fillna(self.fftt,inplace=True)

        df_26 = df[-5*self.numOfusers: -4*self.numOfusers]
        df_26 = df_26.groupby(['user'],as_index=False).mean()
        trips_tt_26 = df_26.groupby(pd.cut(df_26["departure"], np.arange(0,725,histbins))).mean()
        trips_tt_26.fillna(self.fftt,inplace=True)

        df_27 = df[-4*self.numOfusers: -3*self.numOfusers]
        df_27 = df_27.groupby(['user'],as_index=False).mean()
        trips_tt_27 = df_27.groupby(pd.cut(df_27["departure"], np.arange(0,725,histbins))).mean()
        trips_tt_27.fillna(self.fftt,inplace=True)
        
        df_28 = df[-3*self.numOfusers: -2*self.numOfusers]
        df_28 = df_28.groupby(['user'],as_index=False).mean()
        trips_tt_28 = df_28.groupby(pd.cut(df_28["departure"], np.arange(0,725,histbins))).mean()
        trips_tt_28.fillna(self.fftt,inplace=True)

        df_29 = df[-2*self.numOfusers: -1*self.numOfusers]
        df_29 = df_29.groupby(['user'],as_index=False).mean()
        trips_tt_29 = df_29.groupby(pd.cut(df_29["departure"], np.arange(0,725,histbins))).mean()
        trips_tt_29.fillna(self.fftt,inplace=True)

        df_30 = df[-1*self.numOfusers:]
        df_30 = df_30.groupby(['user'],as_index=False).mean()
        trips_tt_30 = df_30.groupby(pd.cut(df_30["departure"], np.arange(0,725,histbins))).mean()
        trips_tt_30.fillna(self.fftt,inplace=True)

        fig, ax = plt.subplots(figsize=(12,9))
        label_ls = [str(0)+"th day", str(self.numOfdays-4)+"th day", str(self.numOfdays-3)+"th day", str(self.numOfdays-2)+"th day", str(self.numOfdays-1)+"th day", str(self.numOfdays)+"th day"]
        plt.plot(np.arange(200,720,histbins)/60, trips_tt_1['tt'].values[40:144],label=label_ls[0])
        plt.plot(np.arange(200,720,histbins)/60, trips_tt_26['tt'].values[40:144],label=label_ls[-5])
        plt.plot(np.arange(200,720,histbins)/60, trips_tt_27['tt'].values[40:144],label=label_ls[-4])
        plt.plot(np.arange(200,720,histbins)/60, trips_tt_28['tt'].values[40:144],label=label_ls[-3])
        plt.plot(np.arange(200,720,histbins)/60, trips_tt_29['tt'].values[40:144],label=label_ls[-2])
        plt.plot(np.arange(200,720,histbins)/60, trips_tt_30['tt'].values[40:144],label=label_ls[-1])
        
        plt.xlabel("Time (hr)", fontsize=15)
        plt.ylabel("TT (5-min)", fontsize=15)
        ax.tick_params(axis='x', labelsize= 15)
        ax.tick_params(axis='y', labelsize= 15)
        plt.legend(fontsize=15)
        fig.savefig(folder+'/'+figname, dpi=fig.dpi)
        
    # Plot the transaction 
    def plot_user_trade_amount(self, folder="."):
        df = self.usertradedf.fillna(0)
        fig, ax = plt.subplots(figsize=(12,9))
        plt.plot(np.arange(0,720)/60, df["sell"]/self.numOfusers, label = "Sell")
        plt.plot(np.arange(0,720)/60, df["buy"]/self.numOfusers, label = "Buy")
        plt.ylabel("Transaction (%)",fontsize=15)
        plt.xlabel("Time (hr)",fontsize=15)
        ax.tick_params(axis='x', labelsize= 15)
        ax.tick_params(axis='y', labelsize= 15)
        plt.legend(fontsize=15)
        fig.savefig(folder+'/transaction amount', dpi=fig.dpi)

# Plot the average token trade amount
    def plot_token_trade_amount(self, folder="."):
        df = self.tokentradedf.fillna(0)
        fig, ax = plt.subplots(figsize=(12,9))
        plt.plot(np.arange(0,720)/60, df["sell"], label = "Sell")
        plt.plot(np.arange(0,720)/60, df["buy"], label = "Buy")
        plt.ylabel("Average amount ($)",fontsize=15)
        plt.xlabel("Time (hr)",fontsize=15)
        ax.tick_params(axis='x', labelsize= 15)
        ax.tick_params(axis='y', labelsize= 15)
        plt.legend(fontsize=15)
        fig.savefig(folder+'/token average amount', dpi=fig.dpi)

    def plot_speed(self,df,figname,histbins=5,folder='.',day=1):
        df = df[-day*self.numOfusers:]
        df = df[df["departure"]>0]
        roadlen = 45/6
        df['speed'] = roadlen/(df['tt']/60)
        df = df.groupby(['user'],as_index=False).mean()
        trips_speed = df.groupby(pd.cut(df["departure"], np.arange(0,725,histbins))).mean()
        trips_speed.fillna(45,inplace=True)
        fig, ax = plt.subplots(figsize=(12,9))
        plt.plot(np.arange(200,720,histbins)/60, trips_speed['speed'].values[40:144],label='speed')
        plt.xlabel("Time (hr)",fontsize=15)
        plt.ylabel("Speed (5-min)",fontsize=15)
        ax.tick_params(axis='x', labelsize= 15)
        ax.tick_params(axis='y', labelsize= 15)
        plt.legend(fontsize=15)
        fig.savefig(folder+'/'+figname, dpi=fig.dpi)

    def plot_cumflow(self,df,figname,toll=True,histbins =5,folder=".",day=1):
        df = df[-day*self.numOfusers:]
        df = df[df["departure"]>0]
        df = df.groupby(['user'],as_index=False).mean()
        trips_flow = df.groupby(pd.cut(df["departure"], np.arange(0,725,histbins))).count()
        fig, ax = plt.subplots(figsize=(12,9))
        plt.plot(np.arange(200,720,histbins)/60, np.cumsum(trips_flow['departure'].values[40:144]),label='cumdep')
        trips_arr = df.groupby(pd.cut(df["arrival"], np.arange(0,725,histbins))).count()
        plt.plot(np.arange(200,720,histbins)/60, np.cumsum(trips_arr['arrival'].values[40:144]),label='cumarr')
        plt.xlabel("Time (hr)",fontsize=25)
        plt.ylabel("Cumulative flow (5-min)",fontsize=25)
        ax.tick_params(axis='x', labelsize= 20)
        ax.tick_params(axis='y', labelsize= 20)
        plt.legend(fontsize=25)
        fig.savefig(folder+'/'+figname, dpi=fig.dpi)

    def plot_toll(self, folder):
        fig, ax = plt.subplots(figsize=(12,9))
        plt.plot(np.arange(200,720)/60, self.toll[np.arange(200,720)], color='black')
        ax.tick_params(axis='y', labelsize= 20)
        plt.ylabel('Toll ($)',fontsize = 25)
        plt.xlabel("Time (hr)",fontsize=25)
        fig.savefig(folder+'/toll profile.png', dpi=fig.dpi)


    def plot_flow(self,df,figname,toll=True,histbins =5,folder=".",day=1):
        df = df[df["departure"]>0]
        df_26 =  df[-5*self.numOfusers:-6*self.numOfusers]
        trips_flow_26 = np.histogram(df_26["departure"], bins =144, range = (0, 720))[0]
        df_trips_flow_26 =pd.DataFrame(data = trips_flow_26.tolist(), columns =[ "Flow (5-min)"])
        df_trips_flow_26["Time (hr)"] = np.arange(0, 720, 5)/60

        df_27 = df[-4*self.numOfusers:-3*self.numOfusers]
        trips_flow_27 = np.histogram(df_27["departure"],bins =144, range = (0, 720))[0]
        df_trips_flow_27 =pd.DataFrame(data = trips_flow_27.tolist(), columns =[ "Flow (5-min)"])
        df_trips_flow_27["Time (hr)"] = np.arange(0, 720, 5)/60

        df_28 = df[-3*self.numOfusers:-2*self.numOfusers]
        trips_flow_28 = np.histogram(df_28["departure"],bins =144, range = (0, 720))[0]
        df_trips_flow_28 =pd.DataFrame(data = trips_flow_28.tolist(), columns =[ "Flow (5-min)"])
        df_trips_flow_28["Time (hr)"] = np.arange(0, 720, 5)/60

        df_29 = df[-2*self.numOfusers:-1*self.numOfusers]
        trips_flow_29 = np.histogram(df_29["departure"],bins =144, range = (0, 720))[0]
        df_trips_flow_29 =pd.DataFrame(data = trips_flow_29.tolist(), columns =[ "Flow (5-min)"])
        df_trips_flow_29["Time (hr)"] = np.arange(0, 720, 5)/60

        df_30 = df[-1*self.numOfusers:]
        trips_flow_30 = np.histogram(df_30["departure"],bins =144, range = (0, 720))[0]
        df_trips_flow_30 =pd.DataFrame(data = trips_flow_30.tolist(), columns =[ "Flow (5-min)"])
        df_trips_flow_30["Time (hr)"] = np.arange(0, 720, 5)/60

        df_con= pd.concat([df_trips_flow_26, df_trips_flow_27, df_trips_flow_28, df_trips_flow_28, df_trips_flow_29, df_trips_flow_30])
        # 95% confidence-inteval 
        fig, ax = plt.subplots(figsize=(12,9))
        ax = sns.lineplot(data=df_con, x= "Time (hr)" , y="Flow (5-min)" )
        plt.xlabel("Time (hr)",fontsize=25)
        plt.ylabel("Flow (5-min)",fontsize=25)
        plt.savefig(folder+'/'+figname)

    def plot_swconvergence(self,swvec, folder="."):
        fig, ax = plt.subplots(figsize=(12,9))
        plt.plot(swvec,label='Welfare')
        plt.xlabel("Day",fontsize=25)
        plt.ylabel("Social welfare per capita($)",fontsize=25)
        # plt.title("social welfare convergence")
        # plt.legend(fontsize=20)
        ax.tick_params(axis='x', labelsize= 20)
        ax.tick_params(axis='y', labelsize= 20)
        fig.savefig(folder+'/social welfare converge.png', dpi=fig.dpi)

    def plot_priceconvergence(self,pricevec,folder="."):
        fig, ax = plt.subplots(figsize=(12,9))
        plt.plot(pricevec,label='price')
        plt.xlabel("Day",fontsize=25)
        plt.ylabel("Price ($)",fontsize=25)
        # plt.legend(fontsize=25)
        ax.tick_params(axis='x', labelsize= 20)
        ax.tick_params(axis='y', labelsize= 20)
        fig.savefig(folder+'/price converge.png', dpi=fig.dpi)

    def plot_ttconvergence(self, df, folder="."):
        df_1 = df.fillna(0)
        ttvec = []
        for i in range(self.numOfdays):
            ttvec.append(np.mean(df_1[i*self.numOfusers: (i+1)*self.numOfusers]["tt"]))
        fig, ax = plt.subplots(figsize=(12,9))
        plt.plot(ttvec,label='Daily average travel time')
        plt.xlabel("Day",fontsize=25)
        plt.ylabel("Average travel time (min)",fontsize=25)
        ax.tick_params(axis='x', labelsize= 20)
        ax.tick_params(axis='y', labelsize= 20)
        fig.savefig(folder+'/travel time converge.png', dpi=fig.dpi)

    def plot_ptshare_number(self, ptsharevec, folder="."):
        fig, ax = plt.subplots(figsize=(12,9))
        plt.plot(ptsharevec,label='PT share number')
        plt.xlabel("Day",fontsize=25)
        plt.ylabel("Public transit traveler number",fontsize=25)
        # plt.title("PT share")
        # plt.legend(fontsize=20)
        ax.tick_params(axis='x', labelsize= 20)
        ax.tick_params(axis='y', labelsize= 20)
        fig.savefig(folder+'/pt converge.png', dpi=fig.dpi)

    def plot_indwelfare(self,x,y1,xlim=(0,50),ylim=(0,100),figname = "Income.png",folder ="."):
        fig, ax = plt.subplots(figsize=(12,9))
        plt.scatter(x,y1,label='income')
        plt.xlabel('half day disposable income ($)',fontsize = 20)
        plt.ylabel('individual welfare ($)',fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.ylim(ylim)
        plt.xlim(xlim)
        fig.savefig(folder+'/'+figname, dpi=fig.dpi)

    def plot_indallowance(self,x,y1,xlim=(0,50),ylim=(0,100),figname = "allowance.png",folder ="."):
        fig, ax = plt.subplots(figsize=(12,9))
        plt.scatter(x,y1,label='income')
        plt.xlabel('half day disposable income ($)',fontsize = 20)
        plt.ylabel('individual allowance ($)',fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.ylim(ylim)
        plt.xlim(xlim)
        fig.savefig(folder+'/'+figname, dpi=fig.dpi)



def main():
    print("\n")
    scenario = "NT"
    tollparams = [7.90084456,65.01906779,2.49667875]
    toll_type = "normal"
    print("Scenario: ", scenario)

    if scenario == 'CP':
        allowance = {'policy': 'personalization','ctrl':1.125,'cap':float("inf")}
    else:
        allowance = {'policy': False,'ctrl':1.048,'cap':float("inf")}
    marketPrice = 1
    # only applies if scenario is Trinity
    if scenario == 'Trinity':
        allocation = {'AR':0.00269,'way':'continuous','FTCs': 0.05,'FTCb':0.05,'PTCs': 0.00,'PTCb':0.00,"Decaying": False}
    else:
        allocation = {'AR':0.0,'way':'lumpsum','FTCs': 0,'FTCb':0,'PTCs': 0,'PTCb':0,"Decaying":False}
    CV = False
    if scenario == 'NT':
        CV = False
        
    start_time = timeit.default_timer()
    simulator = Simulation(_numOfdays= numOfdays, _user_params = user_params,
                            _scenario=scenario,_allowance=allowance, 
                            _marketPrice=marketPrice, _allocation = allocation,
                            _deltaP = deltaP, _numOfusers=numOfusers, _RBTD = RBTD, _Tstep=Tstep, 
                            _Plot = Plot, _seed = seed, _verbose = verbose, 
                            _unusual = unusual, _storeTT=storeTT, _CV=CV, save_dfname='./output/Bottleneck/NT/NT',  toll_type=toll_type)
    simulator.simulate(tollparams)
    res = simulator.metrics()
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    print(res)




    print("\n")
    scenario = "Trinity"
    tollparams = [7.90084456,65.01906779,2.49667875]
    toll_type = "normal"   
    print("Scenario: ", scenario, " Toll: ",  toll_type)

    if scenario == 'CP':
        allowance = {'policy': 'personalization','ctrl':1.125,'cap':float("inf")}
    else:
        allowance = {'policy': False,'ctrl':1.048,'cap':float("inf")}
    marketPrice = 1
    # only applies if scenario is Trinity
    if scenario == 'Trinity':
        allocation = {'AR':0.00269,'way':'continuous','FTCs': 0.05,'FTCb':0.05,'PTCs': 0.00,'PTCb':0.00,"Decaying": False}
    else:
        allocation = {'AR':0.0,'way':'lumpsum','FTCs': 0,'FTCb':0,'PTCs': 0,'PTCb':0,"Decaying":False}
    CV = False
    if scenario == 'NT':
        CV = False

    start_time = timeit.default_timer()
    simulator = Simulation(_numOfdays= numOfdays, _user_params = user_params,
                            _scenario=scenario,_allowance=allowance, 
                            _marketPrice=marketPrice, _allocation = allocation,
                            _deltaP = deltaP, _numOfusers=numOfusers, _RBTD = RBTD, _Tstep=Tstep, 
                            _Plot = Plot, _seed = seed, _verbose = verbose, 
                            _unusual = unusual, _storeTT=storeTT, _CV=CV, save_dfname='./output/Bottleneck/Trinity_normal/Trinity_normal', toll_type=toll_type)
    simulator.simulate(tollparams)
    res = simulator.metrics()
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    print(res)




    print("\n")
    scenario = "Trinity"
    toll_type = "step"
    tollparams = [1.72371018e+00 ,3.77113898e+00, 4.03201588e+00, 1.13938644e-01,
                  2.62033233e+00, 3.77824154e+02, 3.92469832e+02, 4.57734666e+02,
                  5.61573844e+02, 5.71905883e+02, 5.92428327e+02]
    print("Scenario: ", scenario, " Toll: ",  toll_type)

    if scenario == 'CP':
        allowance = {'policy': 'personalization','ctrl':1.125,'cap':float("inf")}
    else:
        allowance = {'policy': False,'ctrl':1.048,'cap':float("inf")}
    marketPrice = 1
    # only applies if scenario is Trinity
    if scenario == 'Trinity':
        allocation = {'AR':0.00269,'way':'continuous','FTCs': 0.05,'FTCb':0.05,'PTCs': 0.00,'PTCb':0.00,"Decaying": False}
    else:
        allocation = {'AR':0.0,'way':'lumpsum','FTCs': 0,'FTCb':0,'PTCs': 0,'PTCb':0,"Decaying":False}
    CV = False
    if scenario == 'NT':
        CV = False

    start_time = timeit.default_timer()
    simulator = Simulation(_numOfdays= numOfdays, _user_params = user_params,
                            _scenario=scenario,_allowance=allowance, 
                            _marketPrice=marketPrice, _allocation = allocation,
                            _deltaP = deltaP, _numOfusers=numOfusers, _RBTD = RBTD, _Tstep=Tstep, 
                            _Plot = Plot, _seed = seed, _verbose = verbose, 
                            _unusual = unusual, _storeTT=storeTT, _CV=CV, save_dfname='./output/Bottleneck/Trinity_step/Trinity_step', toll_type=toll_type)
    simulator.simulate(tollparams)
    res = simulator.metrics()
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    print(res)


if __name__ == "__main__":
    start_time = time.time()
    print("start_time ",time.asctime(time.localtime(start_time)))

    main()
    
    end_time = time.time()
    print("end_time ",time.asctime(time.localtime(end_time)))
    print("total elapsed time ",end_time-start_time)
