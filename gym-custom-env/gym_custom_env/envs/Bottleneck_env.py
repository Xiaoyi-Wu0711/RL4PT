import pandas as pd
from numba import njit, prange
import numpy as np

_capacity = 42


def gini(x):
     # (Warning: This is a concise implementation, but it is O(n**2)
     # in time and memory, where n = len(x).  *Don't* pass in huge
     # samples!)

     # Mean absolute difference
     mad = np.abs(np.subtract.outer(x, x)).mean()
     # Relative mean absolute difference
     rmad = mad/np.mean(x)
     # Gini scoefficient
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
                _fftt=24, _dist=18, _choiceInterval = 30, _seed=5843, _unusual=False,_CV=True,_numOfdays=50):

        self.AR = _allocation['AR']
        self.ARway = _allocation['way'] # It means that the allocation or distribution is made as a one-time, whole amount, rather than being spread out over a period of time or divided into smaller installments.
        self.FTCs = _allocation['FTCs']
        self.FTCb = _allocation['FTCb']
        self.PTCs = _allocation['PTCs']
        self.PTCb = _allocation['PTCb']
        
        self.users = np.arange(_numOfusers)
        self.numOfusers = _numOfusers
        self.hoursInA_Day = _hoursInA_Day
        self.Tstep = _Tstep
        self.fftt = _fftt # free flow travel time
        self.dist = _dist
        self.mpg = 23 
        self.fuelprice = 4
        self.ptfare = 2 # public transport fees
        self.ptspeed = 25
        self.pttt = self.dist/self.ptspeed*60 # public travel time in minutes
        self.ffspeed = 45
        self.ptheadway = 10

        self.choiceInterval = _choiceInterval # departure interval h
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
        self.userCV = np.zeros(self.numOfusers) # calculate the user's willingness to pay for the shared ride based on their cost variation (CV) measure.
        self.norm_list = []
        # initialize user accounts
        self.userAccounts = np.zeros(self.numOfusers)+self.AR*self.hoursInA_Day*60
        self.distribution = np.zeros(self.numOfusers) # modify the allowance distribution for each user based on the defined policy.

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
    
        # Generate travelers parameters (vot, sde, sdl, mu, epsilon, income)
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
        # print("annual income ginit",gini(annualincome),"remaining I gini",gini(self.I))	

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
        a_exp = 2 * np.repeat(distribution,numOfcost).reshape(numOfdist,numOfcost)
        # cost is positive
        cost = np.tile(Th,(numOfdist,1))
        income_c = self.user_params['lambda']*np.log(self.user_params['gamma'] + I - cost + a_exp)
        sys_util_c = -cost + income_c + a_exp
        sys_util = sys_util_t + sys_util_c
        util = sys_util + utileps
        idx = np.argmax(util,axis=1)
        MUI = 1 + self.user_params['lambda']/(self.user_params['gamma'] + I-Th[idx] + distribution)
        a_index = (np.abs(MUI - x)).argmin()
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
    
    # calculate the utility and make choice based on logit model
    # Make preday choice according to attributes (travel time, schedule delay, toll)
    # PT user: suppose all tw windows are taken by cars
    # Car user: suppose N-1 tw windows with 1 real travel experience time 
    def update_choice(self, _predictedTT, _beginTime, _totTime,_toll, RR, price,day):
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
        # th_arr = np.zeros((self.numOfusers, self.choiceInterval*2+1+1)) # one additional slot for no travel
        # utileps_arr = np.zeros((self.numOfusers, self.choiceInterval*2+1+1)) # one additional slot for no travel
        # sysutilt_arr = np.zeros((self.numOfusers, self.choiceInterval*2+1+1))
        # tautilde_arr = np.zeros((self.numOfusers, self.choiceInterval*2+1+1))
        # ch_arr = np.zeros((self.numOfusers, self.choiceInterval*2+1+1))

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
            ch = np.zeros(len(utileps))
            ASC = np.zeros(len(utileps))
            W = np.zeros(1+len(possibleDepartureTimes)) # waiting time
            W[0] = 1/2*self.ptheadway
            # ASC[0] = self.betant # no more no travel
            sysutil_t = ASC+(-2*vot *tautilde  - sde * SDE - sdl * SDL-2*vow*W)

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
            # th_arr[user,:] = (ch/2+prev_allowance)*2
            # utileps_arr[user,:] = utileps
            # sysutilt_arr[user,:] = 	sysutil_t
            # tautilde_arr[user,:] = 	tautilde
            # ch_arr[user,:] = 	ch

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

        if self.scenario == 'NT':
            sysutil_arr[user,:] = sysutil
            np.save("./output/Bottleneck/NT/NT_sysutil",sysutil_arr)

    # realize selling and buying behavior
    def sell_and_buy(self, _t, _currToll, _toll, _price, _totTime):

        FW = self.hoursInA_Day*60*self.AR
        userBuy = np.zeros_like(self.userAccounts)
        userSell = np.zeros_like(userBuy)
        p = _price
        departureTime  = self.predayDeparture.copy()
        mask_cansell = np.where(departureTime!=-1, departureTime!=_t, True)
        departureTime = np.where(departureTime!=-1, np.where(departureTime<_t, departureTime + self.hoursInA_Day * 60, departureTime), departureTime)

        if self.ARway == 'lumpsum':
            userBuy[~mask_cansell] = np.maximum((_currToll - self.userAccounts)[~mask_cansell],0)
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
            self.userAccounts[~mask_cansell] = np.maximum((self.userAccounts-_currToll)[~mask_cansell], 0)
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
            mask_sellnow = (mask_cansell & mask_positiveprofit ) & (mask_needbuy | mask_FW | mask_needbuynext)
            userSell[mask_sellnow] = self.userAccounts[mask_sellnow]

            #### update accounts
            currTime = np.mod(_t, self.hoursInA_Day*60) # range from 0 to hoursInA_Day*60
            # handle selling
            self.userAccounts[mask_sellnow] = self.AR # sell all and get new allocation

            # handle paying toll and buying
            self.userAccounts[~mask_cansell] = np.maximum((self.userAccounts-_currToll)[~mask_cansell],0)
            self.userAccounts[~mask_cansell] = np.minimum(self.userAccounts[~mask_cansell]+self.AR,FW) # add new allocation and cap it at FW
            # handle do nothing (expire oldest tokens if reach maximum life time and get new allocation)
            mask_donothing = ~(mask_sellnow | ~mask_cansell)
            self.userAccounts[mask_donothing] = np.minimum(self.userAccounts[mask_donothing]+self.AR,FW)
            
        return [userBuy, userSell]

    # actual arrival is the combination of estimated arrival(if he didn't go at those time points) 
    # and real arrival time(if he chooses this time point)
    def update_arrival(self, actualArrival):
        self.actualArrival = actualArrival

    # perform day to day learning
    def d2d(self, actualTT):
       self.actualTT = actualTT
       self.predictedTT = 0.9*self.predictedTT+0.1*self.actualTT 
       c_perceived=self.predictedTT
       c_cs = self.actualTT
       self.norm_list.append(np.linalg.norm(c_perceived-c_cs,ord=1)/self.numOfusers)

class Regulator():
    # regulator account balance
    def __init__(self, marketPrice=1, RBTD = 100, deltaP = 0.05):
        self.RR = 0
        self.tollCollected = 0
        self.allowanceDistributed = 0
        self.marketPrice = marketPrice
        self.RBTD = 100  # a constant threshold
        self.deltaP = 0.05

    # update regulator account
    def update_balance(self,userToll,userReceive):
        # userToll: regulator revenue
        # userReceive: regulator cost
        self.tollCollected = np.sum(userToll)
        self.allowanceDistributed = np.sum(userReceive)
        self.RR = self.tollCollected - self.allowanceDistributed
    
    # update token price
    def update_price(self):
        if self.RR > self.RBTD:
            self.marketPrice += self.deltaP
        elif self.RR < -self.RBTD:
            self.marketPrice -= self.deltaP


class Bottleneck_simulation():
    # simulate one day

    def __init__(self,
                  _user_params, 
                  _allocation,
                  _scenario='NT',
                  _allowance=False,
                  _numOfdays=50,
                  _numOfusers=7500,
                  _Tstep=1,
                  _hoursInA_Day=12,
                  _fftt=24,
                  _marketPrice = 1,
                  _RBTD = 100, 
                  _deltaP=0.05, 
                  _Plot = False, 
                  _seed=5843, 
                  _verbose = False,
                  _unusual=False, 
                  _storeTT=False,
                  _CV=True, 
                  save_dfname='CPresult.csv', 
                  toll_type="normal"
                ):
        self.numOfdays = _numOfdays
        self.hoursInA_Day = _hoursInA_Day
        self.numOfusers = _numOfusers
        self.allowance = _allowance
        self.save_dfname = save_dfname
        self.capacity = _capacity
        self.currday = 0
        self.fftt = _fftt
        self.user_params = _user_params
        self.Tstep = _Tstep
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
        self.decaying = _allocation['Decaying']
        self.flow_array = np.zeros((self.numOfdays, self.numOfusers, 3)) # departure, arrival, user, travel time
        self.usertrade_array = np.zeros((self.numOfdays,self.hoursInA_Day*60, 2)) # buy and sell 
        self.tokentrade_array = np.zeros((self.numOfdays, self.hoursInA_Day*60, 2)) # buy and sell 
        self.users = Travelers(self.numOfusers,_user_params=self.user_params,_allocation=_allocation,
                             _fftt=_fftt, _hoursInA_Day=_hoursInA_Day,_Tstep=self.Tstep,
                            _allowance=self.allowance, _scenario = self.scenario, _seed=_seed, 
                            _unusual=self.unusual, _CV = _CV,_numOfdays = _numOfdays)
        self.users.generate_params()
        self.regulator = Regulator(_marketPrice,_RBTD,_deltaP)
        self.originalAtt = {}
        self.presellAfterdep = np.zeros(self.numOfusers,dtype=int)
        self.toll_type = toll_type
        timeofday = np.arange(self.hoursInA_Day*60)
        self.toll = np.array([0]*len(timeofday))

    def steptoll_fxn(self,x,height,interval,mu):
        toll_profile =(x>=(mu-interval))*(x<(mu+interval))*3*height  \
            + (x>=(mu-2*interval))*(x<(mu-interval))*2.5*height  \
            + (x>=(mu+interval))*(x<(mu+2*interval))*2.5*height  \
            + (x>=(mu-3*interval))*(x<(mu-2*interval))*2*height \
            + (x>=(mu+2*interval))*(x<(mu+3*interval))*2*height \
            + (x>=(mu+3*interval))*(x<(mu+4*interval))*1.5*height\
            + (x>=(mu-4*interval))*(x<(mu-3*interval))*1.5*height\
            + (x>=(mu-5*interval))*(x<(mu-4*interval))*height \
            + (x>=(mu+4*interval))*(x<(mu+5*interval))*height  
        return toll_profile

    # get the toll fee
    def bimodal(self,x, A, mu, sigma):
        toll_profile = A*np.exp(-(x-mu)**2/2/(sigma)**2)
        return toll_profile

    # MFD simulation
    def RL_simulateOneday(self, day, state_aggravate, space_shape):
        self.currday = day
        beginTime = day*self.hoursInA_Day*60
        totTime =  (day+1)*self.hoursInA_Day*60
        self.users.update_choice(self.users.predictedTT,beginTime,totTime,self.toll,self.regulator.tollCollected,self.regulator.marketPrice,self.currday)
        self.numOfundesiredTrading = np.zeros(self.numOfusers)
        sellTime = np.zeros(self.numOfusers)
        
        actualTT = np.zeros(self.hoursInA_Day*60)
        numDepart = np.zeros(self.hoursInA_Day*60)

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
            else:
                outputcounter = self.capacity 

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

            # if number of people departing now less than output capacity
            # we continue to dissipate people in queue
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
                if buy_user_amount != 0:
                    buyamount[t-beginTime] = np.sum(tempuserBuy*self.regulator.marketPrice)/buy_user_amount
                if sell_user_amount != 0:
                    sellamount[t-beginTime] = np.sum(tempuserSell*self.regulator.marketPrice)/sell_user_amount

                self.numOfundesiredTrading = np.where(((userSell >1e-6)|(self.presellAfterdep)) & (tempuserBuy>1e-6), 1, self.numOfundesiredTrading)
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
            # regulat

        self.usertrade_array[self.currday,:, 0] = buyvec
        self.usertrade_array[self.currday,:, 1] = sellvec
        self.tokentrade_array[self.currday,:, 0] = sellamount
        self.tokentrade_array[self.currday,:, 1] = buyamount

        actualTT = np.divide(actualTT, numDepart, out=np.zeros_like(actualTT)+self.fftt, where=numDepart!=0)
        self.users.update_arrival(actualArrival)

        self.flow_array[self.currday, :, 0]  =  np.maximum(self.users.predayDeparture-beginTime,-1)
        self.flow_array[self.currday, :, 1]  =  np.maximum(actualArrival-beginTime,-1)
        self.flow_array[self.currday, :, 2]  =  np.where(self.flow_array[self.currday, :, 0]!=-1, self.flow_array[self.currday, :, 1] -  self.flow_array[self.currday, :, 0] , self.users.pttt)
       
        # day to day learning
        # update regulator account balance at the end of day
        self.userSell = userSell
        self.userSelltc = userSelltc
        self.userBuytc = userBuytc
        self.userToll = userToll

        if self.scenario == 'Trinity':
            # update regulator balance
            self.userBuy = userBuy
            self.regulator.update_balance(self.userBuy+self.userBuytc, self.userSell+self.userSelltc)
        else:
            self.userBuy = userToll
            self.regulator.update_balance(userToll, self.users.distribution)

        market_price = self.regulator.marketPrice
        pt_share_number = len(self.users.ptshare)
        sw = self.calculate_sw()

        if self.unusual['unusual'] and self.currday == self.unusual['day']:
            self.users.d2d(self.users.predictedTT)
            self.regulator.marketPrice = self.originalAtt['price']
        else:
            # d2d learnining
            self.users.d2d(actualTT)
            if self.scenario == 'Trinity':
                # update token price
                self.regulator.update_price()	

        self.presellAfterdep = sellTime>self.users.predayDeparture

        if space_shape[0] == 5:
            state_ls = [numDepart, actualTT, buyamount, sellamount, self.toll]
            state = np.concatenate(state_ls)
            # aggravate the state from 1 min to 5 min
            encode_shape = int(self.hoursInA_Day*60/state_aggravate)
            state_encode = np.zeros(shape = encode_shape * len(state_ls))

            for j in range(encode_shape):
                state_encode[j] = np.mean(numDepart[j*state_aggravate:(j+1)*state_aggravate])
                state_encode[j+1*encode_shape] = np.mean(actualTT[j*state_aggravate:(j+1)*state_aggravate])
                state_encode[j+2*encode_shape] = np.mean(sellvec[j*state_aggravate:(j+1)*state_aggravate])
                state_encode[j+3*encode_shape] = np.mean(buyvec[j*state_aggravate:(j+1)*state_aggravate])
                state_encode[j+4*encode_shape] = np.mean(self.toll[j*state_aggravate:(j+1)*state_aggravate])
        
        elif space_shape[0] == 4:
            # TODO: average_accumulation, average_tt
            state_ls = [numDepart, buyamount, sellamount, self.toll]
            state = np.concatenate(state_ls)
            # aggravate the state from 1 min to 5 min
            encode_shape = int(self.hoursInA_Day*60/state_aggravate)
            state_encode = np.zeros(shape = encode_shape * len(state_ls))
            for j in range(encode_shape):
                state_encode[j] = np.mean(numDepart[j*state_aggravate:(j+1)*state_aggravate])
                state_encode[j+1*encode_shape] = np.mean(sellvec[j*state_aggravate:(j+1)*state_aggravate])
                state_encode[j+2*encode_shape] = np.mean(buyvec[j*state_aggravate:(j+1)*state_aggravate])
                state_encode[j+3*encode_shape] = np.mean(self.toll[j*state_aggravate:(j+1)*state_aggravate])
        
        elif space_shape[0] == 1:
            state_ls = [self.toll]
            state = np.concatenate(state_ls)
            # aggravate the state from 1 min to 5 min
            encode_shape = int(self.hoursInA_Day*60/state_aggravate)
            state_encode = np.zeros(shape = encode_shape * len(state_ls))
            for j in range(encode_shape):
                state_encode[j] = np.mean(numDepart[j*state_aggravate:(j+1)*state_aggravate])
        
        else:
            print(" not this type of state_shape")
            exit(1)
        return state_encode, market_price, pt_share_number, sw
    
    # calculate social welfare value
    def calculate_sw(self):
        TT = np.where(self.users.predayDeparture!=-1, self.users.actualArrival-self.users.predayDeparture, self.users.pttt) 
        SDE = np.where(self.users.predayDeparture!=-1, np.maximum(0, self.users.desiredArrival+self.currday*self.hoursInA_Day*60-self.users.actualArrival), 0)
        SDL = np.where(self.users.predayDeparture!=-1, np.maximum(0, self.users.actualArrival-(self.users.desiredArrival+self.currday*self.hoursInA_Day*60)), 0)
        allowance = self.users.distribution
        # either car fuel cost or transit fare
        fuelcost = np.where(self.users.predayDeparture!=-1,self.users.dist/self.users.mpg*self.users.fuelprice ,self.users.ptfare)
        ASC = np.zeros(self.numOfusers)
        ptwaitingtime = np.where(self.users.predayDeparture!=-1,0 ,self.users.ptheadway)
        util = ASC + (-2 * self.users.vot * TT - self.users.sde * SDE - self.users.sdl * SDL - self.users.waiting * ptwaitingtime
             + self.user_params['lambda'] * np.log(self.user_params['gamma'] + self.users.I - 2 * self.userBuy + 2 * self.userSell + 2 * allowance - 2 * fuelcost)
             + self.users.I - 2 * self.userBuy + 2 * self.userSell + 2 * allowance - 2 * fuelcost) + self.users.predayEps
        NTMUI = 1 + self.user_params['lambda']/(self.user_params['gamma'] + self.users.I) # no toll marginal utility
        if self.scenario == 'NT':
            obj = np.sum(util)
        else:
            NT_util = np.load("./output/Bottleneck/NT/NT_util.npy")
            userBenefits = (util - NT_util)
            obj = np.sum(userBenefits) + 2 * self.regulator.RR
        return obj

    
    
