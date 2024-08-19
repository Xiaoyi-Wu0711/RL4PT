import pandas as pd
from numba import njit, prange
import numpy as np
import math

ffspeed = 45
# capacity = int(7000*0.9)
capacity = int(7000)

# print(" capacity in MFD ", capacity)

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

@njit
def V1(x):
    return np.square(1 - x / capacity) * ffspeed

# (1-n/1000)^2*ffspeed*n, if the flow is larger than this, it is congested
@njit
def V(x):
    if isinstance(x, list):
        return [np.square(1 - i / capacity) * ffspeed for i in x]
    else:
        return np.square(1 - x / capacity) * ffspeed

@njit(parallel=True)
def estimated_TT(all_time_matrix, time_list, car_number, _Accumulation, dist, ffspeed):
    user_number = all_time_matrix.shape[0]
    departure_steps = all_time_matrix.shape[1]
    user_in_the_network = car_number
    Accumulation = _Accumulation
    new_time_ls = time_list # store everyone's last-day departure and arrival time as 1-d list
    actual_TT_tmp = np.zeros((user_number, departure_steps)) # predict the travel time in each possible departure time
    for user in prange(user_number):
        for t in prange(departure_steps): 
            start_time = all_time_matrix[user, t]
            known_ls = new_time_ls[new_time_ls > start_time]  
            # if (user == 799):
            #     print(" t ", t)
            #     print(" known_ls ", known_ls)
            #     print(" user_in_the_network ", user_in_the_network)
            if len(known_ls) == 0: # the fictional departure happens after all travelers
                texp = dist/ffspeed * 60
            elif len(known_ls) == user_in_the_network * 2:  # the fictional departure happens before all travelers
                texp = 0
                count = 0
                left_len = dist- ffspeed /60 * (known_ls[0] - start_time) # compute the left trip length till the first real traveler enter 
                if left_len < 0: # this fictional traveler end his trip before the first real traveler enter the network
                    texp = dist/ffspeed * 60
                else: # compute the travel speed between 2 consecutive events
                    V_list = np.array([V1(x) for x in Accumulation[user_in_the_network * 2 - len(known_ls): -1]])
                    # trip length traveled in each time interval between two consecutive events
                    len_piece = np.diff(known_ls) * V_list / 60
                    cum_len = np.cumsum(len_piece)
                    count = np.sum(cum_len < left_len)
                    texp = known_ls[count + 1] - start_time  \
                            + (left_len - cum_len[count]) / V1(Accumulation[count]) * 60
            else: # fictional departure happens after some real travelers have entered the network
                texp = 0 
                count = 0
                left_len = dist - V1(Accumulation[user_in_the_network * 2 - len(known_ls) - 1]) / 60 * (known_ls[0] - start_time) 
                
                if left_len < 0:  # if this fictional traveler end his trip before the next real event occurs
                    texp = dist / V1(Accumulation[user_in_the_network * 2 - len(known_ls) - 1]) * 60
                    print(" case 1 ", texp)
                else:
                    # travel speed in each time interval between two consecutive events
                    V_list = np.array(
                        [V1(x) for x in Accumulation[user_in_the_network * 2 - len(known_ls): -1]])
                    len_piece = np.diff(known_ls) * V_list / 60
                    cum_len = np.cumsum(len_piece)
                    count = np.sum(cum_len < left_len)
                    if count == 0:
                        texp = known_ls[count] - start_time  \
                                + (left_len - (known_ls[count] - start_time) * V1(1)/60) / ffspeed * 60
                    # this fictional traveler's is not finished even after all real
                    # travelers finish their trips
                    elif count == len(cum_len):
                        texp = known_ls[count] - start_time + \
                            (left_len - cum_len[count - 1]) / ffspeed * 60
                        # print(" case 2 ", texp)
                    else:  # this fictional traveler finishes the trip before all real travelers finish their trips
                        texp = known_ls[count + 1] - start_time  \
                                + (left_len - cum_len[count]) / V1(Accumulation[user_in_the_network * 2 - len(known_ls) + count]) * 60	
                # print(" texp 3 ", texp)
            actual_TT_tmp[user, t] = texp
    return actual_TT_tmp
    
    
class Travelers():
    # user parameters
    # user accounts
    # predicted departure times
    # update trip intentions
    # wihtin day mobility decisions
    # sell and buy
    # compute user account 
    # distance is asummed to be 16miles
    def __init__(self,
                 _numOfusers,
                 _user_params,
                 _allowance,
                 _allocation,
                 _scenario,
                 _hoursInA_Day = 12,
                 _Tstep = 1,
                 _fftt=24, 
                 _dist=18, 
                 _choiceInterval = 60, 
                 _seed=333,
                 _unusual=False,
                 _CV=True,
                 _numOfdays=50, 
                 input_save_dir = "tmp/", 
                ):

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
        # self.ptspeed = 25
        self.ptspeed = 18
        # self.ptspeed = 9

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
        self.norm_list = []
        if self.CV:
            self.NT_sysutil = np.load("./output/MFD/NT/NT_sysutil.npy")
        self.userCV = np.zeros(self.numOfusers) # calculate the user's willingness to pay for the shared ride based on their cost variation (CV) measure.
        self.input_save_dir = input_save_dir
        # initialize user accounts
        self.userAccounts = np.zeros(self.numOfusers)+self.AR*self.hoursInA_Day*60
        self.distribution = np.zeros(self.numOfusers) # modify the allowance distribution for each user based on the defined policy.
        self.toll_parameter = np.array([0,  0.1921139*120+420, 10*0.31822232+60])

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
        samps_array = np.array(samps).astype(int)
        return samps_array
    
     # calculate the utility and make choice based on logit model
    def calculate_utility(self, _x,_FW, _ps, _price, _pb, _day, _RR, _toll, _begin_time):
        price = _price
        x = _x # user account
        FW = _FW
        ps = _ps
        pb = _pb
        toll = _toll
        RR = _RR
        day = _day
        begin_time = _begin_time
        origtoll = _toll
        fuelcost = self.dist/self.mpg*self.fuelprice

        for user in self.users:
            s1 = self.DepartureLowerBD[user]
            s2 = self.DepartureUpperBD[user]
            tstar = self.desiredArrival[user]
            possibleDepartureTimes = np.arange(s1, s2 + 1, self.Tstep)
            possibleDepartureTimes = (possibleDepartureTimes/self.Tstep).astype(int)
            vot = self.vot[user] # value of time
            sde = self.sde[user] # sde constant
            sdl = self.sdl[user] # sdl constant
            vow = self.waiting[user] # value of waiting time
            I = self.I[user]  # income
            prev_allowance = self.distribution[user]

            # handle budget constraint
            if self.ARway == 'continuous':
                R = np.where(x[user,:]>=origtoll, (FW-origtoll)*ps-self.FTCs , (FW-x[user,:])*ps-self.FTCs-((origtoll-x[user,:])*pb+self.FTCb))
                overBudget = np.where(-R+fuelcost>I/2+prev_allowance)[0]
            else:
                overBudget = np.where((origtoll-self.userAccounts[user])*price+fuelcost>I/2+prev_allowance)[0]
            
            possibleDepartureTimes =  self.all_time_matrix[user, :]
            possibleDepartureTimes = (possibleDepartureTimes/self.Tstep).astype(int)
            mask = (np.zeros(len(possibleDepartureTimes))[:]+1).astype(int)
            if len(overBudget)>0:
                overBudget = (overBudget/self.Tstep).astype(int)	
                originlen = len(possibleDepartureTimes)
                mask  = np.in1d(possibleDepartureTimes,(overBudget/self.Tstep), invert = True) # if it is false: overbudgeted departure time
                possibleDepartureTimes = possibleDepartureTimes[~np.in1d(possibleDepartureTimes,(overBudget/self.Tstep))]
            
                if len(possibleDepartureTimes) < originlen and len(possibleDepartureTimes)>0:
                    self.numOfbindingI += 1
                    self.bindingI.append(user)

                if len(possibleDepartureTimes) == 0 : 
                    print(" why \n")
                    exit(1)
                    firstOne = overBudget[0]
                    lastOne = overBudget[-1]
                    shift = s2-firstOne+1
                    s1 = max(s1-shift,0)
                    s2 = min(firstOne-1, self.hoursInA_Day*60)
                    possibleDepartureTimes =  np.arange(s1, s2 + 1, self.Tstep)
                    possibleDepartureTimes = (possibleDepartureTimes/self.Tstep).astype(int) # round into time interval
                    # TODO: change the mask funciton below, since the possibleDepartureTimes range has been changed
                    mask =  np.in1d(self.all_time_matrix[user, :], possibleDepartureTimes) 
                    print(" possibleDepartureTimes ", possibleDepartureTimes)
                    print(" self.all_time_matrix[user, :] ", self.all_time_matrix[user, :])
                    print(" firstOne ", firstOne)
                    print(" s1 ", s1)
                    print(" s2 ", s2)
                    print(" mask 2 ", mask)

            utileps = np.zeros(1+len(possibleDepartureTimes)) # utility of each alternative (choice) for a particular user. It is a numpy array that stores the utility values corresponding to different departure time alternatives.
            utileps[0] = self.user_eps[user, -1] # add random term of no travel
            utileps[1:] = self.user_eps[user, possibleDepartureTimes] 
            
            tautilde = np.zeros(1+len(possibleDepartureTimes)) # estimated travel time
            tautilde[0] = self.pttt
            A = self.toll_parameter[0]
            A_floor = math.floor(A)
            if int(A_floor)==7:
                A_floor = 6
            # tautilde[1:] =self.predictedTT[user, mask]
            tautilde[1:] =self.predictedTT[A_floor, user, mask]

            Th = np.zeros(1+len(possibleDepartureTimes)) # Current toll fees
            Th[0] = self.ptfare
            Th[1:] = toll[possibleDepartureTimes] 
            if self.ARway == "continuous":
                possibleAB = x[user, possibleDepartureTimes] #possible account balance
            
            SDE = np.zeros(1+len(possibleDepartureTimes))
            SDE[1:] = np.maximum(0,tstar-(possibleDepartureTimes+tautilde[1:]+self.Tstep/2)) # use middle point of time interval to calculate SDE
            
            SDL = np.zeros(1+len(possibleDepartureTimes))
            SDL[1:] = np.maximum(0,(possibleDepartureTimes+tautilde[1:]+self.Tstep/2)-tstar)# use middle point of time interval to calculate SDL
           
            ASC = np.zeros(len(utileps)) # Alternative Specific Constant." It represents a constant term in a discrete choice model that captures the systematic factors affecting the utility of an alternative (choice)
            
            W = np.zeros(1+len(possibleDepartureTimes)) # waiting time
            W[0] = 1/2*self.ptheadway

            sysutil_t = ASC+(-2*vot*tautilde - sde * SDE - sdl * SDL-2*vow*W) # for all day, double travel time but not sde and sdl
        
            ch = np.zeros(len(utileps))	#the expected cost, equals to opportunity cost plus operation cosT
            if self.scenario == 'Trinity':
                if self.ARway == 'lumpsum':
                    buy = np.maximum(Th-self.userAccounts[user], 0)*price
                    sell = np.maximum(self.userAccounts[user] - Th, 0)*price
                    ch[:] = buy-sell
                    ch[0] = Th[0]-np.maximum(self.userAccounts[user], 0)*price
                elif self.ARway == 'continuous':
                    possibleAB = x[user, possibleDepartureTimes] #possible account balance
                    # calculate opportunity cost
                    tempTh = Th[1:] # exclude the first element corresponding to use PT
                    if self.decaying:
                        ch[1:] = -np.where(possibleAB>=tempTh, (FW-tempTh)*ps-(ps*self.AR*((FW-tempTh)/self.AR)**2/(2*self.hoursInA_Day*60))-self.FTCs,
                                    np.maximum(-(tempTh-possibleAB)*pb-self.FTCb+(FW-possibleAB)*ps-(ps*self.AR*((FW-possibleAB)/self.AR)**2/(2*self.hoursInA_Day*60))-self.FTCs, 0))
                        ch[0] = Th[0]-np.maximum((FW)*ps-(ps*self.AR*((FW)/self.AR)**2/(2*self.hoursInA_Day*60))-self.FTCs, 0)
                    else:
                        ch[1:] = -np.where(possibleAB>=tempTh, 
                                           (FW-tempTh)*ps-self.FTCs , 
                                           (FW-possibleAB)*ps-self.FTCs-((tempTh-possibleAB)*pb+self.FTCb))
                        ch[0] = Th[0]-np.maximum((FW)*ps-self.FTCs,0)			
            else:
                buy = Th*price
                sell = np.zeros_like(Th)
                ch = buy-sell-prev_allowance
        
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

            sysutil = (sysutil_t + self.user_params['lambda']*np.log(self.user_params['gamma'] + I-ch) + I-ch)
            util = sysutil + utileps
            if self.CV:
                if day == self.numOfdays - 1:
                    self.userCV[user] =  self.calculate_swcv(self.mu[user], [0], max(int(500/self.mu[user]), 500),
                        len(possibleDepartureTimes)+1, self.NT_sysutil[user,:], sysutil_t, ch, self.I[user])
                else:
                    self.userCV[user] = 0
    
            if np.argmax(util) == 0: # choose no travel
                self.ptshare.append(user)
                self.predayDeparture[user] = -1
            else:
                departuretime = possibleDepartureTimes[np.argmax(util)-1] + begin_time
                # np.random.seed(333)
                departuretime = int(np.random.choice(np.arange(departuretime, 
                                                               departuretime+self.Tstep), 
                                                               1)[0])
                self.predayDeparture[user] = departuretime
            self.predayEps[user] = utileps[np.argmax(util)]
        

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
        # np.random.seed(seed=self.seed)
        self.betant = 20        
        annualincome = np.load(self.input_save_dir +"annualincome.npy")
        self.vot = np.load(self.input_save_dir +"vot.npy")
        self.sde = np.load(self.input_save_dir +"sde.npy")
        self.sdl = np.load(self.input_save_dir +"sdl.npy")
        self.desiredArrival = np.load(self.input_save_dir +"desiredArrival.npy")
        self.user_eps = np.load(self.input_save_dir +"user_eps.npy")
        self.I = np.load(self.input_save_dir +"I.npy")

        if self.user_params['hetero'] != 1.6:
            newvot = self.newvot(cov = self.user_params['hetero'])
            self.vot[np.argsort(annualincome)] = newvot[np.argsort(newvot)]
       
        self.waiting = self.vot*3

        # generate predicted travel
        if self.unusual['read'] and self.unusual['unusual']:
            self.predictedTT = np.load(self.unusual['read'])
            self.actualTT =  np.load(self.unusual['read'])
        else:
            self.predictedTT = self.fftt * np.ones((7, self.numOfusers, 2 * self.choiceInterval + 1))
            self.actualTT = self.fftt * np.ones((self.numOfusers, 2 * self.choiceInterval + 1))

        self.desiredDeparture = self.desiredArrival-self.fftt # generate desired departure time: user_len 1-d array 
        self.DepartureLowerBD = self.desiredDeparture-self.choiceInterval
        self.DepartureUpperBD = self.desiredDeparture+self.choiceInterval
        
        y = np.repeat([self.desiredDeparture], 2*self.choiceInterval+1, axis=1)
        z = y.reshape(-1,2*self.choiceInterval+1)
        x = np.repeat([range(-self.choiceInterval, self.choiceInterval+1)], self.numOfusers, axis=0)
        # print("x.shape ", x.shape )
        # print("z.shape ", z.shape )
        self.all_time_matrix = z+x # desired departure time with choice interval 
        
        # initialize predayDeparture:  store the chosen departure time for each user on the previous day.
        self.predayDeparture = np.zeros(self.numOfusers, dtype=int) + self.desiredDeparture


    # Make preday choice according to attributes (travel time, schedule delay, toll)
    # PT user: suppose all tw windows are taken by cars
    # Car user: suppose N-1 tw windows with 1 real travel experience time 
    def update_choice_MFD(self, _toll, _price, _day, _RR, _begin_time):
        self.ptshare = [] # add user who take bus
        self.bindingI = []
        self.predayEps = np.zeros(self.numOfusers) # stores the utility values corresponding to the chosen departure time alternative for each user in the pre-day choice
        self.numOfbindingI = 0
        toll = _toll
        price = _price
        day = _day
        RR = _RR
        begin_time = _begin_time
        pb = price*(1+self.PTCb) # selling price
        ps = price*(1-self.PTCs) # buying price

        # MFD simulation attributes
        # TODO: change heterogenous trip length
        trip_len = np.zeros(self.numOfusers)
        trip_len[:] = self.dist

        if day != 0: # if it is not the first day, make choice based on history; else depart at desired departure time
            _toll = np.mean(np.array(_toll).reshape(-1, self.Tstep), axis=1)  # get the toll for each Tstep
            x = np.zeros((self.numOfusers, self.hoursInA_Day*60)) # token account balance
            FW = 0
            if self.ARway == 'continuous':
                # predict today's account balances
                FW = self.AR*self.hoursInA_Day*60 # full wallet
                x[:, 0] = self.userAccounts # equals to initial token balance
                td = np.where(self.predayDeparture !=-1, np.mod(self.predayDeparture, self.hoursInA_Day*60), -self.hoursInA_Day*60)
                Td = np.where(td != -self.hoursInA_Day*60, _toll[td-td%self.Tstep], td) # toll fees
                for t in range(self.hoursInA_Day*60-1): # calculate the profit at time t
                    td = np.where(td != -self.hoursInA_Day*60, np.where(td<t, td+self.hoursInA_Day*60, td), td) 
                    profitAtT = np.zeros(self.numOfusers) #user profit at time t
                    mask_cansell = td!=t
                    FA = np.where(td!=-self.hoursInA_Day*60, np.minimum((td-t)*self.AR, FW), 0)
                    if self.decaying:
                        profitAtT = x[:, t]*ps-(ps*(x[:, t])**2/(2*self.hoursInA_Day*60*self.AR))-self.FTCs-np.where(Td>FA,(Td-FA)*pb+self.FTCb,0)
                    else:
                        profitAtT = x[:, t]*ps-self.FTCs-np.where(Td>FA,(Td-FA)*pb+self.FTCb,0)
                    profitAtT[~mask_cansell] = 0.0
                    mask_positiveprofit = profitAtT>1e-10
                    mask_needbuy = Td>=FA
                    mask_needbuynext = Td>=np.maximum(FA-self.AR, 0)
                    mask_FW = np.abs(x[:, t]-FW)<1e-10
                    mask_sellnow = (mask_cansell & mask_positiveprofit) & (mask_needbuy | mask_FW | mask_needbuynext)
                    x[mask_sellnow, t] = 0
                    x[:, t+1] = np.maximum(FW, x[:, t]+self.AR)
            self.calculate_utility(x, FW, ps, price, pb, day, RR, toll, begin_time)
    # end of update_choice_MFD

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
    

    # get 0-1 index for all users, indicating their selling and buying behavior
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
    
    
    
    # compute future user accounts:
    def update_account(self):
        return

    # actual arrival is the combination of estimated arrival(if he didn't those time points) 
    # and real arrival time(if he chooses this time point)
    def update_arrival(self, actualArrival):
        self.actualArrival = actualArrival

    # perform day to day learning
    def d2d(self, day):
        A = self.toll_parameter[0]
        # print("A: ", str(A))
        A_floor = math.floor(A)
        if int(A_floor)==7:
            A_floor = 6
        self.predictedTT[A_floor, :, :] = 0.9*self.predictedTT[A_floor, :, :] + 0.1*self.actualTT[:] 
        
        
        # np.save("new_perceived_method/predictedTT_"+str(day)+".npy", self.predictedTT)     
        # np.save("new_perceived_method/actualTT_"+str(day)+".npy", self.actualTT)     

        # c_perceived =  self.predictedTT
        # c_cs = self.actualTT
        # self.norm_list.append(np.linalg.norm(c_perceived-c_cs,ord=1)/self.numOfusers)


class Regulator():
   # regulator account balance
    def __init__(self, marketPrice=1, RBTD = 100, deltaP = 0.05):
        self.RR = 0
        self.tollCollected = 0
        self.allowanceDistributed = 0
        self.marketPrice = marketPrice
        self.RBTD = RBTD  # a constant threshold
        self.deltaP = deltaP

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
        # self.marketPrice = 1

class MFD_simulation():
    # simulate one day

    def __init__(self,
                 _numOfdays=50,
                  _user_params = None,
                  _scenario='NT',
                  _allowance=False,
                  _marketPrice = 1,
                  _allocation = None,
                  _deltaP = 0.05,
                  _numOfusers=7500,
                  _RBTD = 100, 
                  _Tstep=1,
                  _Plot = False,
                  _verbose = False, 
                  _unusual=False,
                  _storeTT=False, 
                  _CV=True, 
                  save_dfname='CPresult.csv', 
                  toll_type="normal", 
                  _hoursInA_Day=12,
                  _choiceInterval=60,                  
                  _input_save_dir = "tmp",
                  _fftt  = 24, 
                  _seed = 333,
                ):
        self.numOfdays = _numOfdays
        self.hoursInA_Day = _hoursInA_Day
        self.numOfusers = _numOfusers
        self.allowance = _allowance
        self.save_dfname = save_dfname
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

        self.choiceInterval = _choiceInterval
        self.toll_type = toll_type
        self.input_save_dir = _input_save_dir
            
        self.toll_parameter = np.array([0,  0.1921139*120+420, 10*0.31822232+60])

        self.users = Travelers(self.numOfusers,
                               _user_params = self.user_params,
                               _allocation = _allocation,
                               _fftt = _fftt,
                               _hoursInA_Day=_hoursInA_Day,
                               _Tstep=self.Tstep,
                               _allowance=self.allowance,
                               _scenario = self.scenario,
                               _seed=_seed,
                               _unusual=self.unusual,
                               _CV = _CV,
                               _numOfdays = _numOfdays,
                               _choiceInterval = self.choiceInterval,
                               input_save_dir= self.input_save_dir,
                            )
        self.users.generate_params()
        self.regulator = Regulator(_marketPrice, _RBTD, _deltaP)
        self.originalAtt = {}
        self.presellAfterdep = np.zeros(self.numOfusers,dtype=int)
        timeofday = np.arange(self.hoursInA_Day*60)
        self.toll = np.array([0]*len(timeofday)) # store the toll fees for 720 min
        self.capacity = capacity
        self.toll_type = toll_type

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
    def bimodal(self, x):
        A = self.toll_parameter[0]
        mu = self.toll_parameter[1]
        sigma = self.toll_parameter[2]
        toll_profile = A*np.exp(-(x-mu)**2/2/(sigma)**2)
        return toll_profile

    # MFD simulation
    def MFD(self, day):
        # MFD simulation attributes
        trip_len = np.zeros(self.numOfusers)
        trip_len[:] = self.users.dist
        # create a dict to store the information of each agent
        vehicle_information = {}
        vehicle_information['vehicle'] = np.arange(self.numOfusers)
        vehicle_information['trip_len(m)'] =trip_len[:].astype(np.float64) #left length
        vehicle_information['t_exp'] = np.zeros(self.numOfusers) # experienced time
        vehicle_information['account'] = np.zeros(self.numOfusers) 

        t_ls = []  # record the event time which has not been removed
        Accumulation = [] # event based accumulation record

        n = 0  # Number of vehicle (accumulation)
        j = 0  # index of event
        vehicle_index = [] # vehicle in the event
        Departure_time  = self.users.predayDeparture # departure time
        Arrival_time  = np.where(Departure_time > 0, Departure_time+self.fftt, -1)
    
        # find the car users
        car_index = np.where(Departure_time>0)[0]
        pt_index = np.where(Departure_time<0)[0]
        car_number = car_index.shape[0]
        pt_number = pt_index.shape[0]

        # Define event list of departures
        Event_list1_array = np.zeros((car_number, 4))
        Event_list1_array[:, 0] = car_index
        Event_list1_array[:, 1] = Departure_time[car_index] 
        Event_list1_array[:, 2] = np.ones(car_number) 
        Event_list1_array[:, 3] = trip_len[car_index]

        # Define event list of arrivals
        Event_list2_array =  np.zeros((car_number, 4))
        Event_list2_array[:, 0] = car_index
        Event_list2_array[:, 1] = Arrival_time[car_index]   # time(min)
        Event_list2_array[:, 2] = np.ones(car_number) * 2  # arrival indicator: 2
        Event_list2_array[:, 3] = trip_len[car_index]  # trip length
                
        # S_Event_list_array: 4 columns
        # vehicle_index  time(min)  event_indicator  trip_len
        S_Event_list_array = np.concatenate(
            (Event_list1_array, Event_list2_array), axis=0)
        # Sort the list by time in ascending order
        S_Event_list_array = S_Event_list_array[S_Event_list_array[:, 1].argsort()]
        # get time of the first event
        t_ls.append(S_Event_list_array[0, 1])  # initial time

        while S_Event_list_array.shape[0] > 0: 

            j = j + 1
            t_ls.append(S_Event_list_array[0, 1])
            Event_index = int(S_Event_list_array[0, 0])
            Event_type =  int(S_Event_list_array[0, 2])
            # print("j ", j)
            if Event_type == 1: # if it is departure event
                
                # add the vehicle index which has entered the network 
                vehicle_index.append(Event_index)

                # update the left trip length for cars which have departured before 
                trip_len1 = vehicle_information['trip_len(m)']
                trip_len1[vehicle_index[0:-1]] = trip_len1[vehicle_index[0:-1]] - V(n) / 60 * (t_ls[j] - t_ls[j - 1])
                vehicle_information['trip_len(m)'] = trip_len1
        
                # update the accumulation in the network
                n = n + 1
                
                # keep track of the accumulation
                Accumulation.append(n)

                # update the predicted arrival time for all cars which has entered the network
                travel_started_vehicles = np.where((S_Event_list_array[:, 2] == 2) & 
                                                (np.isin(S_Event_list_array[:, 0], vehicle_index)))
                temp = S_Event_list_array[(travel_started_vehicles)][:, 0] # get the vehicle index where the travel has been started
                if np.size(temp) == 0:
                    temp = np.array([])
                S_Event_list_array[(travel_started_vehicles), 1] \
                    = t_ls[j] + vehicle_information['trip_len(m)'][temp.astype(int)] /  V(n) * 60 

            else: # if it is an arrival event
                # update the trip lenth which has entered the network 
                trip_len1 = vehicle_information['trip_len(m)']
                trip_len1[vehicle_index] = trip_len1[vehicle_index] -  V(n) / 60 * (t_ls[j] - t_ls[j - 1]) 
                vehicle_information['trip_len(m)'] = trip_len1
                n = n-1 

                # keep track of the accumulation
                Accumulation.append(n)

                # update t_exp
                vehicle_information['t_exp'][Event_index]\
                    = S_Event_list_array[0, 1] - Departure_time[Event_index] # actual experienced travel time

                # remove the agent that finishes the trip
                vehicle_index.remove(Event_index)

                # Update the predicted arrival time
                travel_started_vehicles = np.where((S_Event_list_array[:, 2] == 2) & (
                    np.isin(S_Event_list_array[:, 0], vehicle_index)))
                temp = S_Event_list_array[(travel_started_vehicles)][:, 0]
                if np.size(temp) == 0:
                    temp = np.array([])
                S_Event_list_array[(travel_started_vehicles), 1]\
                    = t_ls[j] + vehicle_information['trip_len(m)'][temp.astype(int)] /  V(n) * 60

            # remove event from the list
            S_Event_list_array = np.delete(S_Event_list_array, (0), axis=0)
            S_Event_list_array = S_Event_list_array[S_Event_list_array[:, 1].argsort()]           

        vehicle_information['t_dep'] = Departure_time[:]
        vehicle_information['t_arr'] = np.where(vehicle_information["t_dep"]> 0, vehicle_information["t_dep"] + vehicle_information['t_exp'], -1)
        vehicle_information['t_exp'] = np.where(vehicle_information["t_dep"]> 0, vehicle_information['t_exp'], self.users.pttt)

        time_list = np.concatenate((
            vehicle_information['t_dep'],
            vehicle_information['t_arr']), 
            axis=0)
        time_list_2 = time_list[np.where(time_list>0)]-(day)*720
        time_list_2 = np.sort(time_list_2, axis=None)
        
        actual_TT_tmp = estimated_TT(self.users.all_time_matrix, time_list_2, car_number, Accumulation, self.users.dist, self.users.ffspeed)
        self.users.actualTT[:] =  actual_TT_tmp[:]

        return vehicle_information, time_list_2, Accumulation, car_number
        # end of MFD

    def RL_simulateOneday(self, day, state_aggravate, state_shape):
        self.currday = day
        beginTime = day*self.hoursInA_Day*60
        totTime = (day+1)*self.hoursInA_Day*60
        self.users.update_choice_MFD(self.toll, 
                                     self.regulator.marketPrice, 
                                     self.currday, 
                                     self.regulator.tollCollected, 
                                     beginTime)
        self.numOfundesiredTrading = np.zeros(self.numOfusers)
        sellTime = np.zeros(self.numOfusers)
         
        actualArrival = np.zeros(self.numOfusers) # traveler's arrival time
        userSell = np.zeros(self.numOfusers) # user who sell tokens
        userBuy = np.zeros(self.numOfusers) 
        userBuytc = np.zeros(self.numOfusers) # sell amount
        userSelltc = np.zeros(self.numOfusers)
        userToll = np.zeros(self.numOfusers)
        buyvec = np.zeros(totTime-beginTime)
        sellvec = np.zeros(totTime-beginTime)
        buyamount = np.zeros(totTime-beginTime) # average buy token amount
        sellamount = np.zeros(totTime-beginTime) # average sell token amount

        average_tt = np.zeros(totTime-beginTime)
        average_accumulation = np.zeros(totTime-beginTime)

        if self.unusual['unusual'] and self.currday == self.unusual['day']:
            self.originalAtt['price'] = self.regulator.marketPrice
            self.originalAtt['FTCb'] = self.users.FTCb
            self.originalAtt['FTCs'] = self.users.FTCs
            self.originalAtt['PTCb'] = self.users.PTCb
            self.originalAtt['PTCs'] = self.users.PTCs
            self.originalAtt['AR'] = self.users.AR

        vehicle_information, time_list, Accumulation, car_number = self.MFD(day)		
        actualArrival = vehicle_information["t_arr"] #update actual travel time
        travel_time =  vehicle_information["t_exp"] # daily travel time without PT
        Accumulation = np.array(Accumulation)

        for t in range(beginTime, totTime):
            tmp = np.mod(t-t%self.Tstep, self.hoursInA_Day*60)
            currToll = self.toll[tmp]
            start_t = tmp
            finish_t = tmp + self.Tstep 
            current_event_index = np.where((time_list>=start_t) & (time_list<finish_t))
            if np.any(current_event_index[0]) :
                mean_acc = np.mean(Accumulation[current_event_index]) # get the average mean accumulation in this time interval
            else :
                mean_acc = 0 
            average_accumulation[tmp] = mean_acc

            departure_time = vehicle_information["t_dep"] - day*720
            travel_time = vehicle_information["t_exp"]

            # has filtered pt transit
            departured_car_index = np.where((departure_time>=start_t) & (departure_time<finish_t))
            if np.any(departured_car_index[0]):
                mean_travel_time = np.mean(travel_time[departured_car_index])# get the average mean accumulation in this time interval
            else:
                mean_travel_time = self.fftt
            average_tt[tmp] = mean_travel_time # get the average mean travel time
 
            if self.scenario == 'Trinity':
                tempuserBuy, tempuserSell = self.users.sell_and_buy(t, currToll, self.toll, self.regulator.marketPrice, totTime)
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
                    userSell += np.where(tempuserSell>1e-6, tempuserSell*self.regulator.marketPrice*(1),0)
                    userSelltc += np.where(tempuserSell>1e-6, tempuserSell*self.regulator.marketPrice*(-self.PTCs)-self.FTCs, 0)

        self.usertrade_array[self.currday,:, 0] = buyvec
        self.usertrade_array[self.currday,:, 1] = sellvec

        self.tokentrade_array[self.currday,:, 0] = sellamount
        self.tokentrade_array[self.currday,:, 1] = buyamount

        self.users.update_arrival(actualArrival)
 
        self.flow_array[self.currday, :, 0]  =  np.maximum(self.users.predayDeparture-beginTime,-1)
        self.flow_array[self.currday, :, 1]  =  np.maximum(actualArrival-beginTime, -1)
        self.flow_array[self.currday, :, 2]  =  np.where(self.flow_array[self.currday, :, 0]!=-1, 
                                                         self.flow_array[self.currday, :, 1] -  self.flow_array[self.currday, :, 0], 
                                                         self.users.pttt)
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
        sw, tt_util, sde_util, sdl_util, ptwaiting_util, I_util, userBuy_util, userSell_util, fuelcost_util = self.calculate_sw()

        if self.unusual['unusual'] and self.currday == self.unusual['day']:
            self.users.d2d(self.currday)
            self.regulator.marketPrice = self.originalAtt['price']
        else:
            # d2d learnining
            self.users.d2d(self.currday)
            if self.scenario == 'Trinity':
                # update token price
                self.regulator.update_price()	

        self.presellAfterdep = sellTime>self.users.predayDeparture
        
        tt_state = np.zeros(shape =144)
        accumulation_state = np.zeros(shape =144)
        sell_state = np.zeros(shape =144)        
        buy_state = np.zeros(shape =144)

        for j in range(144):
            tt_state[j] = np.mean(average_tt[j*5:(j+1)*5])  
            accumulation_state[j] = np.mean(average_accumulation[j*5:(j+1)*5])  
            sell_state[j] = np.mean(sellvec[j*5:(j+1)*5])  
            buy_state[j] = np.mean(buyvec[j*5:(j+1)*5])  

        # if space_shape[0] == 5:
        #     state_ls = [numDepart, actualTT, buyamount, sellamount, self.toll]
        #     state = np.concatenate(state_ls)
        #     # aggravate the state from 1 min to 5 min
        #     encode_shape = int(self.hoursInA_Day*60/state_aggravate)
        #     state_encode = np.zeros(shape = encode_shape * len(state_ls))

        #     for j in range(encode_shape):
        #         state_encode[j] = np.mean(numDepart[j*state_aggravate:(j+1)*state_aggravate])
        #         state_encode[j+1*encode_shape] = np.mean(actualTT[j*state_aggravate:(j+1)*state_aggravate])
        #         state_encode[j+2*encode_shape] = np.mean(sellvec[j*state_aggravate:(j+1)*state_aggravate])
        #         state_encode[j+3*encode_shape] = np.mean(buyvec[j*state_aggravate:(j+1)*state_aggravate])
        #         state_encode[j+4*encode_shape] = np.mean(self.toll[j*state_aggravate:(j+1)*state_aggravate])
        
        # elif space_shape[0] == 4:
        #     # TODO: average_accumulation, average_tt
        #     state_ls = [numDepart, buyamount, sellamount, self.toll]
        #     state = np.concatenate(state_ls)
        #     # aggravate the state from 1 min to 5 min
        #     encode_shape = int(self.hoursInA_Day*60/state_aggravate)
        #     state_encode = np.zeros(shape = encode_shape * len(state_ls))
        #     for j in range(encode_shape):
        #         state_encode[j] = np.mean(numDepart[j*state_aggravate:(j+1)*state_aggravate])
        #         state_encode[j+1*encode_shape] = np.mean(sellvec[j*state_aggravate:(j+1)*state_aggravate])
        #         state_encode[j+2*encode_shape] = np.mean(buyvec[j*state_aggravate:(j+1)*state_aggravate])
        #         state_encode[j+3*encode_shape] = np.mean(self.toll[j*state_aggravate:(j+1)*state_aggravate])
        
        # elif space_shape[0] == 1:
        #     state_ls = [self.toll]
        #     state = np.concatenate(state_ls)
        #     # aggravate the state from 1 min to 5 min
        #     encode_shape = int(self.hoursInA_Day*60/state_aggravate)
        #     state_encode = np.zeros(shape = encode_shape * len(state_ls))
        #     for j in range(encode_shape):
        #         state_encode[j] = np.mean(numDepart[j*state_aggravate:(j+1)*state_aggravate])
        
        # else:
        #     print(" not this type of state_shape")
        #     exit(1)
        return tt_state, accumulation_state, sell_state, buy_state, pt_share_number, market_price, sw, tt_util, sde_util, sdl_util, ptwaiting_util, I_util, userBuy_util, userSell_util, fuelcost_util
    
    # calculate social welfare value
    def calculate_sw(self):
        TT = np.where(self.users.predayDeparture!=-1, self.users.actualArrival-self.users.predayDeparture, self.users.pttt) 
        # print(" self.users.actualArrival ",  self.users.actualArrival)
        # print(" self.users.predayDeparture ", self.users.predayDeparture)
        SDE = np.where(self.users.predayDeparture!=-1, np.maximum(0, self.users.desiredArrival+self.currday*self.hoursInA_Day*60-self.users.actualArrival), 0)
        SDL = np.where(self.users.predayDeparture!=-1, np.maximum(0, self.users.actualArrival-(self.users.desiredArrival+self.currday*self.hoursInA_Day*60)), 0)
        allowance = self.users.distribution
        # either car fuel cost or transit fare
        fuelcost = np.where(self.users.predayDeparture!=-1, self.users.dist/self.users.mpg*self.users.fuelprice, self.users.ptfare)
        ASC = np.zeros(self.numOfusers)
        ptwaitingtime = np.where(self.users.predayDeparture!=-1,0 ,self.users.ptheadway)
        
        util = ASC + (-2 * self.users.vot * TT - self.users.sde * SDE - self.users.sdl * SDL - self.users.waiting * ptwaitingtime
             + self.user_params['lambda'] * np.log(self.user_params['gamma'] + self.users.I - 2 * self.userBuy + 2 * self.userSell + 2 * allowance - 2 * fuelcost)
             + self.users.I - 2 * self.userBuy + 2 * self.userSell + 2 * allowance - 2 * fuelcost) + self.users.predayEps
        NTMUI = 1 + self.user_params['lambda']/(self.user_params['gamma'] + self.users.I)
        
        tt_util = np.sum(2 * self.users.vot * TT)
        sde_util = np.sum(self.users.sde * SDE)
        sdl_util = np.sum(self.users.sdl * SDL)
        ptwaiting_util = np.sum(self.users.waiting * ptwaitingtime)
        I_util = np.sum(self.users.I)
        userBuy_util =  np.sum(2 * self.userBuy)
        userSell_util = np.sum(2 * self.userSell )
        fuelcost_util = np.sum(2 * fuelcost)

        if self.scenario == 'NT':
            obj = np.sum(util)
        else:
            NT_util = np.load(self.input_save_dir+"NT_util.npy")
            # TODO: change the baseline toll
            userBenefits = (util - NT_util)
            obj = np.sum(userBenefits) + 2 * self.regulator.RR

        return obj, tt_util, sde_util, sdl_util, ptwaiting_util, I_util, userBuy_util, userSell_util, fuelcost_util
