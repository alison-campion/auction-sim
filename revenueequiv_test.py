import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, beta
import scipy.integrate as integrate
from scipy.integrate import quad
import seaborn as sns
from tqdm import tqdm
import numpy as np
import random
sns.set(color_codes = True)
sns.set(rc = {'figure.figsize':(5,5)})

# CDFs 
## Uniform Distribution 
# def cdf_uniform(x,delta,n):
#     return ((x/delta)**(n-1))

# ## (Truncated) Normal Distribution
# def cdf_normal(x, mean, sd, n):
#     return(((norm(loc=mean, scale = sd).cdf(x))/(norm(loc=mean, scale = sd).cdf(0)))**(n-1))

def monte_carlo_integrator(dist, params, lower, upper, n):
    def f(x, lower, upper):
        if dist == 'uniform':
            return ((x-lower)/(upper-lower))**(n-1)
        elif dist == 'normal':
            return (norm.cdf(x, loc = params[0], scale = params[1]))**(n-1)
        elif dist == 'beta':
            return (beta.cdf(x, a = params[0], b = params[1], scale = params[2]))**(n-1)
        
    n = 1000
    ar = np.zeros(n)
    for i in range (len(ar)):
        ar[i] = random.uniform(lower,upper)
    
    integral = 0.0
    for i in ar:
        integral += f(i, lower, upper)
        
    ret = (upper-lower)/float(n)*integral
    return ret



# Seller revenue arrays
def val_simulator(dist, n_sim, n_bid, params):
    FP_revenue = []
    SP_revenue = []
    
    
    for _ in tqdm(range(n_sim), total = n_sim, desc = 'Simulating i.i.d. valutaion draws from a {} distribution:'.format(dist)):
    #    data = uniform.rvs(size=n, loc = start, scale=width)
        FP_optimal_bid = []
        if dist == 'uniform':
            vals = np.random.uniform(params[0], params[1], n_bid)
            ## First Price Auction optimal bida
            for v in vals:
                FP_bid = v - ((monte_carlo_integrator(dist, params, 0, v, n_bid)/((v-params[0])/(params[1]-params[0]))**(n_bid-1)))
                FP_optimal_bid.append(FP_bid)
                
        elif dist == 'normal':
            vals = np.random.normal(params[0], params[1], n_sim)
            ## First Price Auction optimal bid
            for v in vals:
                FP_bid = v - (monte_carlo_integrator(dist, params, 0, v, n_bid)/norm.cdf(v, loc = params[0], scale = params[1])**(n_bid-1))
                FP_optimal_bid.append(FP_bid)
        
        elif dist == 'beta':
            vals = np.random.beta(params[0], params[1], n_sim)
            ## First Price Auction optimal bid
            for v in vals:
                FP_bid = v - (monte_carlo_integrator(dist, params, 0, v, n_bid)/beta.cdf(v, a = params[0], b = params[1], scale = params[2])**(n_bid-1))
                FP_optimal_bid.append(FP_bid)
        FP_optimal_bid.sort()
        ## Second Price Auction optimal bid
        SP_optimal_bid = vals
        SP_optimal_bid.sort()
        
        # Seller Revenues
        FP_revenue.append(max(FP_optimal_bid))
        SP_revenue.append(SP_optimal_bid[-2])
    return FP_revenue, SP_revenue
    


def main():
    dist = input('Distribution of valuation:')
    n_bid = int(input('Numebr of Bidders:'))
    if dist.lower() == 'uniform' or dist.lower() in 'uniform':
        dist = 'uniform'
        lb = float(input('Lower Bound:'))
        ub = float(input('Upper Bound:'))
        params = [lb, ub]
        
    if dist.lower() == 'normal' or dist.lower() in 'normal':
        dist = 'normal'
        mean = float(input('Mean:'))
        sd = float(input('Std. Deviation:')) 
        params = [mean, sd]
    
    #beta parametrization found here: https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html
    if dist.lower() == 'beta':
        dist = 'beta'
        alpha = float(input('Alpha:'))
        beta = float(input('Beta:'))
        params = [alpha, beta]
        
    n_sim = int(input('Number of simulations:'))    
    
    FP_revenue, SP_revenue = val_simulator(dist, n_sim, n_bid, params)
    
    # Test of Revenue Equivalence Theorem - difference in the seller revenue
    diff_revenue = []    
    for FP_rev,SP_rev in zip(FP_revenue, SP_revenue):
        diff_revenue.append(FP_rev - SP_rev)
    sns.distplot(diff_revenue, label = 'Difference in FP and SP revenue') 
        
if __name__ == '__main__':
    main()