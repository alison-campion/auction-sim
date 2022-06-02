import matplotlib.pyplot as plt
from scipy.stats import uniform 
from scipy.stats import norm
import scipy.integrate as integrate
from scipy.integrate import quad
import seaborn as sns
sns.set(color_codes = True)
sns.set(rc = {'figure.figsize':(5,5)})

# CDFs 
## Uniform Distribution 
def cdf_uniform(x,delta,n):
    return ((x/delta)**(n-1))

# Defining auction preliminaries
# number of players in the auction
n = 20

# Player valuations
## Assume uniformly distributed values in [0,100]
start = 0
width = 100

# Monte-Carlo etc
# Number of simulations
n_sim = 100000

# Seller revenue arrays
FP_revenue = []
SP_revenue = []

for sim in range(1,n_sim):

    data = uniform.rvs(size=n, loc = start, scale=width)
    
    # Player optimal bids 
    
    ## First Price Auction
    FP_optimal_bid = []
    for v in data:
        FP_bid = v - ((integrate.quad(cdf_uniform, 0, v, args = ((width-start),n))[0])/cdf_uniform(v, (width-start),n))
        FP_optimal_bid.append(FP_bid)
    
    ## Second Price Auction
    SP_optimal_bid = data
    
    # Winning bids 
    FP_winning_bid = max(FP_optimal_bid)
    SP_winning_bid = max(SP_optimal_bid)
    SP_optimal_bid.sort()
    
    # Seller Revenues
    FP_revenue.append(FP_winning_bid)
    SP_revenue.append(SP_optimal_bid[-2])

# Test of Revenue Equivalence Theorem - difference in the seller revenue
diff_revenue = []    
for FP_rev,SP_rev in zip(FP_revenue, SP_revenue):
    diff_revenue.append(FP_rev - SP_rev)    
sns.distplot(diff_revenue)    

