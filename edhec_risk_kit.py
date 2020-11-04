import pandas as pd
import scipy.stats
import numpy as np
from scipy.stats import norm
import os 
import pathlib
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math


def annualized_rets(r,periods_per_year):
    
    """
    Annualizes a set of returns. 
    We should infer the period per year
    """
    
    compound_growth = (1+r).prod()
    n_periods = r.shape[0]
    
    return compound_growth**(periods_per_year/n_periods)-1

def annualized_vol(r,periods_per_year):
    
    """
    Annualizes volatility of a set of returns.
    We should infer the periods per year.
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r,riskfree_rate,periods_per_year):
    
    """
    Computes the annualized Sharpe ratio of a set of returns
    """
    
    #convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r-rf_per_period
    ann_ex_ret = annualized_rets(excess_ret,periods_per_year)
    ann_vol = annualized_vol(r,periods_per_year)

    return ann_ex_ret/ann_vol







def wealth_index(returns: pd.Series, final_wealth= False, investment=1000):
    
    """
    Returns wealth indices as a DataFrame. If final_wealth is set to True,
    it returns a second DataFrame with the accumulated wealth in last period.
    Investment can be changed by adjusting parameter. It is set to 1000 dollarrs by default.
    """
    
    wealth_ind= investment*(1+returns).cumprod()
    
    if final_wealth == False:
       
        return wealth_ind

    elif final_wealth == True:
        return wealth_ind,pd.DataFrame({"Final wealth":wealth_ind.iloc[-1,:]})
        

def drawdown(return_series: pd.Series, investment=1000):  #if we do it like this, the function will expect a pd.Series
             """
             Takes a time series of asset returns.
             Computes and returns a DataFrame that contains:
             wealth index
             previous peaks
             percent drawdowns
             
             Investment can be changed by adjusting parameter. It is set to 1000 dollarrs by default.
             """
             
             wealth_index = investment*(1+return_series).cumprod()
             previous_peaks = wealth_index.cummax()
             drawdowns = (wealth_index-previous_peaks)/previous_peaks
             
             return pd.DataFrame({
                 "Wealth": wealth_index,
                 "Peaks": previous_peaks,
                 "Drawdown": drawdowns
             })
            
            

def get_csv(csv_file):
    
    """
    Finds a given CSV file in the computer and reads it. Function is pretty slow because it looks for the file in disk C.
    https://stackoverflow.com/questions/33823468/read-file-in-unknown-directory/33823685
    
    """
    
    for dirpath, dirnames, filenames in os.walk("C:\\"):
        for filename in filenames:
            if filename == csv_file:
                file = os.path.join(dirpath, filename)
                
    csv_read = pd.read_csv(file,
                    header=0, index_col=0, parse_dates=True, na_values=-99.99)
    
    return csv_read

def get_ffme_returns():
    
    """
    Load and format the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCAP
    
    
    """

                
    me_m = get_csv('Portfolios_Formed_on_ME_monthly_EW.csv')
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap','LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period('M')
    
    return rets

def get_hfi_returns():
    
    """
    Load and format EDHEC Hedge Fund Indices Data Set.
    """
       
    hfi = get_csv('edhec-hedgefundindices.csv')
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    
    return hfi

def get_ind_returns():
    
    """
    Load and format Ken French 30 Industry Portfolios Value Weighted Monthly Returns.
    """
       
    ind = get_csv('ind30_m_vw_rets.csv')
    ind = ind/100  #returns are given as a percentage, so we have to divide by 100
    ind.columns = ind.columns.str.strip() #gets rid of trailing spaces in some columns: "Food " -> "Food"
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    
    return ind


def get_ind_size():
    
    """
    Load and format Ken French 30 Industry Portfolios Value Size.
    """
       
    ind = get_csv('ind30_m_size.csv')
    ind.columns = ind.columns.str.strip() #gets rid of trailing spaces in some columns: "Food " -> "Food"
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    
    return ind

def get_ind_firms():
    
    """
    Load and format Ken French 30 Industry Portfolios Value Number of Firms.
    """
       
    ind = get_csv('ind30_m_nfirms.csv')
    ind.columns = ind.columns.str.strip() #gets rid of trailing spaces in some columns: "Food " -> "Food"
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    
    return ind


def get_total_market_index_returns():
    
    """
    Returns total market index returns fom Ken French 30 Industry Portfolios
    """
    
    ind_return = get_ind_returns()
    ind_size = get_ind_size()
    ind_nfirms = get_ind_firms()
    ind_mktcap = ind_nfirms * ind_size
    total_mktcap = ind_mktcap.sum(axis="columns") #we sum the market cap of all industries
    ind_capweight = ind_mktcap.divide(total_mktcap,axis="rows")
    
    return pd.DataFrame({"Market return":(ind_capweight*ind_return).sum(axis="columns")})

def skewness(r):
    
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series    
    The standard deviation calculated will be for the population, not a sample (we have a complete dataset),
    so we have to set dof=0:  https://numpy.org/doc/stable/reference/generated/numpy.std
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    
    """
    Alternative to scipy.stats.skew()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series    
    The standard deviation calculated will be for the population, not a sample (we have a complete dataset),
    so we have to set dof=0:  https://numpy.org/doc/stable/reference/generated/numpy.std
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r,level=0.01):
    
    jb,pvalue = scipy.stats.jarque_bera(r)
    
    return pvalue > level

    """
    Applies the Jarque-bera test to determine if a Series is normal or not.
    Test is applied at the 1% level by default.
    Returns True if the hypothesis of normality is accepted, False otherwise
    """

def jb_pvalue_normal(r):
    
    jb_pval_isnormal_aux = [[],[],[]]
    for i in range(len(r.columns)):
        jb, pvalue = scipy.stats.jarque_bera(r.iloc[:,i])
        jb_pval_isnormal_aux[0].append(jb)
        jb_pval_isnormal_aux[1].append(pvalue)
        jb_pval_isnormal_aux[2].append(is_normal(r.iloc[:,i]))
    
    jb_pval_isnormal=pd.DataFrame({"JB":jb_pval_isnormal_aux[0],
                                   "p_value": jb_pval_isnormal_aux[1],
                                   "Normal": jb_pval_isnormal_aux[2]})

    return jb_pval_isnormal

def semideviation(r):
        
        '''
        Returns the semideviation aka negative semideviation of r.
        r must be a Series or DataFrame
        '''
        
        return r[r<0].std(ddof=0)
    
def var_historic(r,per=5):
    
   """
   Returns historical VaR
   
   """
   
   if isinstance(r, pd.DataFrame):
        """
        If r is not pd.DataFrame, it returns a False. If True,
        with aggregate we run the very same function on every column
        The difference is that when it is called again, it will be called
        as Series. If you run np.percentile() on a DataFrame, you will           see that it returns the percentiles as an array, not as a Series.           Hence, the need for aggregate, which returns the percentiles as           Series

        """
        return r.aggregate(var_historic, per=per) 
    
   elif isinstance(r, pd.Series):
        return -np.percentile(r,per) #- because VaR is given as positive
    
   else:
        raise TypeError("Expected r to be Series or DataFrame")
     
    
def var_gaussian(r, quant=0.05): #it is a quantile, not a percentile   
    
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    # computes the z-score assuming it was Gaussian
    z = norm.ppf(quant)
    
    return -(r.mean() + z*r.std(ddof=0))

def var_cornish_fisher(r, quant=0.05):
    
    """
    Return the Cornish-Fisher VaR of a Series or DataFrame.
    """
    
    s = skewness(r)
    k = kurtosis(r)
    z = norm.ppf(quant)
    
    z_mod = z + ((z**2-1)*(s))/6 + ((z**3-3*z)*(k-3))/24 -((2*z**3-5*z)*(s**2))/36
            
    return -(r.mean() + z_mod*r.std(ddof=0))
    
def var_comparison(r,per=5,quant=0.05,plot=False):
    
    """
    Returns all VaRs we have defined into a nice DataFrame.
    If plot=True, it will plot a comparison
    """
    var_df = pd.DataFrame({"Historic": var_historic(r,per),
                           "Gaussian": var_gaussian(r,quant),
                           "Cornish-Fisher": var_cornish_fisher(r,quant)})
    
    if plot==True:
    
        var_df.plot.bar(title="VaR Comparison")
    
    
    return var_df

def cvar_historic(r,per=5):
    
   """
   Returns historical CVaR. It is almost a copypaste of VaR. What we do      is ask for the mean of the returns that have a value lower than the      historic VaR
   
   """
   
   if isinstance(r, pd.DataFrame):    
        return r.aggregate(cvar_historic, per=per)  
   elif isinstance(r, pd.Series):
        return -r[r<=-var_historic(r,per=per)].mean() # var_historic is positive, we have to add another - sign.   
   else:
        raise TypeError("Expected r to be Series or DataFrame")
        
def cvar_gaussian(r,quant=0.05):
    """
    Returns Gaussian CVaR
    """
    return -r[r<=-var_gaussian(r,quant=quant)].mean() 
    
    
def cvar_cornish_fisher(r,quant=0.05):
    """
    Returns Cornish-Fisher CVaR.
    """
    return -r[r<=-var_cornish_fisher(r,quant=quant)].mean() 

def cvar_comparison(r,per=5,quant=0.05,plot=False):
    
    """
    Returns all VaRs we have defined into a nice DataFrame.
    If plot=True, it will plot a comparison.
    It is a copypaste from var_comparison
    """
    cvar_df = pd.DataFrame({"Historic": cvar_historic(r),
                           "Gaussian": cvar_gaussian(r),
                           "Cornish-Fisher": cvar_cornish_fisher(r)})
    
    if plot==True:
    
        cvar_df.plot.bar(title="CVaR Comparison")
    
    
    return cvar_df


def portfolio_return(weights,returns):
    
    """
    It gives the average return of our portfolio
    """
    
    return weights.T @ returns #transpose of weights, @: matrix multiplication



def portfolio_vol(weights,covmat):
    
    """
    It gives the volatility of our portfolio
    """
    
    return (weights.T @ covmat @ weights.T)**0.5 
    #we use square root, because otherwise we would have variance
    
    
def plot_ef2(n_points,er,cov):
    
    """
    Plots the 2-asset efficient frontier. N_points is the number of
    points of the frontier, er the annualized returns (expected returns), and cov is the covariance
    """
    
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    rets = [portfolio_return(w,er) for w in weights]
    vols = [portfolio_vol(w,cov) for w in weights]
    ef_data = pd.DataFrame({
                            "Returns": rets, 
                            "Volatility": vols})
    
    return ef_data.plot.scatter(x = "Volatility", y="Returns",title="Efficient Frontier",s=0.3,c="red")



def minimize_vol(target_return,er,cov):
    """
    Given some constraints and an initial guess, it returns the weights
    that produce the minimum volatility for a target return.
    """
    
    n = er.shape[0] #number of assets
    init_guess = np.repeat(1/n,n) # same weights for everyone
    
    #We need some constraints
    
    bounds = ((0.0,1.0),)*n # w create n tuples with bound from 0 to 1
    
    return_is_target = {
        'type': 'eq', #constraint that says our function below has to equal zero
        'args':(er,) , #makes sure that additional arguments (that are constant) are added to the function
        'fun' : lambda weights, er: target_return - portfolio_return(weights,er) #function that has to meet the constraint
    }
    
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    
    results = minimize(portfolio_vol,init_guess, args=(cov,), method = "SLSQP", options={"disp": False},
                      constraints=(return_is_target, weights_sum_to_1), bounds=bounds)
    
    # In minimize, we have a function called portfolio_vol that depends on two arguments: covariance matrix
    # and weights. However, we only want the minimization of the function by changing the weights, as the covariance
    # matrix is a constant (for this reason, we use args=(cov,) ). Apart from this, the function minimize also
    # requires constraints and bound, and the method we will use to optimize.
    
    
    return results.x #solution array, have a look at documentation


def optimal_weights(n_points,er,cov):
    """ 
    Returns a list of weights that minimize the volatility associated to n equally spaced points
    in the interval [min(er),max(er)], where er is the expected return. Cov is the covariance 
    matrix.
    """
    target_returns = np.linspace(er.min(),er.max(),n_points)
    weights = [minimize_vol(target_ret,er,cov) for target_ret in target_returns]
    
    return weights
    


def ef_multi(n_points,er,cov,plot=False):
    
    """
    Returns a Dataframe a multi-asset efficient frontier, i.e returns and associated weights and minimized
    volatility. If plot==True, it plots the efficient frontier.
    """
    
    weights = optimal_weights(n_points,er,cov)
    rets = [portfolio_return(w,er) for w in weights]
    vols = [portfolio_vol(w,cov) for w in weights]
    
    weights_dataframe = pd.DataFrame (weights, columns = er.index)
    ret_vol_dataframe = pd.DataFrame({"Returns": rets,
                       "Volatility": vols,
                                     })
    ef = pd.concat([ret_vol_dataframe,weights_dataframe], axis=1)
    
    if plot == False:
        return ef
    
    if plot == True:
        ef.plot.scatter(figsize=(10, 5),x = "Volatility", y="Returns",title="Efficient Frontier",s=0.3,c="red",label="Efficient frontier")
       
        return ef
    
    
    
    


def ef_complete(riskfree_rate,er,cov,n_points=None, show_sr_df=False, show_gmv_df=False, show_ew_df=False,
                plot_ef=False, plot_sr=False,plot_gmv=False,plot_ew=False):
    """
    Returns DataFrames with returns, volatilities and weights and plots of Efficient Frontier (ef), Maximizimed Sharpe
    ratio (sr), Capital Market Line (sr too), Global Variance Minimum portfolio (gmw) and Equally Weighted portfolio (ew).   In order to obtain the DataFrame/Plot, the corresponding "show__df"/"plot__" must be set to True. For the plots, the  corresponding "show__df" and "plot_ef" must also be set to true
    """
        
#------------------------------------------------------- DATAFRAMES

    sr_df=gmv_df=ew_df=None #unless we ask to show them, the dataframes will be returned as Nones
    
    if show_sr_df == True:  #gives DataFrame with data for Maximimized Sharpe ratio
     
        n = er.shape[0] #number of assets
        init_guess = np.repeat(1/n,n) # same weights for everyone
    
        #We need some constraints for Sharpe ratio.
    
        bounds = ((0.0,1.0),)*n # w create n tuples with bound from 0 to 1
    
        weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
        }
    
    
        # Please, have a look at minimize_vol for more explanation. The difference is that instead of maximizing
        # Sharpe ratio, we minimize the negative Sharpe ratio, so we obtain the maximized Sharpe ratio, but with 
        # a negative sign.
    
        def neg_sharpe_ratio(weights,riskfree_rate,er,cov):
            """
            Returns the negative of Sharpe ratio.
            """
            r = portfolio_return(weights,er)
            vol = portfolio_vol(weights,cov)
            return - (r-riskfree_rate)/vol
    
    
        minimized_weights = minimize(neg_sharpe_ratio,init_guess, args=(riskfree_rate,er,cov,), method = "SLSQP", options={"disp": False}, constraints=(weights_sum_to_1), bounds=bounds)

        #We create the DataFrame with minimized weights, vol, return and max Sharpe ratio
        
        max_sharpe_ratio = -neg_sharpe_ratio(minimized_weights.x,riskfree_rate,er,cov)
    
        minimized_weights_dataframe = pd.DataFrame(np.reshape(minimized_weights.x,(1,minimized_weights.x.shape[0])), columns = er.index)
        ret_vol_max_sr_dataframe = pd.DataFrame({"Returns MSR": [portfolio_return(minimized_weights.x,er)], 
                                                 "Volatility MSR":[portfolio_vol(minimized_weights.x,cov)],
                                                 "Maximized Sharpe Ratio": [max_sharpe_ratio]
                                                 })
    
        sr_df = pd.concat([ret_vol_max_sr_dataframe,minimized_weights_dataframe], axis=1)
        
        
        
    if show_ew_df == True:  #gives DataFrame with data for Equal Weight portfolio
        
        n = er.shape[0] #number of assets
        weights_ew = np.repeat(1/n,n) # same weights for everyone
        ew_df = pd.DataFrame({"Returns EW": [portfolio_return(weights_ew,er)], 
                       "Volatility EW":[portfolio_vol(weights_ew,cov)]
                              })
        
    if show_gmv_df == True:
        
        #we have to find the MSR of the equal ER per asset portfolio
        
        n=er.shape[0]
        equal_er=np.repeat(1,n) #the value of the expected return does not matter, riskfree ratio does not matter either
        
        def neg_sharpe_ratio(weights,riskfree_rate,er,cov): 
            """
            Returns the negative of Sharpe ratio.
            """
            r = portfolio_return(weights,er)
            vol = portfolio_vol(weights,cov)
            return - (r-riskfree_rate)/vol
        
        weights_gmv = minimize(neg_sharpe_ratio,init_guess, args=(riskfree_rate,equal_er,cov,), method = "SLSQP", options={"disp": False}, constraints=(weights_sum_to_1), bounds=bounds) 
        
        weights_gmv_dataframe = pd.DataFrame(np.reshape(weights_gmv.x,(1,weights_gmv.x.shape[0])), columns = er.index)
        ret_vol_max_gmv_dataframe = pd.DataFrame({"Returns GMV": [portfolio_return(weights_gmv.x,er)], 
                                                 "Volatility GMV":[portfolio_vol(weights_gmv.x,cov)]
                                                 })
    
        gmv_df = pd.concat([ret_vol_max_gmv_dataframe,weights_gmv_dataframe], axis=1)
    
        
#------------------------------------ PLOTTING  
   
    if plot_ef == True: #plot efficient frontier. This will always be plotted and must be true in order to plot the rest
        
        ef_data = ef_multi(n_points,er,cov,plot=True) # we have to set the number of points for the frontier   
        
        
        if plot_sr == True: #plot CML and tangency point
               
            def cml(x,riskfree_rate,max_sharpe_ratio): #we create the function for the Capital Market Line
            
                return riskfree_rate + max_sharpe_ratio*x
        
            x = np.linspace(0, ret_vol_max_sr_dataframe["Volatility MSR"],n_points) #we choose the points the line will have
            
            #Plotting CML and Tangency Point
            plt.plot(x,cml(x,riskfree_rate,max_sharpe_ratio), label="Capital Market Line",linestyle="dashed")
            plt.scatter(ret_vol_max_sr_dataframe["Volatility MSR"],ret_vol_max_sr_dataframe["Returns MSR"], label = "Tangency Point", s=15, c="green",marker='o' )
            plt.xlim(0,ef_data["Volatility"].max())
            
                       
        if plot_ew == True: #plot equal weight portfolio
            
            plt.scatter(ew_df["Volatility EW"], ew_df["Returns EW"],s=15, label="EW portfolio")
       
        
        if plot_gmv == True:
            
             plt.scatter(gmv_df["Volatility GMV"], gmv_df["Returns GMV"],s=15, label="GMV portfolio")
            
        
        
        plt.legend()
        plt.title("Efficient Frontier")
    
    
    return sr_df,ew_df,gmv_df
    
    
    
    
    
def cppi(risky_r,floor,init_invest, safe_return=0.03, m=3,drawdown=None):

    """
    Returns  a dictionary the account value,cushion and risky asset weight history of a given portfolio,
    along with other stuff. Risky asset returns whose index are the date of return, floor and initial 
    investment must be given. Safe asset return and m factor are assumed, but parameters can be changed.
    """
    
    #Creating safe return dataframe
    safe_r = pd.DataFrame().reindex_like(risky_r)
    safe_r[:] = safe_return/12
    
    dates = risky_r.index
    n_steps = len(dates)
    account_value = init_invest
    floor_value = floor*init_invest
    peak = init_invest
    # we save the history of account value, cushion and weight of risky asset
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    floor_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None: #setting dynamic floor
            peak = np.maximum(peak,account_value)
            floor_value = peak*(1-drawdown)
            floor_history.iloc[step] = floor_value     
        cushion = (account_value - floor_value)/account_value #given as a percentage
        risky_w = m*cushion
        risky_w = np.minimum(risky_w,1) #if we have a cushion of 40%, for example, risky_w would be 120%. We have to 
        #add a constraint so it doesn't go further than 100%, because this means we would have to add money to the  investment
        risky_w = np.maximum(risky_w,0) # we can also have negative weight, which means we should short. This constraint avoids that
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w

        #Update account value

        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])

        #Saving data to have history

        account_history.iloc[step] = account_value
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        
        backtest_result = {
            "Wealth": account_history,
            "Risky Wealth": init_invest*(1+risky_r).cumprod(),
            "Risk Budget": cushion_history,
            "Risky allocation": risky_w_history,
            "m": m ,
            "start": init_invest,
            "floor":floor,
            "risky_r": risky_r,
            "safe_r" : safe_r,
            "Dynamic Floor": floor_history
        }
        
    return backtest_result



def summarize_stats(r,riskfree_rate=0.03):
    
    """
    Returns a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    
    ann_r = r.aggregate(annualized_rets,periods_per_year=12)
    ann_vol = r.aggregate(annualized_vol,periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio,riskfree_rate=riskfree_rate,periods_per_year=12)
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_cornish_fisher)
    cf_cvar5 = r.aggregate(cvar_cornish_fisher)
    
    
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Cornish-Fisher CVaR": cf_cvar5,
        "Sharpe Ratio": ann_sr
    })

def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0,prices=True):
    
    """
    Evolution of a Stock Price using a Geometric Brownian Motion Model.
    n_years: duration of evolution
    n_scenarios: number of stocks
    mu: expected annualized return
    sigma: annualized volatility
    steps_per_year: number of "dts" in a year, that is, number of returns per year
    s_0: initial stock price
    
    Returns stock prices if prices==True (default). If False, it returns the returns
    """
    
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year)
    #We generate random numbers with mean (1 + mu*dt) and standard deviation sigma*sqrt(dt).
    # If we do this, our function will be much faster than if we generate random numbers following
    #standard gaussian distribution. The 1 is added already too because we are going to calculate the cummulative product
    rets_plus_1 = pd.DataFrame(np.random.normal(loc=1+mu*dt, scale=sigma*np.sqrt(dt),size = (n_steps,n_scenarios)))
    rets_plus_1.iloc[0] = 1 #that way, all the prices start at 100
    stock_price = s_0*rets_plus_1.cumprod()
    if prices==True:
        return stock_price
    elif prices==False:
        return rets_plus_1 - 1
    
    
def show_gbm(n_scenarios,mu,sigma,steps_per_year=12,s_0=100,n_years=10):
    
    """
    Evolution of a Stock Price using a Geometric Brownian Motion Model.
    n_years: duration of evolution
    n_scenarios: number of stocks
    mu: expected annualized return
    sigma: annualized volatility
    steps_per_year: number of "dts" in a year, that is, number of returns per year
    s_0: initial stock price
    """
    
    
    prices = gbm(n_years=n_years,n_scenarios=n_scenarios, steps_per_year=steps_per_year, mu=mu,sigma=sigma,s_0=s_0)
    plt.figure(figsize=(12,6))
    plt.plot(prices,color="indianred",alpha=0.5,linewidth=2,label='_nolegend_')
    plt.hlines(s_0,0,n_years*steps_per_year,ls=":",color="black",label="Starting price")
    plt.plot(0,s_0,marker='o',color='darkred',alpha=0.2,label="_nolegend_")
    plt.ylabel("Stock price")
    plt.xlabel("Number of steps")
    
    plt.legend()
    
    

def show_cppi(n_years=10,n_scenarios=50,steps_per_year=12,mu=0.07,
              sigma=0.15,m=3,floor=0.8,riskfree_rate=0.03,y_max=100,s_0=100):
    """
    Plots the Monte Carlo Simulations of CPPI along with an histogram of final wealth values.
    It also shows the floor violations
    Parameters are the same as in previous functions.
    """
    
    #Calculating CPPI
    sim_rets = gbm(n_years=n_years,n_scenarios=n_scenarios,mu=mu,
                      sigma=sigma,prices=False,steps_per_year=steps_per_year,s_0=s_0)
    risky_r = pd.DataFrame(sim_rets)
    wealth = cppi(risky_r,floor=floor,init_invest=s_0,safe_return=riskfree_rate)["Wealth"]
    y_max=wealth.values.max()*y_max/100
    
    # Calculating violations
    terminal_wealth = wealth.iloc[-1]
    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    failure_mask = np.less(terminal_wealth,s_0*floor) #array with booleans (True or False)
    n_failures = failure_mask.sum()
    p_fail = n_failures/n_scenarios
    
    e_shortfall = np.dot(terminal_wealth-s_0*floor, failure_mask)/n_failures if n_failures > 0 else 0.0 
    #expected shortfall is the equivalent of conditional VaR whn the floor is trespassed. It is the mean
    #of those prices that are below the floor. We are performing the dot product with the failure_mask
    #in order to have only those prices that meet that condition.
    
    #Plotting
    
    fig, axes = plt.subplots(1, 2,sharey="all",figsize=(24,9), gridspec_kw={'width_ratios':[3,2]}) #creatig subplot with ratio 3:2
    plt.subplots_adjust(wspace=0) #no separation between figures
    
    axes[0].plot(wealth, alpha=0.3,color="indianred",label="_nolegend_")
    axes[0].axhline(s_0,0,n_years*steps_per_year,ls=":",color="black",label="Starting price")
    axes[0].axhline(s_0*floor,0,n_years*steps_per_year,ls="--",color="blue",label="Floor")
    axes[0].set_ylabel("Stock price")
    axes[0].set_xlabel("Number of steps")
    axes[0].set_ylim(0,y_max)
    axes[0].legend()
    axes[1].hist(terminal_wealth,bins=50, color="indianred", orientation="horizontal")
    axes[1].annotate(f"Mean: {int(tw_mean)}",xy=(.5,.9), xycoords='axes fraction', fontsize=24)
    axes[1].annotate(f"Median: {int(tw_median)}",xy=(.5,.85), xycoords='axes fraction', fontsize=24)
    if floor>0.01:
        axes[1].annotate(f"Violations: {n_failures} ({p_fail*100:2.2f}%)\nE(shortfall)=${e_shortfall:2.2f}",xy=(.5,.75), xycoords='axes fraction', fontsize=24)
        
        
        
def discount(t,r):
    
    """
    Compute the price of a pure discount bond that pays a dollar at time t (years), given interest rate.
    Returns the discounts in a DataFrame indexed by the time period t.
    We do this so r can be a vector as well.
    """
    
    discounts = pd.DataFrame([(r+1)**(-i) for i in t ])
    discounts.index = t
    return discounts




def pv(flows,r):
    
    """
    Computes the present value of a sequence of liabilities
    flows is indexed by the time, and the values are the amount of each liability
    """
    
    dates = flows.index
    discounts = discount(dates,r)
    return discounts.multiply(flows, axis='rows').sum()


def discount_simple(t,r):
    
    """
    Compute the price of a pure discount bond that pays a dollar at time t (years), given interest rate.
    Returns the discounts in a DataFrame indexed by the time period t.
    We do this so r can be a vector as well.
    """
    return (1+r)**(-t)


def pv_simple(flows,r):
    
    """
    Computes the present value of a sequence of liabilities
    flows is indexed by the time, and the values are the amount of each liability
    """
    date = flows.index
    discounts = discount_simple(date,r)
    return np.dot(discounts,flows)
    

    
    
def funding_ratio(assets,liabilities,r):
    """
    Computes the funding ratio of some assets given liabilities and interest rate
    """
    
    return float(pv(assets,r)/pv(liabilities,r))


def funding_ratio_simple(assets,liabilities,r):
    """
    Computes the funding ratio of some assets given liabilities and interest rate
    """
    
    return assets/pv(liabilities,r)

def show_funding_ratio(assets,liabilities,r):
    fr = funding_ratio(assets,liabilities,r)
    print(f"{fr*100:2f}")
    
    
def show_funding_ratio_simple(assets,liabilities,r):
    fr = float(funding_ratio_simple(assets,liabilities,r))
    print(f"{fr*100:2f}")
    
    
    
    
def inst_to_ann(r):
    
    """
    Converts short/instantaneous rate to an annualized rate
    """
    
    return np.expm1(r)

def ann_to_inst(r):
    
    """
    Converts annualized rate to short rate
    """
    
    return np.log1p(r)


def cir(n_years=10,n_scenarios=1,a=0.05,b=0.03,sigma=0.05,steps_per_year=12,r_0=None):
    
    """
    Implements the CIR model for interest rates
    Returns annualized rates and prices
    """
    
    if r_0 is None:
        r_0 = b
        
    r_0 = ann_to_inst(r_0)
    dt= 1/steps_per_year
    #We generate random numbers with mean 0 and standard deviation sigma*sqrt(dt).
    # The scale should be 1 because we are using standard Gaussian distribution,
    # but this makes the following code faster
    num_steps = int(n_years*steps_per_year) + 1 #we add the one because the first row would be the initial rate
    shock = np.random.normal(0,scale=np.sqrt(dt),size=(num_steps,n_scenarios))  #dW, Brownian motion
    rates = np.empty_like(shock) #empty array with the size of shock
    rates[0] = r_0
    
    ## For Price Generation
    
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    
    def price(ttm,r):
        
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P= _A*np.exp(-_B*r)
        return _P
    
    prices[0] = price(n_years,r_0)
    
    for step in range(1,num_steps): 
        r_t=rates[step-1] 
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t) #these rates are instantaneous. In particular, this is the r_{t+1},
        #which we are storing for the next loop, which corresponds to t+1, and with which we will
        #obtain dr_{t+1}
        #Generates prices at time t as well
        prices[step] = price(n_years-step*dt,rates[step])
        
    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps)) #Return everything annualized
    prices = pd.DataFrame(data=prices, index=range(num_steps)) #Return everything annualized
    
    return rates, prices



def show_cir_rates(r_0=0.03,a=0.5,b=0.03,sigma=0.05,n_scenarios=5):
        cir(r_0=r_0,a=a,b=b,sigma=sigma,n_scenarios=n_scenarios)[0].plot(legend=False)
        
def show_cir_prices(r_0=0.03,a=0.5,b=0.03,sigma=0.05,n_scenarios=5):
        cir(r_0=r_0,a=a,b=b,sigma=sigma,n_scenarios=n_scenarios)[1].plot(legend=False)
        

        
def bond_cash_flows(maturity,principal=100,coupon_rate=0.03,coupons_per_year=12):
    
    """
    Returns a Series of cash flows generated by a bond,
    indexed by a coupon number
    """
    
    n_coupons = int(maturity*coupons_per_year)
    coupon_amt = principal*(coupon_rate/coupons_per_year)
    coupon_times = np.arange(1,n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal #in last period, initial investment is returned
    return cash_flows





def bond_price(maturity,principal=100,coupon_rate=0.03,coupons_per_year=12,discount_rate=0.03):
    
    """
    Computes the price of a bond that pays regular coupons until maturity, at which
    time the principal and the final coupon is returned.
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon 
    date and the bond value is computed over time.
    """
    
    if isinstance(discount_rate,pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year,principal,coupon_rate,
                                       coupons_per_year, discount_rate.loc[t]) #maturity decreases with each loop
        return prices
    
    else: 
        if maturity <=0: return principal + (principal*coupon_rate)/coupons_per_year #single time period
        cash_flows = bond_cash_flows(maturity,principal,coupon_rate,coupons_per_year)
        return pv(cash_flows,discount_rate/coupons_per_year)


def bond_price_simple(maturity,principal=100,coupon_rate=0.03,coupons_per_year=12,discount_rate=0.03):
    
    """
    Computes the price of a bond that pays regular coupons until maturity, at which
    time the principal and the final coupon is returned.
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon 
    date and the bond value is computed over time.
    """
    
    if isinstance(discount_rate,pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year,principal,coupon_rate,
                                       coupons_per_year, discount_rate.loc[t]) #maturity decreases with each loop
        return prices
    
    else: 
        if maturity <=0: return principal + (principal*coupon_rate)/coupons_per_year #single time period
        cash_flows = bond_cash_flows(maturity,principal,coupon_rate,coupons_per_year)
        return pv_simple(cash_flows,discount_rate/coupons_per_year)
    

def macaulay_duration(flows,discount_rate):
    
    """
    Computes the Macauly Duration of a sequence of cash flows
    """
    
    discounted_flows = discount_simple(flows.index,discount_rate)*flows
    weights = discounted_flows/discounted_flows.sum()
    return (flows.index*weights).sum()


def match_durations(cf_t,cf_s,cf_l,discount_rate):
    """"
    Returns the weight W in cf_s that, along with (1-W) in cfl_l, will have an
    effective duration that matches cf_t
    
    cf_t: Total liability to be paid/Cash flow to be matched
    cf_s: Cash flow of short coupon bond
    cf_l: Cash flow of long coupon_bond
    discount_rate: same for all of them. Money devalues the same way.
    """
    
    d_t = macaulay_duration(cf_t,discount_rate)
    d_s = macaulay_duration(cf_s,discount_rate)
    d_l = macaulay_duration(cf_l,discount_rate)
    
    return (d_l-d_t)/(d_l-d_s)



def bond_total_return(monthly_prices,principal,coupon_rate,coupons_per_year):
    
    """
    Computes the total return of a Bond based on monthly bond prices and coupon payments.
    Assumes that dividends (coupons) are apid out at the end of the period (e.g. end of 3 months for quarterly div)
    and hat dividends are reinvested in the bond
    """
    
    coupons = pd.DataFrame(data=0, index=monthly_prices.index,columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year,t_max,int((coupons_per_year*t_max)/12),dtype=int)
    coupons.iloc[pay_date] = principal*(coupon_rate/coupons_per_year) #coupons are only added in the corresponding months
    # the other positions were set as 0 when the DataFrame was created
    total_returns = (monthly_prices + coupons)/monthly_prices.shift() - 1 #actual monthly price + coupons / immediately    previous monthly price
    return total_returns.dropna()
    
    
    
def bt_mix(r1,r2, allocator, **kwargs):
    
    """
    Runs a bak test (simulation) of allocating between a two sets of returns.
    r1 and r2 DataFrames or returns with the index as a time and N scenarios.
    "allocator" is a function that takes two sets of returns and allocator specific parameters,
    and produces and allocation to the first portfolio (the rest of the money is invested in GHP)
    as a DataFrame.
    Returns a DataFrame f the resulting N portfolio scenarios
    """
    
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 need to be the same shape")
        
    weights = allocator(r1,r2,**kwargs)
    
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights that dont match r1")
        
    r_mix = weights*r1 + (1-weights)*r2
    
    return r_mix 




def fixedmix_allocator(r1,r2,w1,**kwargs):
    
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios.
    PSP and GHP are DataFrames that represent the returns of the PSP and GHP such that:
         each column is a scenario
         each row is the price for a timestep
    Returns a DataFrame of PSP Weights
    """
    
    #Weight is fixed for any time, so it will return an r1 (or r2, it is the same) shaped
    #DataFrame with value w1 in all positions 
    
    return pd.DataFrame(data=w1, index=r1.index,columns=r1.columns) #r1 and r2 index and columns should be the same
    
    
def terminal_values(rets):
    
    """
    Returns the final values of a dollar at the end of the return period for each scenario
    """
    
    return (rets+1).prod()


def terminal_stats(rets,floor=0.8,cap=np.inf,name="Stats"):
    
    """
    Produce Summary Statistics on the terminal values per invested dollar across
    a range of N scenarios.
    rets is a DataFrame with returns, with time as index.
    Returns a 1 row DataFrame of Summary Stats:
    p_breach: percentage of floor breaches
    e_breach: mean of the breach value
    p_reach: percentage of cap reaches
    e_reach: mean of the reach value
    
    """
    
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan #percentage of breaches
    p_reach = reach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor - terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan #only proceeds when breach==True
    e_surplus = (cap-terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame({"Mean": [terminal_wealth.mean()],
                             "STD": [terminal_wealth.std()],
                             "P_breach": [p_breach],
                             "E_short": [e_short],
                             "P_reach": [p_reach],
                             "E_surplus": [e_surplus]},
                            index=[name])

    return sum_stats



def glidepath_allocator(r1,r2,start_glide=1,end_glide=0):
    
    """
    Simulates a Target-Date-Fund style gradual move from r1 to r2
    """
    
    
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide,end_glide,n_points))
    paths = pd.concat([path]*n_col,axis=1) #we have n_col scenarios, and all of them follow the same path.
    #This way we have n_col columns (one for each scenario) with the same path. Concatenating with axis=1
    #is important. In order to understand, try the following: pd.concat([pd.Series([2,3])]*2,axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths



def floor_allocator(psp_r,ghp_r,floor,zc_prices,m=3):
    
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside 
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP.
    Returns a DataFrame with the same shape as the PSP/GHP representing the weights in the PSP
    """
    
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC prices must have the same shape")
    n_steps,n_scenarios = psp_r.shape
    account_value = np.repeat(1,n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame (index=psp_r.index,columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] #PV of Floor assuming todays'rates and flat Yield Curve
        cushion = (account_value-floor_value)/account_value #gives cushion as percentage.
        psp_w = (m*cushion).clip(0,1) # values of m*cushion smaller than become 0 and values bigger than 1 become 1.
        #It is a way to bound the weights
        ghp_w = 1- psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        #recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
        
    return w_history



def drawdown_allocator(psp_r,ghp_r,maxdd, m=3):
    
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside 
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP.
    Floor is based on the maximum drawdown you would never want to meet at any time (this is the change
    with respect to floor_allocator)
    Returns a DataFrame with the same shape as the PSP/GHP representing the weights in the PSP
    """
    
    n_steps,n_scenarios = psp_r.shape
    account_value = np.repeat(1,n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1,n_scenarios)
    w_history = pd.DataFrame (index=psp_r.index,columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value #Floor is based on previous peak. 
        cushion = (account_value-floor_value)/account_value #gives cushion as percentage.
        psp_w = (m*cushion).clip(0,1) # values of m*cushion smaller than become 0 and values bigger than 1 become 1.
        #It is a way to bound the weights
        ghp_w = 1- psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        #recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value,account_value) #peak must be updated to establish new floor
        w_history.iloc[step] = psp_w
        
    return w_history
