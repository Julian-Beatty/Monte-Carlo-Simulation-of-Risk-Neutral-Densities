import os
import numpy as np
from KDEpy import*
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, LeaveOneOut
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from math import erf
from scipy.stats import ks_2samp
from statsmodels.nonparametric.kernel_regression import KernelReg


class monte_carlo:
    def __init__(self,svi_dict):
        self.svi_dict=svi_dict    
        
    def simulate_mc(self,svi_dictmodels,models_dict,runs):
        """
        

        Returns
        -------
        None.

        """
        # ---------- 1) simulate svi curve from svi dictionary, obtain iv,calls and rnd ----------
        svi_dict=self.svi_dict
        
        k=svi_dict["strikes"]
        s=svi_dict["underlying_price"]
        r=svi_dict["risk_free_rate"]
        t=svi_dict["maturity"]
        
        
        a=svi_dict["a"]
        b=svi_dict["b"]
        rho=svi_dict["rho"]
        m=svi_dict["m"]
        sigma=svi_dict["sigma"]
        
        svi_iv=simulate_svi(k,s,r,t,a,b,rho,m,sigma)
        calls=black_scholes_call_from_iv(k,s,r,t,svi_iv)
        svi_rnd=rnd_from_calls(k,calls,r,t)
        
        np.random.seed(42)
        ##Iterative monte carlo loop begins here M=
        # ---------- 2) perturb rnd by multiplying it by a moving sin wave. We call this the true rnd. Returns the new underlying price from the mean of this rnd. ----------
        #runs=1000
        for i in range(0,runs):
            #wave_dict=self.wave_dict
            
            wave_dict={"a":0.015,"w":np.random.uniform(0.15, 0.25),"c":0.02}

            
            
            true_rnd,true_s=truernd_from_sinwave(k,svi_rnd,r,t,wave_dict)
            true_call=calls_from_rnd(true_rnd, k, s, r, t, normalize=False)            
            #plt.plot(k/true_s,true_rnd)
        # ---------- 3) Add noise to each call option price to mimic microstructure noise ----------

            noisy_call = noisy_data_function(k, true_call, true_s, {
                'mode': 'rel_atm', 'sigma': 0.03, 'decay': 0.05, 'p': 1            })
            noisy_rnd=rnd_from_calls(k,noisy_call,r,t)
            noisy_iv=implied_vol_from_calls(noisy_call, true_s, k, r, t, q=0.0, tol=1e-8, max_iter=100, vol_low=1e-8, vol_high=5.0)
            
            
            
            #plt.plot(k,noisy_iv)
            #plt.plot(k,noisy_rnd)
            #plt.plot(k,true_rnd)
            #plt.plot(k,noisy_calls)
        

            # kde = NaiveKDE(bw="scott").fit(k,weights=noisy_rnd)
            # kde_pdf=kde.evaluate(k)
            # bw=kde.bw/8
            # kde = NaiveKDE(bw=bw).fit(k,weights=noisy_rnd)
            # kde_pdf=kde.evaluate(k)

            # area=np.trapz(kde_pdf,k)
            # kde_pdf=kde_pdf/area
            # plt.plot(k,noisy_rnd)
            # plt.plot(k,kde_pdf,label="kde")
            # plt.plot(k,true_rnd)
            # plt.legend()

        # ---------- 2) Apply models to each function ----------
            p_value_collector=[]
        #for M in range (0,10):
            result_dict=compare_rnd_models(true_call, true_rnd, noisy_call,noisy_iv, k, true_s, r, t,i, models_dict) #returns the KS statistic for each model in form of {model_nick:ks..}
            p_value_collector.append(result_dict) ##appends into list
        
        
        # ---------- 3)Aggregate into dataframe and summarize ----------
        pvalue_df = pd.DataFrame(p_value_collector)
        return pvalue_df


#####################Functions


def simulate_svi(k, s, r, t, a, b, rho, m, sigma):
    """
    Compute implied vols from raw SVI given strikes, spot S, and rate R.

    Parameters
    ----------
    K : array-like
        Strikes.
    S : float
        Spot price.
    R : float
        Continuously compounded risk-free rate (annualized).
    T : float
        Time to maturity in years.
    a, b, rho, m, sigma : floats
        Raw SVI parameters with b > 0, sigma > 0, |rho| < 1.

    Returns
    -------
    iv : np.ndarray
        Implied volatility for each strike K.
    """
    
    
    
    
    
    k = np.asarray(k, dtype=float)
    f = s * np.exp(r * t)                         # forward from s and r
    k = np.log(np.maximum(k, 1e-300) / f)         # log-moneyness vs f

    w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))  # total variance
    w = np.maximum(w, 1e-12)                      # guard
    iv = np.sqrt(w / t)                           # implied vol
    return iv

def black_scholes_call_from_iv(K, S, R, T, iv, q=0.0):
    """
    Convert IVs to Black–Scholes call prices C(K).

    Parameters
    ----------
    K : array-like
        Strikes.
    S : float
        Spot price.
    R : float
        Continuously compounded risk-free rate.
    T : float
        Maturity in years.
    iv : array-like
        Implied vols corresponding to K.
    q : float
        Continuous dividend yield (default 0).

    Returns
    -------
    C : np.ndarray
        Call prices for each strike K.
    """
    K = np.asarray(K, dtype=float)
    iv = np.asarray(iv, dtype=float)

    eps = 1e-12
    volT = np.maximum(iv * np.sqrt(max(T, eps)), eps)

    d1 = (np.log(np.maximum(S, eps) / np.maximum(K, eps)) + (R - q + 0.5 * iv**2) * T) / volT
    d2 = d1 - volT

    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)

    disc_r = np.exp(-R * T)
    disc_q = np.exp(-q * T)

    C = disc_q * S * Nd1 - disc_r * K * Nd2

    # Handle near-degenerate cases
    intrinsic = np.maximum(disc_q * S - disc_r * K, 0.0)
    near_degenerate = (iv < 1e-8) | (T < 1e-8)
    C = np.where(near_degenerate, intrinsic, C)

    return C

def _norm_cdf(x):
    """Standard normal CDF without SciPy, vectorized via math.erf."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))

def rnd_from_calls(K, C, R, T, clip_zero=True):
    """
    Risk-neutral density f_Q(K) via Breeden–Litzenberger:
        f_Q(K) = exp(R*T) * d^2 C / dK^2

    Parameters
    ----------
    K : array-like
        Strikes (monotone increasing recommended).
    C : array-like
        Call prices corresponding to K.
    R : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity in years.
    clip_zero : bool
        If True, clip tiny negative densities to 0 (from numerical noise).

    Returns
    -------
    f : np.ndarray
        Risk-neutral density evaluated on K.
    """
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)

    dC_dK   = np.gradient(C, K, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK, K, edge_order=2)

    f = np.exp(R * T) * d2C_dK2
    if clip_zero:
        f = np.maximum(f, 0.0)
    return f

def truernd_from_sinwave(k,svi_rnd,r,t,wave_dict):
    """
    

    Parameters
    ----------
    k : TYPE
        DESCRIPTION.
    svi_rnd : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    wave_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    
    ## f(k)=a*sin(w*k)**2 + c
    a=wave_dict["a"]
    w=wave_dict["w"]
    c=wave_dict["c"]
    
    ## Multiply by moving sin wave, then normalize. Extract the "true" stock price as the mean of this density, discounted to preseent
    moving_sin=a*(np.sin(w*k))**2+c
    perturbed_pdf=svi_rnd*moving_sin
    area=np.trapz(perturbed_pdf,k)
    true_rnd=perturbed_pdf/area
    true_s=np.trapz(true_rnd*k,k)*np.exp(-r*t)
    
    
    #
    plt.plot(k,true_rnd)
    plt.plot(k,svi_rnd)
    
    
    
    
    return true_rnd,true_s

def calls_from_rnd(rnd, K, S, R, T, normalize=False):
    """
    Recover Black–Scholes/Merton-style call prices from a risk-neutral PDF on S_T.

    Formula:
        C(K) = exp(-R*T) * ∫_{K}^{∞} (x - K) f_Q(x) dx
             = exp(-R*T) * [  ∫_{K}^{∞} x f_Q(x) dx  -  K ∫_{K}^{∞} f_Q(x) dx ]

    Parameters
    ----------
    rnd : array-like
        Risk-neutral density f_Q evaluated on the same grid as K (i.e., x-grid of S_T).
    K : array-like
        Monotonically increasing grid (interpreted as terminal price grid); prices are returned at these K's.
    S : float
        Spot price today (unused in formula, but kept for interface consistency/sanity).
    R : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity (years).
    normalize : bool (default False)
        If True, renormalizes `rnd` to integrate to 1 on the provided grid.

    Returns
    -------
    C : np.ndarray
        Call price curve evaluated at strikes equal to the grid K.
        (Tail beyond max(K) is assumed negligible.)
    """
    K = np.asarray(K, dtype=float)
    f = np.asarray(rnd, dtype=float)
    n = K.size
    if f.size != n:
        raise ValueError("rnd and K must have the same length.")
    if np.any(np.diff(K) <= 0):
        raise ValueError("K must be strictly increasing.")

    # Optional normalization of the PDF over the finite grid
    dx = np.diff(K)
    area = np.sum(0.5 * (f[:-1] + f[1:]) * dx)
    if normalize and area > 0:
        f = f / area

    # Right-tail cumulative integrals using trapezoids:
    # Tail probability A_i = ∫_{K_i}^{∞} f(x) dx
    seg_area_f = 0.5 * (f[:-1] + f[1:]) * dx
    A = np.zeros(n)
    if n > 1:
        A[:-1] = np.cumsum(seg_area_f[::-1])[::-1]  # right cumulative
        A[-1] = 0.0

    # Tail first moment B_i = ∫_{K_i}^{∞} x f(x) dx
    g = K * f
    seg_area_g = 0.5 * (g[:-1] + g[1:]) * dx
    B = np.zeros(n)
    if n > 1:
        B[:-1] = np.cumsum(seg_area_g[::-1])[::-1]
        B[-1] = 0.0

    # Call prices: C_i = e^{-R T} [ B_i - K_i * A_i ]
    disc = np.exp(-R * T)
    C = disc * (B - K * A)

    # Numerical guard: tiny negatives to zero
    return np.maximum(C, 0.0)


def implied_vol_from_calls(C, S, K, R, T, q=0.0, tol=1e-8, max_iter=100, vol_low=1e-8, vol_high=5.0):
    """
    Invert Black–Scholes to get implied vol(s) from call price(s) via bisection.

    Parameters
    ----------
    C : array-like
        Call prices (same shape as K).
    S : float
        Spot.
    K : array-like
        Strikes.
    R : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity (years).
    q : float
        Continuous dividend yield.
    tol : float
        Absolute tolerance on price error.
    max_iter : int
        Max bisection iterations.
    vol_low, vol_high : float
        Initial volatility bracket (in annualized vol units).

    Returns
    -------
    iv : np.ndarray
        Implied vols; NaN where no solution in the bracket (e.g., price outside BS bounds).
    """
    C = np.asarray(C, dtype=float)
    K = np.asarray(K, dtype=float)
    iv = np.full_like(C, np.nan, dtype=float)

    # No-arbitrage price bounds for a European call (with carry q)
    disc_r = np.exp(-R*T); disc_q = np.exp(-q*T)
    lower = np.maximum(disc_q*S - disc_r*K, 0.0)
    upper = disc_q*S  # BS call is increasing in vol, bounded above by discounted spot

    # Precompute price at bracket ends
    Cl = bs_call_price(S, K, R, T, vol_low, q=q)
    Ch = bs_call_price(S, K, R, T, vol_high, q=q)

    # Valid where C within [Cl, Ch] and [lower, upper]
    valid = (C >= np.maximum(lower, Cl) - 1e-12) & (C <= np.minimum(upper, Ch) + 1e-12)

    # Bisection per point
    a = np.full_like(C, vol_low); fa = Cl - C
    b = np.full_like(C, vol_high); fb = Ch - C

    # Ensure signs differ; otherwise mark invalid
    ok = valid & (fa * fb <= 0)

    # Work arrays (only where ok)
    a = a[ok]; b = b[ok]; K_ok = K[ok]; C_ok = C[ok]
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = bs_call_price(S, K_ok, R, T, m, q=q) - C_ok
        left = fm > 0
        # Update brackets
        b = np.where(left, m, b)
        a = np.where(left, a, m)
        if np.max(np.abs(fm)) < tol:
            break

    iv_ok = 0.5 * (a + b)
    iv[ok] = iv_ok
    return iv
def bs_call_price(S, K, R, T, vol, q=0.0):
    S = float(S); K = np.asarray(K, dtype=float); vol = np.asarray(vol, dtype=float)
    eps = 1e-12
    disc_r = np.exp(-R*T)
    disc_q = np.exp(-q*T)
    volT = np.maximum(vol*np.sqrt(max(T, eps)), eps)
    d1 = (np.log(np.maximum(S, eps)/np.maximum(K, eps)) + (R - q + 0.5*vol**2)*T) / volT
    d2 = d1 - volT
    return disc_q*S*_norm_cdf(d1) - disc_r*K*_norm_cdf(d2)

def noisy_data_function(k, true_calls, true_s, noise_dict):
    """
    Add noise that is largest near ATM (K≈S) and decays with |K/S - 1|.

    noise_dict keys (all optional):
      - mode: 'abs' or 'rel_atm' (default 'abs')
      - sigma: base noise scale (default 0.02)
               * 'abs'    -> price units
               * 'rel_atm'-> fraction of ATM call price
      - decay: moneyness scale for decay (default 0.07)  # ~7% away halves quickly
      - p:     steepness of decay (>=1, default 2.0)
      - seed:  RNG seed
    """
    mode  = noise_dict.get('mode', 'abs')
    sigma = float(noise_dict.get('sigma', 0.02))
    decay = float(noise_dict.get('decay', 0.07))
    p     = float(noise_dict.get('p', 2.0))
    seed  = noise_dict.get('seed', None)

    k = np.asarray(k, dtype=float)
    C = np.asarray(true_calls, dtype=float)
    S = float(true_s)
    rng = np.random.default_rng(seed)

    # moneyness distance from ATM
    d = np.abs(k / S - 1.0)

    # weight: 1 at ATM, decays as distance grows
    w = np.exp(- (d / decay) ** p)

    # choose a scale that's NOT larger on deep ITM/OTM ends
    if mode == 'rel_atm':
        atm_idx = np.argmin(np.abs(k / S - 1.0))
        atm_level = max(C[atm_idx], 1e-12)
        scale = sigma * atm_level
    else:  # 'abs'
        scale = sigma

    eps = rng.standard_normal(C.shape)
    noisy = C + (scale * w) * eps

    # basic bounds for calls
    noisy = np.clip(noisy, 0.0, S)
    return noisy


def compute_samples(x, fx, n_samples=50_000, *, clip_negative=True):
    x = np.asarray(x)
    fx = np.asarray(fx)

    if x.ndim != 1 or fx.ndim != 1 or x.size != fx.size:
        raise ValueError("x and fx must be 1-D arrays of the same length.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")

    if clip_negative:
        fx = np.clip(fx, 0, None)
    if np.any(~np.isfinite(fx)):
        raise ValueError("fx contains non-finite values.")

    # Normalize PDF
    area = np.trapz(fx, x)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("PDF area must be positive and finite.")
    fx = fx / area

    # Build CDF (length == len(x))
    dx = np.diff(x)
    cdf_inner = np.cumsum((fx[:-1] + fx[1:]) * 0.5 * dx)
    cdf = np.concatenate(([0.0], cdf_inner))
    # Numerical safety: force last point to 1 exactly
    cdf[-1] = 1.0

    # Draw uniforms and invert CDF -> samples
    u = np.random.random(n_samples)  # (0,1)
    samples = np.interp(u, cdf, x)   # piecewise-linear inverse

    return samples

def compare_rnd_models(true_call,true_rnd,noisy_call,noisy_iv,k,true_s,r,t,run_number,models_dict):
    """
    

    Parameters
    ----------
    true_call : TYPE
        DESCRIPTION.
    true_rnd : TYPE
        DESCRIPTION.
    noisy_call : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    true_s : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    models_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    result_dict={}
    rnd_iv_dict={}
    
    true_rnd_samples=compute_samples(k,true_rnd)

    for model_key in models_dict:
        if model_key=="local_regression":
            
            print("Attempting Local Regression")
            nickname=models_dict[model_key]["nickname"]
            
            
            ##---Extract the dictionary of model parameters for local regression
            local_polynomial_param_dict=models_dict[model_key]
            
            ##---Run the local regression model
            specific_model_dict=local_regression(k,noisy_call,noisy_iv,r,true_s,t,local_polynomial_param_dict) #should return at the very least iv,call,rnd
            
            ##--Extract the rnd, take samples and perform the KS test (extract pvalue).
            rnd=specific_model_dict["rnd"]
            pvalue_list=[]
            for i in range(0,100):
                specific_samples=compute_samples(k,rnd,1000)
                D,pvalue= ks_2samp(true_rnd_samples, specific_samples, alternative='two-sided', method='auto')
                pvalue_list.append(pvalue)  # no assignment here
            average_pvalue=np.mean(pvalue_list)
            
            
            
            ##--Store the KS Pvalue into a dictionary, indexed by the nickname
            result_dict[nickname]=average_pvalue
            ##--Store the IV and RND curves into a seperate dictionary, indexed by nickname (for plotting purposes later)
            rnd_iv_dict[nickname]=specific_model_dict
            
        if model_key=="local_regression_kde":
            nickname=models_dict[model_key]["nickname"]
            print(f"Attempting local Regression with KDE. Saving as: {nickname}")
            
            ##---Extract the dictionary of model parameters for local regression
            local_polynomial_kde_param_dict=models_dict[model_key]
            
            ##---Run the local regression with kde model
            specific_model_dict=local_regression_kde(k,noisy_call,noisy_iv,r,true_s,t,local_polynomial_kde_param_dict) #should return at the very least iv,call,rnd
            
            ##--Extract the rnd, take samples and perform the KS test (extract pvalue).
            rnd=specific_model_dict["rnd"]
            
            ##-- The power of the KS test increases with higher sample. To avoid the trivial case of instant-rejection from high N, I take 100 random draws and average the pvalue.
            pvalue_list=[]
            for i in range(0,100):
                specific_samples=compute_samples(k,rnd,1000)
                D,pvalue= ks_2samp(true_rnd_samples, specific_samples, alternative='two-sided', method='auto')
                pvalue_list.append(pvalue)  # no assignment here
            average_pvalue=np.mean(pvalue_list)

            # # Plot the PDF
            # plt.hist(specific_samples, bins=100, density=True, alpha=0.5, edgecolor='black', label='Sample Histogram')
            
            # # Overlay the original PDF
            # plt.plot(k, rnd, 'r-', linewidth=2, label='Risk-Neutral PDF')
            # plt.plot(k,true_rnd,label='true_rnd')
            # # Labels and legend
            # plt.xlabel('Value')
            # plt.ylabel('Density')
            # plt.title('Histogram vs Risk-Neutral PDF')
            # plt.legend()
            # plt.show()
            
            # plt.plot(k,rnd)
            # plt.plot(k,true_rnd)
            
            result_dict[nickname]=average_pvalue
            rnd_iv_dict[nickname]=specific_model_dict

            
            
    #plot function. Loop has ended. We can extract RND/calls/IVs from result_dict to plot if need be. We implement a function later
    plot_overlaid_rnds(true_rnd,rnd_iv_dict,k,run_number)
    return result_dict
        
def local_regression(k,call,iv,r,s,t, argument_dict):
    """
    Performs cross validation to select the best bandwidth for KernelReg.

    Parameters:
        x (np.array): 1D array of the independent variable (e.g. strikes).
        y (np.array): 1D array of the dependent variable (e.g. iv).
        candidate_bw (array): Array of candidate bandwidth values.
        cv_type (str): 'kfold' (default) for KFold CV or 'loo' for Leave-One-Out CV.
        n_splits (int): Number of folds for KFold CV (ignored for LOO).

    Returns:
        best_bw (float): The candidate bandwidth with the lowest average MSE.
        cv_errors (dict): Dictionary mapping candidate bandwidths to their average MSE.
    """
    ######### Extracting relavant information from the option Dataframe

    ############################################################## Program Begins here
    
    bw_setting=argument_dict.get("bw_setting","recommended")
    cv_method = argument_dict.get('cv_method','loo')
    kde_method = argument_dict.get('kde_method',8)
    nickname=argument_dict.get('nickname',"generic local_polynomial")
    
    original_k=k
    original_iv=iv
    
    
    best_bw = None
    best_error = np.inf
    cv_errors = {}

    mask = ~np.isnan(iv)  # True where iv is not NaN

    iv = iv[mask]
    k = k[mask]
    #### Step 1: Preprocess the IV curve.

    if cv_method == 'loo':
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=cv_method, shuffle=True, random_state=42)
    
    
    #Select bandwidth settings 
    if bw_setting=="recommended": #lower bound is the average strike difference
        candidate_bw=np.linspace(np.mean(np.diff(k)),4*max(np.diff(k)),5)
    if isinstance(bw_setting, (int, float)):        #check if bw_setting is a number
        fixed_bw=True
        best_bw=bw_setting
    else:
        fixed_bw=False
        
        
    if fixed_bw==False:
        for bw in candidate_bw:
            errors = []
            for train_idx, test_idx in cv.split(k):
                # Extract the training and test subsets
                k_train, k_test = k[train_idx], k[test_idx]
                iv_train, iv_test = iv[train_idx], iv[test_idx]
            
                # Reshape so each is (n_samples, 1) rather than 1D or scalar
                k_train = k_train.reshape(-1, 1)
                k_test = k_test.reshape(-1, 1)
            
                kr = KernelReg(endog=iv_train, exog=k_train, reg_type="ll", var_type='c', bw=[bw])
            
                # Now strike_test is guaranteed to be 2D
                y_pred, _ = kr.fit(k_test)
            
                errors.append(mean_squared_error(iv_test, y_pred))
                mean_error = np.mean(errors)
            
                # Checking for overfitting or zero estimates
                strike_grid = np.linspace(k[0], k[-1], 25).reshape(-1, 1)
                iv_est, _ = kr.fit(strike_grid)
                if np.any(iv_est == 0):
                    print("At least one value in iv_est is zero. Discarding this bandwidth.")
                    mean_error = np.inf
    
            cv_errors[bw] = mean_error
            
            if mean_error < best_error:
                best_error = mean_error
                best_bw = bw
        kr_final = KernelReg(endog=iv, exog=k, reg_type="ll", var_type='c', bw=[best_bw])
    
    kr_final = KernelReg(endog=iv, exog=k, reg_type="ll", var_type='c', bw=[best_bw])

    interpolated_iv, _ = kr_final.fit(original_k[:, None])
    interpolated_calls=black_scholes_call_from_iv(original_k, s, r, t, interpolated_iv, q=0.0)
    rnd=rnd_from_calls(original_k, interpolated_calls, r, t, clip_zero=True)
    
    
    
    
    # ### Step 3 Apply Weighted Kernel Density Estimation to Risk Neutral density
    # if kde_method=="ISJ":
    #     kde = NaiveKDE(bw=kde_method).fit(original_k,weights=rnd)
    #     kde_pdf=kde.evaluate(k)
    # else:
    #     kde = NaiveKDE(bw="scott").fit(original_k,weights=rnd)
    #     bw=kde.bw/kde_method
    #     kde = NaiveKDE(bw=bw).fit(original_k,weights=rnd)
    #     kde_pdf=kde.evaluate(original_k)
        

    #     plt.plot(original_k,rnd,label="ll rnd")
    #     plt.plot(original_k,true_rnd,label='true rnd')
    #     plt.plot(original_k,kde_pdf,label="kde rnd")
    #     plt.legend()

        
    return_dict={"iv":interpolated_iv,"calls":interpolated_calls,"rnd":rnd,"nickname":nickname}
   

    return return_dict  
 
def local_regression_kde(k,call,iv,r,s,t, argument_dict):
    """
    Performs cross validation to select the best bandwidth for KernelReg.

    Parameters:
        x (np.array): 1D array of the independent variable (e.g. strikes).
        y (np.array): 1D array of the dependent variable (e.g. iv).
        candidate_bw (array): Array of candidate bandwidth values.
        cv_type (str): 'kfold' (default) for KFold CV or 'loo' for Leave-One-Out CV.
        n_splits (int): Number of folds for KFold CV (ignored for LOO).

    Returns:
        best_bw (float): The candidate bandwidth with the lowest average MSE.
        cv_errors (dict): Dictionary mapping candidate bandwidths to their average MSE.
    """
    ######### Extracting relavant information from the option Dataframe

    ############################################################## Program Begins here
    
    bw_setting=argument_dict.get("bw_setting","recommended")
    cv_method = argument_dict.get('cv_method','loo')
    kde_method = argument_dict.get('kde_method',8)
    nickname=argument_dict.get('nickname',"generic local_polynomial_kde")
    
    original_k=k
    original_iv=iv
    
    
    best_bw = None
    best_error = np.inf
    cv_errors = {}

    mask = ~np.isnan(iv)  # True where iv is not NaN

    iv = iv[mask]
    k = k[mask]
    #### Step 1: Preprocess the IV curve.

    if cv_method == 'loo':
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=cv_method, shuffle=True, random_state=42)
    
    
    #Select bandwidth settings 
    if bw_setting=="recommended": #lower bound is the average strike difference
        candidate_bw=np.linspace(np.mean(np.diff(k)),4*max(np.diff(k)),5)
    if isinstance(bw_setting, (int, float)):        #check if bw_setting is a number
        fixed_bw=True
        best_bw=bw_setting
    else:
        fixed_bw=False
        
        
    if fixed_bw==False:
        for bw in candidate_bw:
            errors = []
            for train_idx, test_idx in cv.split(k):
                # Extract the training and test subsets
                k_train, k_test = k[train_idx], k[test_idx]
                iv_train, iv_test = iv[train_idx], iv[test_idx]
            
                # Reshape so each is (n_samples, 1) rather than 1D or scalar
                k_train = k_train.reshape(-1, 1)
                k_test = k_test.reshape(-1, 1)
            
                kr = KernelReg(endog=iv_train, exog=k_train, reg_type="ll", var_type='c', bw=[bw])
            
                # Now strike_test is guaranteed to be 2D
                y_pred, _ = kr.fit(k_test)
            
                errors.append(mean_squared_error(iv_test, y_pred))
                mean_error = np.mean(errors)
            
                # Checking for overfitting or zero estimates
                strike_grid = np.linspace(k[0], k[-1], 25).reshape(-1, 1)
                iv_est, _ = kr.fit(strike_grid)
                if np.any(iv_est == 0):
                    print("At least one value in iv_est is zero. Discarding this bandwidth.")
                    mean_error = np.inf
    
            cv_errors[bw] = mean_error
            
            if mean_error < best_error:
                best_error = mean_error
                best_bw = bw
        kr_final = KernelReg(endog=iv, exog=k, reg_type="ll", var_type='c', bw=[best_bw])
    
    kr_final = KernelReg(endog=iv, exog=k, reg_type="ll", var_type='c', bw=[best_bw])

    interpolated_iv, _ = kr_final.fit(original_k[:, None])
    interpolated_calls=black_scholes_call_from_iv(original_k, s, r, t, interpolated_iv, q=0.0)
    rnd=rnd_from_calls(original_k, interpolated_calls, r, t, clip_zero=True)
    
    
    
    
    ### Step 3 Apply Weighted Kernel Density Estimation to Risk Neutral density
    if kde_method=="ISJ":
        kde = NaiveKDE(bw=kde_method).fit(original_k,weights=rnd)
        kde_pdf=kde.evaluate(k)
    else:
        kde = NaiveKDE(bw="scott").fit(original_k,weights=rnd)
        bw=kde.bw/kde_method
        kde = NaiveKDE(bw=bw).fit(original_k,weights=rnd)
        kde_pdf=kde.evaluate(original_k)
        
        
        # plt.plot(original_k,rnd,label="ll rnd")
        # plt.plot(original_k,true_rnd,label='true rnd')
        # plt.plot(original_k,kde_pdf,label="kde rnd")
        # plt.legend()

    
    repriced_calls=calls_from_rnd(kde_pdf, original_k, s, r, t, normalize=False)
    
    repriced_iv=implied_vol_from_calls(repriced_calls, s, original_k, r, t, q=0.0, tol=1e-8, max_iter=100, vol_low=1e-8, vol_high=5.0)
    
    # plt.plot(original_k,repriced_calls)
    # plt.plot(original_k,interpolated_calls)
    
    return_dict={"iv":repriced_iv,"calls":repriced_calls,"rnd":kde_pdf,"nickname":nickname}
   

    return return_dict   
        
        
def plot_overlaid_rnds(true_rnd, rnd_iv_dict, k,run_number, normalize=True,
                       save_filename="comparison_rnds.png"):
    """
    Plot the true risk-neutral density (true_rnd) and all RNDs in rnd_iv_dict overlaid vs strikes k,
    and save the figure to 'comparison_rnd_folder' in the current directory.

    Parameters
    ----------
    true_rnd : array_like
        The true RND values aligned with k.
    rnd_iv_dict : dict
        Dict like {
            'model_a': {'rnd': np.array, 'iv': ..., 'calls': ..., 'nickname': '...'},
            'model_b': {...},
            ...
        }
    k : array_like
        1-D array of strikes aligned with each rnd.
    normalize : bool, optional
        If True, scale each RND to integrate to 1 on k.
    save_filename : str, optional
        The name of the saved file (default 'comparison_rnds.png').
    """
    k = np.asarray(k)

    plt.figure(figsize=(9, 5))

    # Plot the true RND
    mask = np.isfinite(k) & np.isfinite(true_rnd)
    kk_true, rr_true = k[mask], np.asarray(true_rnd)[mask]
    order = np.argsort(kk_true)
    kk_true, rr_true = kk_true[order], rr_true[order]

    if normalize:
        area = np.trapz(rr_true, kk_true)
        if np.isfinite(area) and area > 0:
            rr_true = rr_true / area

    plt.plot(kk_true, rr_true, 'k--', linewidth=2.5, label='True RND')

    # Plot model RNDs
    for key, d in (rnd_iv_dict.items() if isinstance(rnd_iv_dict, dict)
                   else enumerate(rnd_iv_dict)):
        rnd = np.asarray(d.get('rnd'))
        if rnd is None:
            continue

        # Align lengths and drop non-finite values
        n = min(len(k), len(rnd))
        kk = k[:n]
        rr = rnd[:n]
        mask = np.isfinite(kk) & np.isfinite(rr)
        kk, rr = kk[mask], rr[mask]

        # Sort by strike
        order = np.argsort(kk)
        kk, rr = kk[order], rr[order]

        # Optional normalization
        if normalize:
            area = np.trapz(rr, kk)
            if np.isfinite(area) and area > 0:
                rr = rr / area

        label = d.get('nickname') or (key if isinstance(key, str) else f"model_{key}")
        plt.plot(kk, rr, linewidth=2, label=label)

    plt.xlabel('Strike (k)')
    plt.ylabel('Density')
    plt.title('Overlaid Risk-Neutral Densities')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    # Create save folder if not exists
    save_filename=f"compared_rnd_{run_number}"
    save_dir = os.path.join(os.getcwd(), "comparison_rnd_folder")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Figure saved to: {save_path}")
    return None



