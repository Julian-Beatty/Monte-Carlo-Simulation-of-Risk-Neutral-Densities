from main_simulate_densities import*




svi_dict={
##Option parameters
    "strikes":np.arange(25,200,1),
    "underlying_price": 100,
    "risk_free_rate": 0.03,   
    "maturity": 1,
##Model Parameters    
    "a": 0,
    "b": 0.1,
    "rho":0.35,
    "m": 0.01,
    "sigma":0.2,
    }

models_dict={
    "local_regression":{"nickname":"local reg. cv","bandwidth_setting":"recommended","cv_method":"loo"},
    "local_regression_kde":{"nickname":"local reg_kde_8","bandwidth_setting":"recommended","cv_method":"loo","kde_method":8},
    }

monte_test=monte_carlo(svi_dict)
resultsz=monte_test.simulate_mc(svi_dict,models_dict,2)