import sys
import os
import os.path
core_path = './core/'
sys.path.insert(0, core_path)
import numpy as np
import prop_store
import likelihood_grad
import autodiff as ad
import goodies
import scipy
import scipy.optimize
import scipy.stats
import tqdm
import bisect
import functools

np.seterr(all='raise')

### How to build an analysis
## Instantiate the store
#
# the_store = prop_store.store()
#
## The store computes values for you and caches the result so you don't compute anything twice.
## If you need a computed value, you go to the store:
#
# my_expensive_numbers = the_store.get_prop("expensive_numbers", physics_parameters)
#
## But first you have to tell the store how to do things.
## So you define a function:
#
# def my_func(foo, bar, physics_param):
#     ret = ...
#     ... compute things ...
#     return ret
#
## Once defined, you must register the function with the store.
## This requires a name for the output of the function,
## the ordered names of parameters it depends on,
## and the function itself.
#
# the_store.add_prop("my_value", ["value_of_foo", "value_of_bar", "physics_value"], my_func)
#
## Now you can register other functions that use the output and so on...
#
# def my_other_func(my_value):
#     return (my_value + 1)/2.0
# the_store.add_prop("my_other_value", ["my_value"], my_other_func)
#
## This implicitly depends on "physics_value", but that is handled by the store.
## Just please don't define an infinite loop via interdependency...
#
## Finally you have to initialize the store so it can work out the implicit dependencies of the
## things you defined, figure out what the physics parameters are, and spin up the object caches
#
# the_store.initialize()
#
## If you are reinitializing the store after adding props and you want to keep the caches
#
# the_store.initialize(keep_cache=True)
#
## Now you can ask the store for values as long as you give it the appropriate physics parameters
## It will work out all the details and try not to recompute anything if it can help it
#
# physics_parameters = {"physics_value": np.pi/4.}
# value = the_store.get_prop("my_value", physics_parameters)

# The inverse of the covariance matrix
def cov_inv(cov):
    return np.linalg.inv(cov)

# The normalization term in the gaussian prior
def gauss_prefactor(cov):
    return -0.5*np.log((2*np.pi)**cov.shape[0]*np.linalg.det(cov))

# The exponential term in the gaussian prior
def gauss_exponenent(scale_factors, center, cov_inv, mc_error=None):
    alpha = np.array(scale_factors)
    beta = center
    diff = alpha - beta
    res = -np.sum(diff[:,None]*diff[None,:]*cov_inv)

    dalpha = np.identity(len(scale_factors))
    res_d = -np.sum((dalpha[:,:,None]*diff[None,None,:] + dalpha[:,None,:]*diff[None,:,None])*cov_inv[None,:,:], axis=(1,2))
    res_d = np.concatenate([res_d, np.zeros(res_d.shape)])
    return np.concatenate([[res], res_d]).T

def add_gaussian_component(the_store, param_def):
    central_value = param_def["center"]
    cov = param_def["cov"]
    name = param_def["component_name"]
    parameter_name = param_def["parameter_name"]
    the_store.add_prop(name + "_center", [], lambda: central_value)
    the_store.add_prop(name + "_cov", [], lambda: cov)
    the_store.add_prop(name + "_cov_inv", [name + "_cov"], cov_inv)
    the_store.add_prop(name + "_gauss_prefactor", [name + "_cov"], gauss_prefactor)
    the_store.add_prop(name + "_gauss_exponent", [parameter_name, name + "_center", name + "_cov_inv"], gauss_exponenent)

def gaussian_component(the_store, component_name, central_value, cov=None, f_cov=None, mc_error=None, f_mc_error=None):
    if cov is None and f_cov is None:
        raise ValueError("Need one of: cov, f_cov")
    central_value = np.asarray(central_value)
    if cov is None:
        cov = np.asarray(cov)
        cov = f_cov * central_value[None, :] * central_value[:, None]

    name = component_name
    param_def = {
            "component_name": name,
            "parameter_name": name+"_scale_factors",
            "format": "array",
            "shape": np.shape(central_value),
            "center": central_value,
            "prior": "gaussian",
            "cov": cov,
            }
    if f_mc_error is not None:
        param_def["relative_mc_error"] = f_mc_error
    elif mc_error is not None:
        param_def["mc_error"] = mc_error
    return param_def


def unconstrained_component(component_name, n_bins, mc_error=None, f_mc_error=None):
    param_def = {
            "component_name": component_name,
            "parameter_name": name + "_scale_factors",
            "format": "array",
            "shape": (n_bins,),
            "prior": "none",
            }
    if f_mc_error is not None:
        param_def["relative_mc_error"] = f_mc_error
    elif mc_error is not None:
        param_def["mc_error"] = mc_error
    return param_def

def unconstrained_positive_component(component_name, n_bins, mc_error=None, f_mc_error=None):
    param_def = {
            "component_name": component_name,
            "parameter_name": name + "_scale_factors",
            "format": "array",
            "shape": (n_bins,),
            "prior": "positive",
            }
    if f_mc_error is not None:
        param_def["relative_mc_error"] = f_mc_error
    elif mc_error is not None:
        param_def["mc_error"] = mc_error
    return param_def

def normalization_component(component_name, central_value, mc_error=None, f_mc_error=None):
    param_def = {
            "component_name": component_name,
            "parameter_name": name + "_normalization",
            "format": "scalar",
            "shape": tuple(),
            "prior": "positive",
            }
    if f_mc_error is not None:
        param_def["relative_mc_error"] = f_mc_error
    elif mc_error is not None:
        param_def["mc_error"] = mc_error
    return param_def

exit(0)

def setup_analysis(tag, include_nue=True, include_numu=True, include_lee=True, combine_sm=False):
    the_store = prop_store.store()
    goodies = get_goodies(tag)
    nue_sm_expect, lee_expect, nue_mc_error, nue_f_cov, numu_expect, numu_f_cov = goodies

    nue_param_def = gaussian_component(the_store, "intrinsic_nue", nue_sm_expect, f_cov=nue_f_cov)
    lee_param_def = unconstrained_positive_component("low_energy_excess")
    if numu_expect is not None and numu_f_cov is not None:
        numu_param_def = gaussian_component(the_store, "intrinsic_numu", numu_expect, f_cov=numu_f_cov)
    elif numu_expect is not None and numu_f_cov is None:
        numu_param_def = normalization_component("intrinsic_numu", numu_expect)
    else:
        numu_param_def = None
    param_defs = [nue_param_def, lee_param_def, numu_param_def]

    add_likelihood(the_store, param_defs, llh_type=None)

    # Just the Standard Model central value
    def sm():
        w = np.array([])
        return w
    the_store.add_prop("sm", [], sm)

    # Just the Standard Model + Low Energy Excess central value
    def lee():
        w = np.array([])
        return w
    the_store.add_prop("lee", [], lee)

    # The center of the gaussian prior
    def center(sm):
        return sm
    the_store.add_prop("center", ["sm"], center)

    # The expected distribution of events based on the free parameters
    def expect(scale_factors, lee_scale_factors):
        scale_factors = np.asarray(scale_factors)
        lee_scale_factors = np.asarray(lee_scale_factors)
        res = scale_factors + lee_scale_factors
        res_d = np.concatenate([np.identity(len(scale_factors))]*2)
        return np.concatenate([[res], res_d]).T
    the_store.add_prop("expect", ["scale_factors", "lee_scale_factors"], expect)
    the_store.add_prop("asimov_expect", ["scale_factors", "lee_scale_factors"], expect)

    # The relative variance from MC statistical fluctuations
    def relative_mc_stat_var():
        return np.full(10, 0.05**2)
    the_store.add_prop("relative_mc_stat_var", [], relative_mc_stat_var)

    # The MC statistical error term
    def expect_sq(scale_factors, relative_mc_stat_var):
        res = relative_mc_stat_var * scale_factors * scale_factors
        res_d = np.diag(2*relative_mc_stat_var*scale_factors)
        res_d = np.concatenate([res_d, np.zeros(res_d.shape)])
        return np.concatenate([[res], res_d]).T
    the_store.add_prop("expect_sq", ["scale_factors", "relative_mc_stat_var"], expect_sq)
    the_store.add_prop("asimov_expect_sq", ["scale_factors", "relative_mc_stat_var"], expect_sq)

    # The systematics fractional covariance matrix
    def f_cov():
        return np.array([])
    the_store.add_prop("f_cov", [], f_cov)

    # The covariance matrix
    # Equal to the fractional covariance matrix times the two means corresponding to each matrix entry
    def cov(f_cov, center):
        return f_cov * center[:,None] * center[None,:]
    the_store.add_prop("cov", ["f_cov", "center"], cov)

    # The inverse of the covariance matrix
    def cov_inv(cov):
        return np.linalg.inv(cov)
    the_store.add_prop("cov_inv", ["cov"], cov_inv)

    # The normalization term in the gaussian prior
    def gauss_prefactor(cov):
        return -0.5*np.log((2*np.pi)**cov.shape[0]*np.linalg.det(cov))
    the_store.add_prop("gauss_prefactor", ["cov"], gauss_prefactor)

    # The exponential term in the gaussian prior
    def gauss_exponenent(scale_factors, center, cov_inv):
        alpha = np.array(scale_factors)
        beta = center
        diff = alpha - beta
        res = -np.sum(diff[:,None]*diff[None,:]*cov_inv)

        dalpha = np.identity(len(scale_factors))
        res_d = -np.sum((dalpha[:,:,None]*diff[None,None,:] + dalpha[:,None,:]*diff[None,:,None])*cov_inv[None,:,:], axis=(1,2))
        res_d = np.concatenate([res_d, np.zeros(res_d.shape)])
        return np.concatenate([[res], res_d]).T
    the_store.add_prop("gauss_exponent", ["scale_factors", "center", "cov_inv"], gauss_exponenent)

    # The gaussian prior
    def gauss(gauss_prefactor, gauss_exponenent):
        res = np.copy(gauss_exponenent)
        res[0] += gauss_prefactor
        return res
    the_store.add_prop("gauss", ["gauss_prefactor", "gauss_exponent"], gauss)

    # Spin up the caches
    the_store.initialize(keep_cache=True)

    return the_store

# Compute the alpha quantile of a weighted set of elements
# The default alpha=0.5 returns the median
def weighted_median(quantity, weights=None, alpha=0.5):
    quantity = np.asarray(quantity)
    order = np.argsort(quantity)
    sorted_q = quantity[order]
    if weights is None:
        sorted_w = np.full(np.shape(quantity), 1.0/len(quantity))
        cumulative_w = np.linspace(0.0,1.0, len(quantity)+1)
    else:
        weights = np.asarray(weights)
        sorted_w = weights[order]
        total = np.sum(weights)
        cumulative_w = np.cumsum(sorted_w) / total

    i = bisect.bisect_left(cumulative_w, alpha) - 1
    if i < 0 or i >= len(quantity):
        return None
    return (sorted_q[i]*sorted_w[i]*(1.0 - alpha) + sorted_q[i+1]*sorted_w[i+1]*(alpha))/(sorted_w[i]*(1.0 - alpha) + sorted_w[i+1]*(alpha))

# Compute the "sigma" from a p-value using the 2-tailed normal distribution definition
def get_sigma(p):
    proportion = 1. - p
    sigma = np.sqrt(2) * scipy.special.erfinv(proportion)
    return sigma

# Compute the p-value from a "sigma" using the 2-tailed normal distribution definition
def get_p(sigma):
    proportion = scipy.special.erf(sigma / np.sqrt(2))
    p = 1. - proportion
    return p

if __name__ == "__main__":
    the_store = setup_analysis()

    # Now we can define the likelihood

    def likelihood(data, params):
        # Get the MC expectation and error terms with their gradients
        expect = the_store.get_prop("expect", params)
        expect_sq = the_store.get_prop("expect_sq", params)

        # Compute the likelihood and its gradient (requires a transformation between gradient representations via ad.unpack)
        # The sum is over the likelihood in each bin (the use of ad.sum is essential for treading the gradient representation correctly)
        say_likelihood = ad.sum(likelihood_grad.LEff(data, ad.unpack(expect), ad.unpack(expect_sq)))
        assert(np.all(say_likelihood[0] <= 0))

        # Compute the gaussian prior with its gradient
        gauss = the_store.get_prop("gauss", params)

        # Add the two terms
        like = ad.plus_grad(say_likelihood, ad.unpack(gauss[None,:]))

        return ad.mul(like, -1.0)

    def asimov_likelihood(asimov_parameters, parameters):
        asimov_expect = the_store.get_prop("asimov_expect", asimov_parameters)[:,0]
        return likelihood(asimov_expect, asimov_parameters)

    def data_likelihood(binned_data, parameters):
        return likelihood(binned_data, parameters)

    # Get the central values
    sm_mu = the_store.get_prop("sm", {})
    lee_mu = the_store.get_prop("lee", {})
    sigma2_over_mu2 = the_store.get_prop("relative_mc_stat_var", {})
    sm_center = tuple(np.array(the_store.get_prop("sm", {})).tolist())
    lee_center = tuple((np.array(the_store.get_prop("lee", {})) - np.array(the_store.get_prop("sm", {}))).tolist())

    n_bins = len(sm_center)

    # Parameter sets that correspond to the central values
    lee_params = {
        "scale_factors": sm_center,
        "lee_scale_factors": lee_center,
    }

    asimov_params = lee_params

    def f_sm(x):
        params = {
            "scale_factors": tuple(np.array(x).tolist()),
            "lee_scale_factors": tuple(np.zeros(np.shape(x)).tolist()),
            }
        v, g = asimov_likelihood(asimov_params, params)
        return v, g[0,:n_bins]

    seed = sm_center
    res_sm = scipy.optimize.minimize(f_sm, seed, bounds=[[0,np.inf]]*len(seed), method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-8, 'maxls': 20}, jac=True)

    asimov_chi2 = res_sm.fun

    sm_data_sm_chi2 = []
    sm_data_lee_chi2 = []
    lee_data_sm_chi2 = []
    lee_data_lee_chi2 = []
    last_save = 0
    for i in tqdm.tqdm(range(20000)):

        # gamma --> cov --> poisson
        alpha = 1.0 / sigma2_over_mu2 + 1
        scale = sm_mu * sigma2_over_mu2
        mu = scipy.stats.gamma.rvs(alpha, scale=scale)
        cov = the_store.get_prop("f_cov", {}) * mu[:,None] * mu[None,:]
        mu_mc = None
        while mu_mc is None or np.any(mu_mc < 0):
            mu_mc = scipy.stats.multivariate_normal.rvs(mu, cov=cov)
        sm_data = scipy.stats.poisson.rvs(mu_mc)

        #alpha = lee_mu**2 / (sigma2_over_mu2*sm_mu**2) + 1
        #scale = sigma2_over_mu2*sm_mu**2 / lee_mu
        alpha = 1.0 / sigma2_over_mu2 + 1
        scale = lee_mu * sigma2_over_mu2
        mu = scipy.stats.gamma.rvs(alpha, scale=scale)
        cov = the_store.get_prop("f_cov", {}) * mu[:,None] * mu[None,:]
        mu_mc = None
        while mu_mc is None or np.any(mu_mc < 0):
            mu_mc = scipy.stats.multivariate_normal.rvs(mu, cov=cov)
        lee_data = scipy.stats.poisson.rvs(mu_mc)


        #### SM data
        def f_sm(x):
            x = np.asarray(x)
            params = {
                "scale_factors": tuple(x.tolist()),
                "lee_scale_factors": tuple([0.0]*len(x)),
                }
            v, g = data_likelihood(sm_data, params)
            return v, g[0,:n_bins]

        seed = sm_data
        res_sm = scipy.optimize.minimize(f_sm, seed, bounds=[[0,np.inf]]*len(seed), method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-8, 'maxls': 20}, jac=True)
        sm_data_sm_chi2.append(res_sm.fun[0])

        def f_lee(x):
            x = np.asarray(x)
            params = {
                    "scale_factors": tuple(x[:n_bins].tolist()),
                    "lee_scale_factors": tuple(x[n_bins:].tolist()),
                }
            v, g = data_likelihood(sm_data, params)
            return v, g

        seed = np.concatenate([sm_center, sm_data - sm_mu])
        seed = np.amax([seed, np.zeros(np.shape(seed))], axis=0)
        res_lee = scipy.optimize.minimize(f_lee, seed, bounds=[[0,np.inf]]*len(seed), method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-8, 'maxls': 20}, jac=True)
        sm_data_lee_chi2.append(res_lee.fun[0])

        #### LEE data
        def f_sm(x):
            params = {
                "scale_factors": tuple(np.array(x).tolist()),
                "lee_scale_factors": tuple([0.0]*len(x)),
                }
            v,g = data_likelihood(lee_data, params)
            return v, g[0,:n_bins]

        seed = lee_data
        res_sm = scipy.optimize.minimize(f_sm, seed, bounds=[[0,np.inf]]*len(seed), method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-8, 'maxls': 20}, jac=True)
        lee_data_sm_chi2.append(res_sm.fun[0])

        def f_lee(x):
            x = np.asarray(x)
            params = {
                    "scale_factors": tuple(x[:n_bins].tolist()),
                    "lee_scale_factors": tuple(x[n_bins:].tolist()),
                }
            v, g = data_likelihood(lee_data, params)
            return v, g

        seed = np.concatenate([sm_center, lee_data - sm_mu])
        seed = np.amax([seed, np.zeros(np.shape(seed))], axis=0)
        res_lee = scipy.optimize.minimize(f_lee, seed, bounds=[[0,np.inf]]*len(seed), method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-8, 'maxls': 20}, jac=True)
        lee_data_lee_chi2.append(res_lee.fun[0])

        if len(sm_data_sm_chi2) % 100 == 0:
            ### Chi2 test
            median_sm_chi2 = weighted_median(sm_data_sm_chi2, alpha=0.5)
            median_lee_chi2 = weighted_median(lee_data_sm_chi2, alpha=0.5)
            ats_dist = np.array(sm_data_sm_chi2)
            print("Median SM Chi2:", median_sm_chi2)
            print("Median LEE Chi2:", median_lee_chi2)
            print("p-value")
            p_value = float(np.count_nonzero(ats_dist > median_lee_chi2)) / float(len(ats_dist))
            print("%.16f" % p_value)
            print("sigma")
            print(get_sigma(p_value))
            print()
            ### TS test
            sm_sm = np.array(sm_data_sm_chi2)
            sm_lee = np.array(sm_data_lee_chi2)
            lee_sm = np.array(lee_data_sm_chi2)
            lee_lee = np.array(lee_data_lee_chi2)
            sm_ts = sm_sm - sm_lee
            lee_ts = lee_sm - lee_lee
            median_sm_ts = weighted_median(sm_ts, alpha=0.5)
            median_lee_ts = weighted_median(lee_ts, alpha=0.5)
            print("Median SM TS:", median_sm_ts)
            print("Median LEE TS:", median_lee_ts)
            print("p-value")
            p_value = float(np.count_nonzero(sm_ts > median_lee_ts)) / float(len(sm_ts))
            print("%.16f" % p_value)
            print("sigma")
            print(get_sigma(p_value))
            print()

