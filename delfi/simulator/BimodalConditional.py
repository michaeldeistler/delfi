import delfi.distribution as dd
import numpy as np

from delfi.simulator.BaseSimulator import BaseSimulator


class BimodalConditional(BaseSimulator):
    def __init__(self, dim=1, var=1.0, corr=0.95, quad_modal=False, bias_quad=1.0, seed=None):
        """Gaussian Mixture simulator

        Toy model that draws data from a mixture distribution with 2 components
        that have mean theta and fixed noise.

        Parameters
        ----------
        dim : int
            Number of dimensions of parameters
        noise_cov : list
            Covariance of noise on observations
        seed : int or None
            If set, randomness is seeded
        """
        super().__init__(dim_param=dim, seed=seed)
        diago = var*corr
        single_noise_cov     = [[var,  diago], [ diago, var]]
        single_noise_cov_neg = [[var, -diago], [-diago, var]]
        self.noise_cov = [single_noise_cov, single_noise_cov_neg]
        self.quad_modal = quad_modal
        if self.quad_modal:
            self.noise_cov = [single_noise_cov, single_noise_cov_neg, single_noise_cov_neg, single_noise_cov]
        self.bias_quad = bias_quad
        if self.quad_modal:
            self.a = [0.25, 0.25, 0.25, 0.25]
        else:
            self.a = [0.5, 0.5]  # mixture weights

    @copy_ancestor_docstring
    def gen_single(self, param):
        # See BaseSimulator for docstring
        param = np.asarray(param).reshape(-1)
        assert param.ndim == 1
        assert param.shape[0] == self.dim_param

        mss = [param, param, param + [self.bias_quad, -self.bias_quad], param + [self.bias_quad, self.bias_quad]]
        if self.quad_modal:
            sample = dd.MoG(a=self.a, ms=mss, Ss = self.noise_cov, seed = self.gen_newseed()).gen(1)
        else:
            mss = [param + [0.0, -self.bias_quad], param + [0.0, self.bias_quad]]
            sample = dd.MoG(a=self.a, ms=mss,
                            Ss=self.noise_cov, seed=self.gen_newseed()).gen(1)

        return {'data': sample.reshape(-1)}
