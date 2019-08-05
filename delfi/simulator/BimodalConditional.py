import delfi.distribution as dd
import numpy as np

from delfi.simulator.BaseSimulator import BaseSimulator


class BimodalConditional(BaseSimulator):
    def __init__(self, dim=2, var=1.0, corr=0.95, bias=1.0, seed=None):
        """Creates a bimodal gaussian that is V-shaped

        Parameters
        ----------
        dim : int
            Number of dimensions of parameters
        var : float
            spread along most variable axis of the gaussians
        corr: float, 0 < corr < 1
            the higher the more narrow the gaussian along the less variable axis
        bias: float
            offset of the mean of the gaussians from the center
        seed : int or None
            If set, randomness is seeded
        """
        assert dim==2, 'Toy model only implemented in 2D. Therefore dim==2 is required.'
        super().__init__(dim_param=dim, seed=seed)
        diago = var*corr
        single_noise_cov     = [[var,  diago], [ diago, var]]
        single_noise_cov_neg = [[var, -diago], [-diago, var]]
        self.noise_cov = [single_noise_cov, single_noise_cov_neg]
        self.bias = bias
        self.a = [0.5, 0.5]  # mixture weights

    @copy_ancestor_docstring
    def gen_single(self, param):
        # See BaseSimulator for docstring
        param = np.asarray(param).reshape(-1)
        assert param.ndim == 1
        assert param.shape[0] == self.dim_param

        mss = [param + [0.0, -self.bias], param + [0.0, self.bias]]
        sample = dd.MoG(a=self.a, ms=mss,
                        Ss=self.noise_cov, seed=self.gen_newseed()).gen(1)

        return {'data': sample.reshape(-1)}
