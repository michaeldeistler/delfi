import numpy as np
import theano
import delfi.distribution as dd
from delfi.inference.BaseInference import BaseInference
from delfi.neuralnet.Trainer import Trainer, ActiveTrainer
from delfi.neuralnet.loss.regularizer import svi_kl_init, svi_kl_zero
from delfi.neuralnet.loss.lossfunc import \
    (snpe_loss_prior_as_proposal, apt_loss_gaussian_proposal,
     apt_loss_MoG_proposal, apt_loss_atomic_proposal)
from delfi.neuralnet.NeuralNet import dtype
from delfi.utils.data import repnewax, combine_trn_datasets
from delfi.utils.box_flow import gen_bounded_theta, calc_leakage
from copy import deepcopy

class APT(BaseInference):
    def __init__(self, generator, obs=None, prior_norm=False, leakage_thr=0.1,
                 pilot_samples=100, reg_lambda=0.01, seed=None, verbose=True,
                 boxMAF_MLE=False, **kwargs):
        """APT
        Core idea is to parameterize the true posterior, and calculate the
        proposal posterior as needed on-the-fly.

        Parameters
        ----------
        generator : generator instance
            Generator instance
        obs : array
            Observation in the format the generator returns (1 x n_summary)
        prior_norm : bool
            If set to True, will z-transform params based on mean/std of prior
        pilot_samples : None or int
            If an integer is provided, a pilot run with the given number of
            samples is run. The mean and std of the summary statistics of the
            pilot samples will be subsequently used to z-transform summary
            statistics.
        n_components : int
            Number of components in final round (PM's algorithm 2)
        reg_lambda : float
            Precision parameter for weight regularizer if svi is True
        seed : int or None
            If provided, random number generator will be seeded
        verbose : bool
            Controls whether or not progressbars are shown
        kwargs : additional keyword arguments
            Additional arguments for the NeuralNet instance, including:
                n_hiddens : list of ints
                    Number of hidden units per layer of the neural network
                svi : bool
                    Whether to use SVI version of the network or not

        Attributes
        ----------
        observables : dict
            Dictionary containing theano variables that can be monitored while
            training the neural network.
        """
        assert obs is not None, "APT requires observed data"
        self.obs = np.asarray(obs)
        super().__init__(generator, prior_norm=prior_norm,
                         pilot_samples=pilot_samples, seed=seed,
                         verbose=verbose, **kwargs)  # initializes network
        assert 0 < self.obs.ndim <= 2
        if self.obs.ndim == 1:
            self.obs = self.obs.reshape(1, -1)
        assert self.obs.shape[0] == 1

        if np.any(np.isnan(self.obs)):
            raise ValueError("Observed data contains NaNs")

        self.reg_lambda = reg_lambda
        self.leakage_thr = leakage_thr
        self.boxMAF_MLE = boxMAF_MLE
        self.exception_info = (None, None, None)
        self.trn_datasets, self.proposal_used = [], []

    def define_loss(self, n, round_cl=1, proposal='gaussian',
                    combined_loss=False):
        """Loss function for training

        Parameters
        ----------
        n : int
            Number of training samples
        round_cl : int
            Round after which to start continual learning
        proposal : str
            Specifier for type of proposal used: continuous ('gaussian', 'mog')
            or 'atomic' proposals are implemented.
        combined_loss : bool
            Whether to include prior likelihood terms in addition to atomic
        """
        if proposal == 'prior':  # using prior as proposal
            loss, trn_inputs = snpe_loss_prior_as_proposal(self.network,
                                                           svi=self.svi)
        elif proposal == 'gaussian':
            assert isinstance(self.generator.proposal, dd.Gaussian)
            loss, trn_inputs = apt_loss_gaussian_proposal(self.network,
                                                          self.generator.prior,
                                                          svi=self.svi)
        elif proposal.lower() == 'mog':
            loss, trn_inputs = apt_loss_MoG_proposal(self.network,
                                                     self.generator.prior,
                                                     svi=self.svi)
        elif proposal == 'atomic':
            loss, trn_inputs = \
                apt_loss_atomic_proposal(self.network, svi=self.svi,
                                         combined_loss=combined_loss)
        else:
            raise NotImplemented()

        # adding nodes to dict s.t. they can be monitored during training
        self.observables['loss.lprobs'] = self.network.lprobs
        self.observables['loss.raw_loss'] = loss

        if self.svi:
            if self.round <= round_cl:
                # weights close to zero-centered prior in the first round
                if self.reg_lambda > 0:
                    kl, imvs = svi_kl_zero(self.network.mps, self.network.sps,
                                           self.reg_lambda)
                else:
                    kl, imvs = 0, {}
            else:
                # weights close to those of previous round
                kl, imvs = svi_kl_init(self.network.mps, self.network.sps)

            loss = loss + 1 / n * kl

            # adding nodes to dict s.t. they can be monitored
            self.observables['loss.kl'] = kl
            self.observables.update(imvs)

        return loss, trn_inputs

    def run(self, n_rounds=1, proposal='gaussian', silent_fail=True,
            **kwargs):
        """Run algorithm
        Parameters
        ----------
        trn_data: list
            tuple of summstats and paramters. If provided, training data is
            taken instead of generated
        n_train : int or list of ints
            Number of data points drawn per round. If a list is passed, the
            nth list element specifies the number of training examples in the
            nth round. If there are fewer list elements than rounds, the last
            list element is used.
        n_rounds : int
            Number of rounds
        proposal : str
            Specifier for type of proposal used: continuous ('gaussian', 'mog')
            or 'atomic' proposals are implemented.
        boxMAF_MLE: bool
            trains a new normalizing flow after each round using Maximum like-
            lihood. Leads to bounded MAFs and solves the issue of leaking mass
        epochs : int
            Number of epochs used for neural network training
        minibatch : int
            Size of the minibatches used for neural network training
        monitor : list of str
            Names of variables to record during training along with the value
            of the loss function. The observables attribute contains all
            possible variables that can be monitored
        round_cl : int
            Round after which to start continual learning
        stop_on_nan : bool
            If True, will halt if NaNs in the loss are encountered
        silent_fail : bool
            If true, will continue without throwing an error when a round fails
        kwargs : additional keyword arguments
            Additional arguments for the Trainer instance

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged while training the networks
        trn_datasets : list of (params, stats)
            training datasets, z-transformed
        posteriors : list of distributions
            posterior after each round
        """
        # support 'discrete' instead of 'atomic' for backwards compatibility
        if proposal == 'discrete':
            proposal = 'atomic'
        elif proposal == 'discrete_comb':
            proposal = 'atomic_comb'

        logs = []
        trn_datasets = []
        logs_MLE = []
        trn_datasets_MLE = []
        posteriors = []
        posteriors_MLE = []

        if 'train_on_all' in kwargs.keys() and kwargs['train_on_all'] is True:
            kwargs['round_cl'] = np.inf
            if proposal == 'gaussian' and self.network.n_components > 1 and \
                    'reuse_prior_samples' not in kwargs.keys():
                # prevent numerical instability (broad unused comps)
                kwargs['reuse_prior_samples'] = False

        for r in range(n_rounds):
            self.round += 1

            if silent_fail:
                try:
                    log, trn_data = self.run_round(proposal, **kwargs)
                except:
                    print('Round {0} failed'.format(self.round))
                    import sys
                    self.exception_info = sys.exc_info()
                    break
            else:
                log, trn_data = self.run_round(proposal, **kwargs)

            logs.append(log)
            trn_datasets.append(trn_data)
            posteriors.append(self.predict(self.obs))

            if self.boxMAF_MLE and self.round > 1:
                leakage = calc_leakage(self.network.cmaf, self.generator.prior, self.obs, n_samples=10000)
                if leakage > self.leakage_thr:
                    print('Leakage of', int(leakage*100), '% detected after round', self.round, '! Retraining network using MLE.')
                    log_MLE, trn_data_MLE = self.run_MLE(**kwargs)
                    logs_MLE.append(log_MLE)
                    trn_datasets_MLE.append(trn_data_MLE)
                    posteriors_MLE.append(self.predict(self.obs))

        return logs, trn_datasets, posteriors, logs_MLE, trn_datasets_MLE, posteriors_MLE

    def run_round(self, proposal=None, **kwargs):

        self.proposal_used.append(proposal if self.round > 1 else 'prior')

        if proposal == 'prior' or self.round == 1:
            return self.run_prior(**kwargs)
        elif proposal == 'gaussian':
            return self.run_gaussian(**kwargs)
        elif proposal.lower() == 'mog':
            return self.run_MoG(**kwargs)
        elif proposal == 'atomic':
            return self.run_atomic(combined_loss=False, **kwargs)
        elif proposal == 'atomic_comb':
            return self.run_atomic(combined_loss=True, **kwargs)
        else:
            raise NotImplemented()

    def run_prior(self, trn_data=None, n_train=100, epochs=100, minibatch=50, n_atoms=None, moo=None, train_on_all=False, round_cl=1,
                  stop_on_nan=False, monitor=None, verbose=False, print_each_epoch=False, reuse_prior_samples=True, **kwargs):

        # simulate data
        self.generator.proposal = self.generator.prior

        # if training data is provided, load data. Otherwise generate data.
        if trn_data is not None:
            trn_data, n_train_round = self.set_trn_data(trn_data, n_train)
        else:
            trn_data, n_train_round = self.gen(n_train)
        self.trn_datasets.append(trn_data)

        if train_on_all and reuse_prior_samples:
            prior_datasets = [d for i, d in enumerate(self.trn_datasets)
                              if self.proposal_used[i] == 'prior']
            trn_data = combine_trn_datasets(prior_datasets)
            n_train_round = trn_data[0].shape[0]

        # train network
        self.loss, trn_inputs = self.define_loss(n=n_train_round,
                                                 round_cl=round_cl,
                                                 proposal='prior')
        t = Trainer(self.network,
                    self.loss,
                    trn_data=trn_data, trn_inputs=trn_inputs,
                    seed=self.gen_newseed(),
                    monitor=self.monitor_dict_from_names(monitor),
                    **kwargs)
        log = t.train(epochs=self.epochs_round(epochs), minibatch=minibatch, verbose=verbose,
                      print_each_epoch=print_each_epoch, stop_on_nan=stop_on_nan)

        return log, trn_data

    def run_gaussian(self, trn_data=None, n_train=100, epochs=100, minibatch=50, n_atoms=None, moo=None,  train_on_all=False,
                     round_cl=1, stop_on_nan=False, monitor=None, verbose=False, print_each_epoch=False,
                     reuse_prior_samples=True, **kwargs):

        # simulate data
        self.set_proposal(project_to_gaussian=True)
        prop = self.generator.proposal
        assert isinstance(prop, dd.Gaussian)
        # if training data is provided, load data. Otherwise generate data.
        if trn_data is not None:
            trn_data, n_train_round = self.set_trn_data(trn_data, n_train)
        else:
            trn_data, n_train_round = self.gen(n_train)

        # here we're just repeating the same fixed proposal, though we
        # could also introduce some variety if we wanted.
        prop_m = np.expand_dims(prop.m, 0).repeat(n_train_round, axis=0)
        prop_P = np.expand_dims(prop.P, 0).repeat(n_train_round, axis=0)
        trn_data = (*trn_data, prop_m, prop_P)
        self.trn_datasets.append(trn_data)

        if train_on_all:
            prev_datasets = []
            for i, d in enumerate(self.trn_datasets):
                if self.proposal_used[i] == 'gaussian':
                    prev_datasets.append(d)
                    continue
                elif self.proposal_used[i] != 'prior' or not reuse_prior_samples:
                    continue
                # prior samples. the Gauss loss will reduce to the prior loss
                if isinstance(self.generator.prior, dd.Gaussian):
                    prop_m = self.generator.prior.mean
                    prop_P = self.generator.prior.P
                elif isinstance(self.generator.prior, dd.Uniform):
                    # model a uniform as an zero-precision Gaussian:
                    prop_m = np.zeros(self.generator.prior.ndim, dtype)
                    prop_P = np.zeros((self.generator.prior.ndim, self.generator.prior.ndim), dtype)
                else:  # can't reuse prior samples unless prior is uniform or Gaussian
                    continue
                prop_m = np.expand_dims(prop_m, 0).repeat(d[0].shape[0], axis=0)
                prop_P = np.expand_dims(prop_P, 0).repeat(d[0].shape[0], axis=0)
                prev_datasets.append((*d, prop_m, prop_P))

            trn_data = combine_trn_datasets(prev_datasets)
            n_train_round = trn_data[0].shape[0]

        # train network
        self.loss, trn_inputs = self.define_loss(n=n_train_round,
                                                 round_cl=round_cl,
                                                 proposal='gaussian')
        t = Trainer(self.network,
                    self.loss,
                    trn_data=trn_data, trn_inputs=trn_inputs,
                    seed=self.gen_newseed(),
                    monitor=self.monitor_dict_from_names(monitor),
                    **kwargs)

        log = t.train(epochs=self.epochs_round(epochs), minibatch=minibatch, verbose=verbose,
                      print_each_epoch=print_each_epoch, stop_on_nan=stop_on_nan)

        return log, trn_data

    def run_atomic(self, trn_data=None, n_train=100, epochs=100, minibatch=50, n_atoms=10, moo='resample', train_on_all=False,
                   reuse_prior_samples=True, combined_loss=False, round_cl=1, stop_on_nan=False, monitor=None,
                   verbose=False, print_each_epoch=False, **kwargs):

        # activetrainer doesn't de-norm params before evaluating the prior
        assert np.all(self.params_mean == 0.0) \
               and np.all(self.params_std == 1.0), "prior_norm+atomic not OK"

        assert minibatch > 1, "minimum minibatch size 2 for atomic proposals"
        if n_atoms is None:
            n_atoms = minibatch - 1 if theano.config.device.startswith('cuda') else np.minimum(minibatch - 1, 9)
        assert n_atoms < minibatch, "Minibatch too small for this many atoms"

        self.set_proposal()
        # if training data is provided, load data. Otherwise generate data.
        if trn_data is not None:
            trn_data, n_train_round = self.set_trn_data(trn_data, n_train)
        else:
            trn_data, n_train_round = self.gen(n_train)
        self.trn_datasets.append(trn_data)  # don't store prior_masks

        if train_on_all:
            if reuse_prior_samples:
                trn_data = combine_trn_datasets(self.trn_datasets, max_inputs=2)
            else:
                trn_data = combine_trn_datasets(
                    [td for td, pu in zip(self.trn_datasets, self.proposal_used) if pu != 'prior'])
            if combined_loss:
                prior_masks = \
                    [np.ones(td[0].shape[0], dtype) * (pu == 'prior')
                     for td, pu in zip(self.trn_datasets, self.proposal_used)]
                trn_data = (*trn_data, np.concatenate(prior_masks))
            n_train_round = trn_data[0].shape[0]

        # train network
        self.loss, trn_inputs = self.define_loss(n=n_train_round,
                                                 round_cl=round_cl,
                                                 proposal='atomic',
                                                 combined_loss=combined_loss and train_on_all)

        t = ActiveTrainer(self.network,
                          self.loss,
                          trn_data=trn_data, trn_inputs=trn_inputs,
                          seed=self.gen_newseed(),
                          monitor=self.monitor_dict_from_names(monitor),
                          generator=self.generator,
                          n_atoms=n_atoms,
                          moo=moo,
                          obs=(self.obs - self.stats_mean) / self.stats_std,
                          **kwargs)

        log = t.train(epochs=self.epochs_round(epochs), minibatch=minibatch, verbose=verbose,
                      print_each_epoch=print_each_epoch, strict_batch_size=True)

        return log, trn_data

    def run_MoG(self, trn_data=None, n_train=100, epochs=100, minibatch=50, n_atoms=None, moo=None, train_on_all=False, round_cl=1,
                stop_on_nan=False, monitor=None, verbose=False, print_each_epoch=False, reuse_prior_samples=True,
                **kwargs):
        assert not train_on_all, "train_on_all is not yet implemented for MoG "\
                "proposals"

        # simulate data
        self.set_proposal(project_to_gaussian=False)
        prop = self.generator.proposal
        assert isinstance(prop, dd.MoG)
        # if training data is provided, load data. Otherwise generate data.
        if trn_data is not None:
            trn_data, n_train_round = self.set_trn_data(trn_data, n_train)
        else:
            trn_data, n_train_round = self.gen(n_train)

        # here we're just repeating the same fixed proposal, though we
        # could also introduce some variety if we wanted.
        nc = prop.n_components
        prop_Pms = repnewax(np.stack([x.Pm for x in prop.xs], axis=0),
                            n_train_round)
        prop_Ps = repnewax(np.stack([x.P for x in prop.xs], axis=0),
                           n_train_round)
        prop_ldetPs = repnewax(np.stack([x.logdetP for x in prop.xs], axis=0),
                               n_train_round)
        prop_las = repnewax(np.log(prop.a), n_train_round)
        prop_QFs = \
            repnewax(np.stack([np.sum(x.Pm * x.m) for x in prop.xs], axis=0),
                     n_train_round)

        trn_data += (prop_Pms, prop_Ps, prop_ldetPs, prop_las,
                     prop_QFs)
        trn_data = tuple(trn_data)

        self.loss, trn_inputs = self.define_loss(n=n_train_round,
                                                 round_cl=round_cl,
                                                 proposal='mog')
        t = Trainer(self.network,
                    self.loss,
                    trn_data=trn_data, trn_inputs=trn_inputs,
                    seed=self.gen_newseed(),
                    monitor=self.monitor_dict_from_names(monitor),
                    **kwargs)

        log = t.train(epochs=self.epochs_round(epochs), minibatch=minibatch, verbose=verbose,
                      print_each_epoch=print_each_epoch, stop_on_nan=stop_on_nan)

        return log, trn_data

    def run_MLE(self, n_train=1000, epochs=100, minibatch=50, seed=None, n_atoms=None,
                moo=None, train_on_all=False, round_cl=1, stop_on_nan=False, monitor=None,
                verbose=False, print_each_epoch=False, **kwargs):

        n_train_round = self.n_train_round(n_train)
        xs = np.repeat(self.obs, n_train_round, axis=0) # list of repeated x_obs
        bounded_thetas = gen_bounded_theta(cmaf=self.network.cmaf, prior=self.generator.prior,
                                           x=self.obs, n_samples=n_train_round, rng=self.rng)
        trn_data = (np.asarray(bounded_thetas), np.asarray(xs))

        loss, trn_inputs = self.define_loss(n=n_train_round,
                                            round_cl=round_cl,
                                            proposal='prior')

        t = Trainer(network=self.network,
                    loss=loss,
                    trn_data=trn_data, trn_inputs = trn_inputs,
                    seed=self.gen_newseed(),
                    observe_leakage=True,
                    leakage_thr_lower=self.leakage_thr,
                    prior=self.generator.prior, obs=self.obs,
                    monitor=self.monitor_dict_from_names(monitor),
                    **kwargs)

        log = t.train(epochs=epochs, minibatch=minibatch, verbose=verbose,
                      print_each_epoch=print_each_epoch, stop_on_nan=stop_on_nan)

        return log, trn_data

    def set_proposal(self, project_to_gaussian=False):
        # posterior estimate becomes new proposal prior
        if self.round == 0:
            return None

        posterior = self.predict(self.obs)

        if project_to_gaussian:
            assert self.network.density == 'mog', "cannot project a MAF"
            posterior = posterior.project_to_gaussian()

        self.generator.proposal = posterior

    def gen(self, n_train, project_to_gaussian=False, **kwargs):
        """Generate from generator and z-transform

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        n_reps : int
            Number of repeats per parameter
        verbose : None or bool or str
            If None is passed, will default to self.verbose
        project_to_gaussian: bool
            Whether to always return Gaussian objects (instead of MoG)
        """
        if 'verbose' in kwargs.keys():
            verbose = kwargs['verbose']
        else:
            verbose = self.verbose
        verbose = '(round {}) '.format(self.round) if verbose else False
        n_train_round = self.n_train_round(n_train)

        trn_data = super().gen(n_train_round, verbose=verbose, **kwargs)
        n_train_round = trn_data[0].shape[0]  # may have decreased (rejection)

        return trn_data, n_train_round


    def n_train_round(self, n_train):
        # number of training examples for this round
        if type(n_train) == list:
            try:
                n_train_round = n_train[self.round-1]
            except:
                n_train_round = n_train[-1]
        else:
            n_train_round = n_train

        return n_train_round

    def epochs_round(self, epochs):
        # number of training examples for this round
        if type(epochs) == list:
            try:
                epochs_round = epochs[self.round-1]
            except:
                epochs_round = epochs[-1]
        else:
            epochs_round = epochs

        return epochs_round

    def set_trn_data(self, trn_data, n_train):
        n_train_round = self.n_train_round(n_train)
        n_samples = deepcopy(n_train_round)
        params = np.zeros((0, self.generator.prior.ndim), dtype=dtype)
        stats = np.zeros((0, self.generator.summary.n_summary), dtype=dtype)
        n_pilot = np.minimum(n_samples, len(self.unused_pilot_samples[0]))
        if n_pilot > 0 and self.generator.proposal is self.generator.prior:  # reuse pilot samples
            params = self.unused_pilot_samples[0][:n_pilot, :]
            stats = self.unused_pilot_samples[1][:n_pilot, :]
            self.unused_pilot_samples = \
                (self.unused_pilot_samples[0][n_pilot:, :],
                 self.unused_pilot_samples[1][n_pilot:, :])
            n_samples -= n_pilot
        if n_samples > 0:
            params_rem = trn_data[0]
            stats_rem = trn_data[1]
            params = np.concatenate((params, params_rem), axis=0)
            stats = np.concatenate((stats, stats_rem), axis=0)
            # z-transform params and stats
        params = (params - self.params_mean) / self.params_std
        stats = (stats - self.stats_mean) / self.stats_std
        trn_data = (params, stats)
        n_train_round = trn_data[0].shape[0]
        return trn_data, n_train_round