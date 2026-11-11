# Methods and interpretation

## Exact population prediction

A context \(c\) contains three four-valued observation tokens. Its exact
probability is \(p(c)\), and its teacher next-token distribution is
\(q(v\mid c)\). The common objective in both arms is

$$
K(w)=\sum_c p(c)\sum_v q(v\mid c)
\log\frac{q(v\mid c)}{p_w(v\mid c)}.
$$

The HMM starts in its uniform stationary distribution. The first observation
updates that prior; each subsequent observation follows a latent transition.
The teacher predicts after one further latent transition and observation.
`source_table` returns all contexts, their weights, next-token targets, and
posterior beliefs about the latent state at the final observed time. Token
labels map to Cartesian latent/observation states in the order `[2, 0, 3, 1]`.
The model receives token IDs, without factor labels.

The default architecture has 32 token-embedding, 24 position-embedding,
192 query/key/value, 64 attention-output, 36 readout, and 280 MLP parameters,
for a total of 628. Initialization randomizes 600 coordinates; the 28 bias
coordinates start at zero. All coordinates subsequently vary. Only the final
attention query is evaluated because the other queries cannot affect its
output in a single attention layer. A test compares this computation with
PyTorch multihead attention.

At nearly exact fits, subtracting teacher entropy from cross entropy loses
precision. `population_kl` instead centers the logit error under the teacher
and evaluates its exponential remainder, using a Taylor polynomial near zero.
Tests compare its gradients with cross entropy and resolve errors with KL of
order \(10^{-18}\).

## Intervention and optimization

Within each seed, the two arms share initialization, data, architecture,
optimizer settings, and an orthogonal rotation of their auxiliary targets.
The target features are centered and standardized under \(p(c)\); their total
centered variance is one in either arm. The joint code adds the posterior
product moment to the four factored features. At independence, that moment
is the product of the marginal moments. Encoding it linearly is redundant
for prediction but changes the prepared feature geometry.

The saved protocol uses 3,000 Adam steps at learning rate 0.01 and up to
1,000 L-BFGS iterations with auxiliary coefficient 0.3. Another 1,000 Adam
steps anneal that coefficient quadratically to zero, at learning rate 0.001.
A fresh optimizer then takes 1,000 Adam steps at 0.001, followed by up to
2,000 L-BFGS iterations on prediction KL alone. A prediction-only affine
readout solve and damped Gauss–Newton refinement are followed by up to
300 final L-BFGS iterations. L-BFGS closure evaluations can exceed its
iteration limit because of line searches.

All three output logit contrasts and all weights remain free during this
refinement. Its teacher-logit equations have the same zero set as prediction
KL. This preparation procedure is distinct from the later plain GD
persistence test. It does not estimate the frequency of joint solutions
under an ordinary random-initialization training protocol.

A saved fit passes numerical convergence checks when KL is at most
\(10^{-10}\), gradient norm is at most \(10^{-7}\), and its smallest numerical
Hessian eigenvalue is at least \(-10^{-7}\). These checks do not prove an
exact local minimum. The `geometry.py` diagnostic also reports a coefficient
conditional on full-rank, exactly realizing product-source assumptions;
that field is not an established LLC of the saved approximate checkpoint.

## Probes and coordinate dependence

The additive regression fits a constant and arbitrary functions of each
factor's eight possible observed histories. Its weighted residual squared
norm divided by total centered activation variance is the raw nonadditivity
score. The operational screen requires less than 1% in the factored arm,
more than 8% in the joint arm, and joint product-moment probe \(R^2>0.99\).
The same fixed raw thresholds define the classes in the GD persistence test.

The belief probes fit an affine function of the eight final hidden
coordinates to the two posterior spin moments and their product moment.
All fits and residuals use exact context probabilities. Probe accuracy on
this complete table measures linear accessibility on the population task.
It is not an estimate from a train/test split of contexts.

Raw variance scores can change under an invertible hidden-basis change.
`residual_basis_transform` transforms the adjacent weights to preserve all
predictions while changing hidden activations from \(h\) to \(hA\).
`activation_controls` therefore also compares weighted centered activation
column spaces, using relative singular-value cutoff \(10^{-12}\). Whitened
nonadditivity is invariant to invertible affine coordinate changes at this
numerical rank threshold. Weak joint features in a factored checkpoint
prevent interpreting raw nonadditivity as the presence or absence of joint
information. A nonorthogonal parameter transformation also changes the
Euclidean localization measure used for free energy.

## Annealed SMC

Displacements are expressed in a fixed orthonormal Fisher basis. At inverse
temperature \(\tau\), the HMC coordinate scaling is
\((\gamma+\tau\lambda_i)^{-1/2}\), including every zero or weak Fisher mode.
The energy and localization are always evaluated for the original weights.
The Fisher matrix is recomputed from the checkpoint, rather than substituting
its saved Hessian for the proposal preconditioner.

The initial particles are independent exact samples from
\(\mathcal N(w_*,I/\gamma)\). At each successive temperature, the estimated
normalizing constant is multiplied by the mean incremental weight
\(\exp[-(\tau_{k+1}-\tau_k)(K(w)-K(w_*))]\). Systematic resampling is followed
by randomized-length HMC trajectories with a Metropolis acceptance test.

A discarded pilot adapts temperatures to 85% incremental ESS and adapts HMC
step sizes toward acceptance 0.85. Production freezes its schedule and step
sizes and uses different seeds. The primary transformer configuration has
128 particles, eight HMC moves per stage, and trajectory lengths drawn from
8–24 below \(10^6\) and 32–96 at and above \(10^6\). The other two
configurations use 128 or 256 particles with lengths 8–24 throughout.
Each configuration uses four populations per checkpoint.

The independent estimates of \(Z\) are averaged before taking the negative
logarithm. Standard errors use between-population variation, with covariance
across temperatures retained in slope errors. Endpoint energies pool
unnormalized moments and divide by pooled evidence. Resampled particles are
not counted as independent error-bar samples. Evidence concentration in a
single population is reported separately; nominal standard errors cannot
exclude contributions missed by all populations.

The primary observables are finite localized free energies. Gaussian
localization has full support, and the result depends on its scale and weight
coordinates. Neither differing finite free energies nor differing finite-window
slopes proves different pointwise asymptotic LLCs. The highest temperature
remains sampling-sensitive, and no slope plateau is observed.

## Persistence

`persistence.py` applies the deterministic update

$$
w_{t+1}=w_t-0.2\nabla K(w_t)
$$

to the prepared checkpoints, without momentum, auxiliary supervision,
preconditioning, or Newton refinement. Optional perturbations are independent
Gaussian coordinates added once at initialization. Recorded probes check
finite-time class retention; they do not prove asymptotic attraction or
stability under persistent noise. Checkpoint-centered perturbation return
fractions are conditional local measurements, rather than masses under the
original network initializer.
