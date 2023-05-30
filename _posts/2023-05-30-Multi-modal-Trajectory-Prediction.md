---
title: Multi-modal trajectory prediction
date: 2023-05-30
math:  true
mermaid: false
img_path: /assets/images/
categories: [autonomous_driving]
tags: [prediction]
---

# Background
## Concepts
**Agent**: An agent in trajectory prediction is a road user with self-cognition such as a pedestrian, a motorist or a cyclist.

**Trajectory**: A trajectory of an agent $i$ in trajectory prediction is defined as a sequence of 2D real-world or pixel coordinates: $\{X_i^T,Y_i^\tau\}$. where $X_i^T=\{X_i^t\mid t\in\[1,T\_{obs}\}$ is the observed trajectory with $T\_{obs}$ time steps, $Y_i^\tau=\{Y\_i^t\mid t\in\(T\_{obs}, T\_ {obs}+T\_{pred}\]\}$ is the ground truth of the future path with $T\_{pred}$ time steps. 

**Trajectory prediction**: The goal of trajectory prediction is to optimize a model $f_{TP}$ to predict $K$ future trajectories

$$ \hat{Y}_ {i,K}^{\tau}=\{\hat{Y}_ {i,k}^{\tau}\mid k=1,\dots,K\} $$

with observed information $X_i^T$ , $X_{[N]\backslash \{i\}}^T$ and $S$ , where $[N]\backslash\{i\}=\{j=1,\dots,N\mid j\neq i\}$ is the observed trajectories of other agents and $S$ is the static information such as map or LiDAR data. 
Thus the problem is aim to find a function such that

$$ \hat{Y}_ {i,K}=f_ {TP}(X_i^T,X_ {[N]\backslash \{i\}}^T, S) $$

if $K=1$, then the task is called *deterministic trajectory prediction* (DTP), otherwise it is called *multi-modal trajectory prediction* (MTP).

## Framework for DTP
The framework for DTP usually follows the sequence-to-sequence structure
- the past encoder extracts the spatial and temporal information from the observed information. 
- The decoder predicts the future path.

# Frameworks for MTP
An overview of the taxonomy of MTP frameworks
![MTP frameworks](MTP_frameworks.png)

| Category | Examples |
|--- | ---|
| Noise-based| GAN, cVAE, NF, DDPM|
| Anchor conditional| PEC, PTC | 
| Grid-based | Beam search, ST-MR, Gumbel sampling|
| Bivariate Gaussian | Stgcnn, MultiPath | 
| Others| Advanced discriminators, advance sampling tricks|


## Noise-based MTP framework
Noised-based methods completes MTP by adding random noise to DTP model.
The prediction is optimized by variety loss using the minimum reconstruction error:

$$ L\_{variety}\(\\hat{Y}\_{i,K}, Y\_i^{\tau}\)=\\min\_{k<K}L\_{rec}\(\\hat{Y}\_{i,K}, Y\_i^{\tau}\) $$

where $L_{rec}$ is the reconstruction error. 

- GAN. The discriminator can be used to discriminate between good or bad predictions. The problem is that GAN suffers from mode collapse.
- cVAE. By maximizing the evidence lower-bound of feature distribution, cVAE is an alternative way to encourage diverse predictions. Also the latent distribution of cVAE can be better controlled and enhanced. The drawback of cVAE is the likelihoods cannot be calculated directly.
- NF. NF can convert a complicated distribution into a tractable form via invertible transformations. However, NF models cannot handle discontinuous manifolds.
- DDPM. Diffusion model can also be used for trajectory prediction, however DDPM suffers from the computation consumption problems.


## Anchor Conditioned MTP
Each prediction should be conditioned on a prior, also named *anchors*, explicit to each modality.
Some anchors:
- endpoints, the final locations the agent may arrive at.
- prototype trajectories, the basic motions the agent may follow.

Anchors can effectively alleviates the mode collapse problem and encourage more robust and explainable predictions.

Anchor Conditioned MTP contains two sub-tasks:
1. *anchor selection*, selects K plausible anchors from an anchor set;
2. *waypoint decoding*, predicts *waypoints*, the final prediction of future trajectory, based on the given anchor.

- PEC. Simple and effective. But avoiding the unreachable endpoints and leveraging the multi-modality are problems.
- PTC. PTC can simplify the training and achieve diversity, but the current prototype trajectories are usually too simple to handle complex scenarios.

## Grid-based MTP framework
Grid-based methods employs occupancy grid maps to indicate which location the agent will go to in the next time step.
Grid-based methods can be highly compliant with scenes with advanced training strategies such as RL or occupancy losses and suitable for long-term prediction.
The drawback of grid-based methods is that significant computation and sensitive to the resolution of the maps.

## Bivariate Gaussian for output representation
the output of TP can be represented as a bivariate Gaussian distribution.
The problem of bivariate Gaussian is that the output positions are sampled individually and may not be temporally correlated, causing unrealistic predictions.

## Some other techniques
- Advanced discriminators can be used to improve the quality of generated trajectories.
- Advanced sampling can be used to improve the efficiency and coverage.

# Evaluation Metrics
Averaged displacement error (ADE) and Final displacement error (FDE) can be used to measure the quality of trajectories.

## Lower-bound-based MTP Metrics
Given K predicted trajectories, each prediction is compared with the ground truth and the best score is recorded without considering the exact confidence.
- Minimum-of-N(MoN). It calculates the minimum error among all predictions:

	$$ MoN=\mathbb{E}_ {i,t\in\tau}\min_ {k<K}DE(\hat{Y}_ {i,k}^t, Y_ i^t) $$
  
	where $DE$ can be any distance metrics.

- Miss Rate (MR). A prediction misses the ground truth if it is more than $d$ meters from the ground truth according to their displacement error and hits otherwise.

   $$ MR=\mathbb{E}_ {i,t\in\tau}\mathrm{sign}(\min_ {k<K}DE(\hat{Y}_ {i,k}^t, Y_ i^t)-d) $$
   
Cons:
- sensitive to randomization.
- Information leak since only the best prediction is used for evaluation based on the distances to the ground truth.

## Probability-aware metrics
This kind metrics measure how likely the ground truth can be sampled from the predicted distribution.
- Most-likely (ML) based metrics. Select the prediction with the highest probability to perform the DTP evaluation.
- topK based metrics. Select candidates with a probability larger than a threshold $\gamma$ among $M>>K$ predictions for MoN evaluation, known as probability cumulative minimum distance (PCMD):

	$$ PCMD = MoN(\hat{Y}_ {i,k}\mid  P(\hat{Y}_ {i,k'}\mid X_ i^T, k'<M)\geq\gamma) $$ 
  
- Gaussian-based metrics. First estimate a Gaussian distribution given $K$ discrete predictions using a method such as kernel density estimation (KDE).

	$$ KDE-NLL = -\mathbb{E}_ {i,t\in\tau}\log P(Y_ i^t\mid KDE(\hat{Y}_ {i,K}^t)) $$

cons:
- Due to noise, ground truth may not be most likely.

## Distribution-aware Metric
The main barrier is that only one ground truth is provided and its distribution cannot be estimated.

- Earth-moving distance can be used to calculate the ADE results with linear sum assignment between predicted and ground truth samples.
- Recall can be used to measure the coverage.

   $$ Recall = \mathbb{E}_ {k<K_G}(\min_ {k'<K_R}\|\hat{Y}_ {i,k}^t-Y_ {i,k'}^t\|_2)<d $$
   
   where $K_G$ is the number of predictions and $K_R$ is the number of annotated ground truths for agent $i$.
- Precision, calculates the ratio of generated samples in the support of the ground truth distribution and penalize out-of-distribution predictions:

	$$ Precision=\mathbb{E}_ {k<K_ R}(\min_{k'<K_ G}\|\hat{Y}_ {i,k}^t-Y_ {i,k'}^t\|_ 2) < d $$

Cons: 
- Require extra annotations and corrections by human experts on real-world datasets.
- Human annotations still cannot guarantee the coverage of all modalities.

# Datasets
There are some datasets can be used for DTP and MTP.

| Datasets| Metrics |Type| Notes|
| ---|---|---|---|
|ETH&UCY | - | Pedestrian |---|
|Drone | - | Pedestrian |---|
|Nuscene | ADE | vehicle | ---|
| Argoverse | FDE | vehicle | ---|
| Waymo | FDE,mAP | vehicle | ---|
| ForkingPath | | pedestrian | multiple ground path|


# Future directions
- **Better evaluation metrics**. Metrics that dataset-unrelated and neglect unacceptable predictions.
- **Motion planning using multi-modal predictions**. Make trajectories generated by motion planning more robust.
- **Language-guided explainable MTP**. Provide human-understandable decision-making for multi-modal predictions.
- **Lightweight MTP Frameworks**. Reduce time and memory consumption of MTP frameworks.
- **MTP with out-of-distribution (OOD) Modalities.** Make the model more robust to unseen environment (OOD).
- **Urban-wide MTP**. Extend to an urban-wide location prediction?


# Reference
1. [Multimodal Trajectory Prediction: A Survey](https://arxiv.org/abs/2302.10463)
