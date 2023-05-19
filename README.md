# Unlabeled Imperfect Demonstrations in Adversarial Imitation Learning

This repository contains the PyTorch code for the paper "Unlabeled Imperfect Demonstrations in Adversarial Imitation Learning" in AAAI 2023. [[Paper]](https://arxiv.org/pdf/2302.06271.pdf)[[Appendix]](https://github.com/yunke-wang/yunke-wang.github.io/blob/main/docs/Appendix_For_UID.pdf)

## Requirements
Experiments were run with Python 3.6 and these packages:
* pytorch == 1.1.0
* gym == 0.15.7
* mujoco-py == 2.0.2.9

## Data Collection
We provide two different kinds of imperfect demonstrations data (i.e., __D1__ and __D2__) to evaluate the performance of UID. 
We firstly train an optimal policy $\pi_o$ by TRPO and $\pi_o$ is used to sample optimal demonstrations $D_o$. 
To collect imperfect demonstrations, 3 non-optimal demonstrators $\pi_n$ are used. 
$\pi_n$ in __D1__ is obtained by saving 3 checkpoints with increasing quality during the RL training. 
In __D2__, we add different Gaussian noise $\xi$ to the action distribution $a^\ast$ of $\pi_o$ to form non-optimal policy $\pi_n$. 
The action of $\pi_n$ is modeled as $a\sim\mathcal{N}(a^\ast, \xi^2)$ and we choose $\xi=[0.25, 0.4, 0.6]$ in these 3 non-optimal policies (i.e., $\pi_{n_3}$, $\pi_{n_2}$ and $\pi_{n_1}$). 

The quality of each demonstrator is provided in the appendix.

## Train UID

 * UID-GAIL / UID-WAIL
 ```
  python uid_main.py --env_id 1/2/3 --il_method uid/uidwail --c_data 1/2 --seed 0/1/2/3/4
 ```
 * GAIL / WAIL / VAIL
 ```
  python uid_main.py --env_id 1/2/3 --il_method gail/irl/vail --c_data 1/2 --seed 0/1/2/3/4
 ```
 * 2IWIL / IC-GAIL
 ```
  python uid_main.py --env_id 1/2/3 --il_method iwil/icgail --c_data 1/2 --seed 0/1/2/3/4
```

For other compared methods, the re-implementation of [T-REX/D-REX](https://dsbrown1331.github.io/CoRL2019-DREX/) can be found in trex_main.py. 

## Contact
For any questions, please feel free to contact me. (Email: yunke.wang@whu.edu.cn)

## Citation
```
@article{wang2023unlabeled,
  title={Unlabeled Imperfect Demonstrations in Adversarial Imitation Learning},
  author={Wang, Yunke and Du, Bo and Xu, Chang},
  journal={arXiv preprint arXiv:2302.06271},
  year={2023}
}
```

## Reference
[1] Generative adversarial imitation learning. NeurIPS 2016.

[2] Learning robust rewards with adversarial inverse reinforcement learning. ICLR 2018.

[3] Variational discriminator bottleneck: Improving imitation learning, inverse rl, and gans by constraining information flow. ICLR 2017.

[4] InfoGAIL: Interpretable Imitation Learning from Visual Demonstrations. NeurIPS 2017

[5] Imitation learning from imperfect demonstration. ICML 2019.

[6] Extrapolating beyond suboptimal demonstrations via inversere inforcement learning from observations. ICML 2019.

[7] Better-than-demonstrator imitation learning via automatically-ranked demonstrations. CoRL 2020.

[8] Variational Imitation Learning with Diverse-quality Demonstrations. ICML 2020.

[[9]](https://github.com/yunke-wang/WGAIL) Learning to Weight Imperfect Demonstrations. ICML 2021

[[10]](https://github.com/yunke-wang/SAIL) Robust Adversarial Imitation Learning via Adaptively-Selected Demonstrations. IJCAI 2021.
