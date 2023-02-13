## Unlabeled Imperfect Demonstrations in Adversarial Imitation Learning

We provide PyTorch code for our paper "Unlabeled Imperfect Demonstrations in Adversarial Imitation Learning".

We will make UID's source code and training data in all environments available to the public after the paper acceptance.

## Requirements
Experiments were run with Python 3.6 and these packages:
* pytorch == 1.1.0
* gym == 0.15.7
* mujoco-py == 2.0.2.9

## Train UID

 * UID
 ```
  python uid_main.py --env_id 1 --il_method uid --c_data 1/2 --seed 0/1/2/3/4
 ```

 * GAIL
 ```
  python uid_main.py --env_id 1 --il_method gail --c_data 1/2 --seed 0/1/2/3/4
  
 ```
 * 2IWIL / IC-GAIL
 ```
  python uid_main.py --env_id 1 --il_method iwil/icgail --c_data 1/2 --seed 0/1/2/3/4
```
 

For other compared methods, the re-implementation of [T-REX/D-REX](https://dsbrown1331.github.io/CoRL2019-DREX/) can be found in trex_main.py. 
