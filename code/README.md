We provide PyTorch code for our paper "Unlabeled Imperfect Demonstrations in Adversarial Imitation Learning". 

## Requirements
Experiments were run with Python 3.6 and these packages:
* pytorch == 1.1.0
* gym == 0.15.7
* mujoco-py == 2.0.2.9

## Train UID

 * UID
 ```
  python uid_main.py --env_id 5 --il_method uid --fix_length --seed 0
 ```
 * UID-WAIL
 ```
  python uid_main.py --env_id 5 --il_method uidwail --fix_length --seed 0
 ```
 * GAIL
 ```
  python uid_main.py --env_id 5 --il_method gail --fix_length --seed 0
 ```

For other compared methods, the re-implementation of [2IWIL/IC-GAIL](https://github.com/kristery/Imitation-Learning-from-Imperfect-Demonstration) and [D-REX](https://github.com/dsbrown1331/CoRL2019-DREX) can be found in core/irl.py and trex_main.py. 
