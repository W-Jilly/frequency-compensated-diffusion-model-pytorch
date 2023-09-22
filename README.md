
<h1 align="left">Frequency Compensated Diffusion Model for Real-scene Dehazing<a href="https://arxiv.org/abs/2308.10510"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a> </h1> 


This is an official implementation of **Frequency Compensated Diffusion Model for Real-scene Dehazing** by **Pytorch**.





<img src="misc/framework-v3.jpg" alt="show" style="zoom:90%;" />
<!-- (a) The training process of the proposed dehazing diffusion model. At step $t$, the network takes an augmented hazy image $I_{aug}$ and a noisy image $J_t$ as inputs. The network architecture adopts special skip connections, i.e., the Frequency Compensation Block (FCB), for better $\epsilon$-prediction. (b) The detailed block design of FCB. The input signals of FCB are enhanced at the mid-to-high frequency band so that the output spectrum has abundant higher frequency modes. (c) The sampling process of the proposed dehazing diffusion model. -->

<!-- -  <img src="./misc/train_prove_v3.jpg" alt="show" style="zoom:90%;" /> 
Power spectrum analysis on $\epsilon$-prediction results of DDPMs at varying $t$.
(a) The power spectra of DDPM and DDPM+FCB.
(b) The PSD analysis of DDPM and DDPM+FCB.
(c) The KL distance between the spectrum of the predicted $\epsilon$ in (b) and that of the groundtruth. The smaller distance, the closer to groundtruth.
 -->
 
## Getting started
### Installation
* This repo is a modification on the [**SR3 Repo**](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement ).

* Install third-party libraries.

```python
pip install -r requirement.txt 
```

### Data Prepare

Download train/eval data from the following links:

Training: [*RESIDE*](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)

Testing:
[*I-Haze*](https://data.vision.ee.ethz.ch/cvl/ntire18//i-haze/#:~:text=To%20overcome%20this%20issue%20we%20introduce%20I-HAZE%2C%20a,real%20haze%20produced%20by%20a%20professional%20haze%20machine.) / 
[*O-Haze*](https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/) /
[*Dense-Haze*](https://arxiv.org/abs/1904.02904#:~:text=To%20address%20this%20limitation%2C%20we%20introduce%20Dense-Haze%20-,introducing%20real%20haze%2C%20generated%20by%20professional%20haze%20machines.) /
[*Nh-Haze*](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/) /
[*RTTS*](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=0) 

<!-- 
### Already
```Already
# username pwd
intern@43.82.40.49  intern123

# start docker
docker exec -it yzq_cas bash

# code
cd /workspace/dehazing_yzq/

# pull from gitlab
git pull origin master

# vim config
vim config/framework_da.json

# train
python train.py 

# infer 
python infer.py
``` -->

## Pretrained Model

We prepared the pretrained model at:

| Type                                                        | Platform                                        |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| Generator                                                 | [GoogleDriver](https://drive.google.com/file/d/1KYQNJFjvrfAhTLi_V1Z5NZWcyK4CuLSM/view?usp=share_link) |

<!-- ```python
# Download the pretrain model and edit ./config/framework_da.json about "resume_state":
"resume_state": '/data/diffussion/tmp/framework_da_230221_121802'
``` -->


<!-- 
### About run 

```python
# train
python train.py -c [config file] 

# Parameter integration
python avg_params.py

# infer
python infer.py -c [config file] -i 

# eval
bash indicator/eval.sh
```

### Param config
```python
# HazeAug config: 
vim  config/framework_da.json
change  "HazeAug": true/false,

# FCB config:
vim  config/framework_da.json
change  "FCB": true/false,
```

### Storage Path

You can change the weight file and output image storage location under the file core/logger.py

```python
    if args.infer:
        experiments_root = os.path.join(
            '/data/diffusion_data/infer', '{}_{}'.format(opt['name'], get_timestamp()))
    else:
        experiments_root = os.path.join(
            '/data/diffusion_data/experiments', '{}_{}'.format(opt['name'], get_timestamp()))
    opt['path']['experiments_root'] = experiments_root
``` -->

## Results
Quantitative comparison on real-world hazy data (RTTS). Bold and underline indicate the best and the second-best, respectively.
<p align="center">
  <img src="misc/RTTS.jpg" width="600">
</p>

## Todo


- [x] Upload configs and pretrained models

- [x] Upload evaluation scripts

- [x] Upload train scripts
