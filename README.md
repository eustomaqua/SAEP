# SAEP

[![Build Status](https://app.travis-ci.com/eustomaqua/SAEP.svg?branch=master)](https://app.travis-ci.com/eustomaqua/SAEP) 
[![Coverage Status](https://coveralls.io/repos/github/eustomaqua/SAEP/badge.svg?branch=master)](https://coveralls.io/github/eustomaqua/SAEP?branch=master) 
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/a758b0c84d3d45cb8f1fa414abd64c09)](https://www.codacy.com/gh/eustomaqua/SAEP/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=eustomaqua/SAEP&amp;utm_campaign=Badge_Grade) 
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/a758b0c84d3d45cb8f1fa414abd64c09)](https://www.codacy.com/gh/eustomaqua/SAEP/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=eustomaqua/SAEP&amp;utm_campaign=Badge_Coverage) 
![](https:\//img.shields.io\/badge\/baseline-adanet-brightgreen) 


This repository is used to release the code for our paper: ***Sub-Architecture Ensemble Pruning in Neural Architecture Search (SAEP)*** [[arxiv]](https://arxiv.org/abs/1910.00370v2) [[paper]](https://ieeexplore.ieee.org/document/9460115) 

The code is conducted based on [AdaNet](https://github.com/tensorflow/adanet), with critical modifications in the searching process: We intend to prune the ensemble architecture on-the-fly based on various criteria and keep more valuable subarchitectures in the end; The criteria used to decide which subarchitectures will be pruned have three proposed solutions here in our SAEP, that is,

- *Pruning by Random Selection (PRS)*,
- *Pruning by Accuracy Performance (PAP)*, and
- *Pruning by Information Entropy (PIE)*.

## Getting Started

You may start with the example below

```shell
$ python main.py -data mnist -model dnn
$ python main.py -data fashion_mnist -model cnn  # or fmnist
$ python main.py -data cifar10 -bi -c0 4 -c1 7
```

To use SAEP, you need to adjust a few details

```shell
$ # baseline
$ .. -type AdaNet.O
$ .. -type AdaNet.W -mix

$ # SAEP
$ .. -type PRS.O
$ .. -type PAP.O
$ .. -type PIE.O -alpha 0.5

$ # Corresponding variants
$ .. -type <SAEP-name>.W -mix
```

You are also free to adjust other training parameters for potential better results. 

PS. Mistakes may exist in the current version. Please do not hesitate to contact us if you find any. You are also more than welcome to send us pull requests.

## Requirements

**Environment**

|         | CUDA | cuDNN |
|:-------:|:----:|:-----:|
| Ubuntu  | 10.0 | 7.6.4 |
| Windows | 10.1 | 7.6.5 |

**Dependency**

Create a virtual environment
```shell
$ conda create -n test python=3.6
$ source activate test
```

Install packages
```shell
$ pip install numpy==1.19.5
$ pip install tensorflow-gpu==2.1.0

$ # pip install -r requirements.txt
```

Remove a virtual environment
```shell
$ source deactivate
$ conda remove -n test --all
```

## Citing this Work

Please cite our paper if you find this repository helpful

```tex
@Article{9460115,
  title     = {Subarchitecture Ensemble Pruning in Neural Architecture Search},
  author    = {Bian, Yijun and Song, Qingquan and Du, Mengnan and Yao, Jun and Chen, Huanhuan and Hu, Xia},
  journal   = {IEEE Transactions on Neural Networks and Learning Systems},
  year      = {2021},
  volume    = {},
  number    = {},
  pages     = {1-9},
  doi       = {10.1109/TNNLS.2021.3085299},
  publisher = {IEEE},
  url       = {https://ieeexplore.ieee.org/document/9460115},
}
```

## License

SAEP is released under the [MIT License](./LICENSE).
