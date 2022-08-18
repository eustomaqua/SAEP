# SAEP


This repository is used to release the code for our paper: ***Sub-Architecture Ensemble Pruning in Neural Architecture Search (SAEP)*** [[arxiv]](https://arxiv.org/abs/1910.00370v2) [[paper]](https://ieeexplore.ieee.org/document/9460115) 

The code is conducted based on [AdaNet](https://github.com/tensorflow/adanet), with critical modifications in the searching process: We intend to prune the ensemble architecture on-the-fly based on various criteria and keep more valuable subarchitectures in the end; The criteria used to decide which subarchitectures will be pruned have three proposed solutions here in our SAEP, that is,

- *Pruning by Random Selection (PRS)*
- *Pruning by Accuracy Performance (PAP)*
- *Pruning by Information Entropy (PIE)*


## Getting Started

To get you started, you may refer to:

**Document**


**Example**

```shell
$ pytest saep
$ pytest . --ignore=saep
```


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

