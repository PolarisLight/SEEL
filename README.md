# SEEL:Supervised Evolutionary Exploration Learning

![Exploratory_](figure\Exploratory_.png)

This repository is the PyTorch implementation of *"Supervised Evolutionary Exploration Learning for Classification
Tasks"*

## Setup

### Environment

Our experiment are conduct on:

- OS: Ubuntu 22.04.3
- GPU: NVIDIA GeForce RTX 3090

### Python environment

Please refer to [requirements.txt](requirements.txt) for the python environment.

### Dataset

We use the following datasets in our experiments:

Sentiment Identification Tasks:

- Laptops
- Restaurant
- Twitter
  which can be downloaded from [here](https://github.com/songyouwei/ABSA-PyTorch/tree/master/datasets)

Image Classification Tasks:

- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
  which we use the implementation from torchvision.datasets.CIFAR100

## Usage

### Sentiment Identification Tasks

Run "run_alsc_mem_pre.py" or "run_alsc_plm_pre.py" for pre-training the model.

Run "run_alsc_mem.py" or "run_alsc_plm.py" for post-training the model with CE, SCL and SEEL strategies.

The hyper-parameters are set in the code and "config.py".

### Image Classification Tasks

Run "run_img_pre.py" for pre-training the model.

Run "run_img.py" for post-training the model with CE, SCL and SEEL strategies.

The hyper-parameters are set in the code and "config.py".

## Results

### Sentiment Identification Tasks:

![ABSA result](figure/absa.png)

### Image Classification Tasks:

Accuracy of the models on the test set of CIFAR-100, Subset.b and Subset.imb. The best results are in bold.

| Strategy    | CIFAR100 | Subset.b | Subset.imb |
|-------------|----------|----------|----------|
| Base        | 76.97    | 70.53    | 61.94    |
| CE          | 77.50    | 71.03    | 61.94    |
| CE_SCL      | 77.51    | 71.05    | 61.94    |
| CE_SCL_SEEL | **77.75** | **71.22** | **63.49** |

### Visualization

t-SNE visualization of the learned representations of the test set in the Restaurant dataset. SEEL can learn more
discriminative representations than CE and SCL.
![Visualization](figure/ABSA_visualization.png)



