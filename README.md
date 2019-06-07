## Installation

`git clone https://github.com/Narsil/brownian.git`

## Using a new conda environment

`conda create --name brownian`
`source activate brownian`
`pip install torchvision scikit-learn matplotlib tqdm tb-nightly numpy`

## Train

`python train.py`


## Monitoring

`tensorboard --logdir=runs`
You can also check the output of the distribution of the output of the transformer
in the `figures` directory

The directory `checkpoints` will contain the models for future loading.






