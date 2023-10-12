# makemore

Reimplementation of Andrej Karpathy [makemore] (https://github.com/karpathy/makemore) library for educatinal purpose.

I've just simple restructured the code to allow for an easier training and inference (and simply testing new model architectures).

It contains different autoregressive models that, fed with a database of names, it can be trained to generate new names (human-like).

The implemented models are:
* Bigram
* MLP
* RNN
* GRU
* Transformer
The architectures can be found in the 'nets/' folder.


## Usage

The dataset is included in the 'data/' folder. As for the original reporsitory, it is a 'names.txt' file containing a databes of 32K names.

To train a new model just simply run the 'train.py' script. The training variables and hyperparameters can be set in the 'config/training.json', where among the other things you can decide the model **architecture** to be trained.

Once a model has been trained, it can be tested to generate random new names using the 'generate.py' script. As for the training, in the 'config/inference.json' you can set the inference parameters (most importantly you should set the 'model_path' to point to the directory where a .pt model is saved).

