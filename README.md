# makemore

Reimplementation of Andrej Karpathy [makemore](https://github.com/karpathy/makemore) repository for educational purpose.

I've just restructured the code to allow for an easier training and inference (and simply testing new model architectures).
It contains different autoregressive models that, fed with a database of names, can be trained to generate new human-like names.

The implemented models are:
* Bigram
* MLP
* RNN
* GRU
* Transformer
  
The architectures can be found in the `nets/` folder.


## Usage

The dataset is included in the `data/` folder. As for the original reporsitory, it is a `names.txt` file containing a databes of 32K names.

To train a new model just simply run the `train.py` script. The training variables and hyperparameters can be set in the `config/training.json`, where among the other things you can decide the model *architecture* to be trained.

Here we can see the test loss trend of the different available model architectures: it shows how we can improve the performance going from the Bigram to the Transformer.

<p align="center">
<img width="808" alt="makemore_test_loss" src="https://github.com/LucaMancaITA/makemore/assets/79864480/cf95da07-40b9-4aad-b6f9-723e2c391887">
</p>


Once a model has been trained, it can be tested to generate random new names using the `generate.py` script. As for the training, in the `config/inference.json` you can set the inference parameters (most importantly you should set the *model_path* to point to the directory where a .pt model is saved).

