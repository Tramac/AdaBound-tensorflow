# AdaBound-tensorflow/keras
A tensorflow and keras implementation of [AdaBound](https://openreview.net/forum?id=Bkg3g2R9FX) optimizer from 2019 paper by Luo. The official and original PyTorch code can be found [here](https://github.com/Luolc/AdaBound).

## Usage
First add the adabound-tensorflow/keras to your project.<br>
- Keras
```python
from adabound_keras.adabound import AdaBound
optimizer = AdaBound(lr=0.001, final_lr=0.1)
```
- TensorFlow
```python
from adabound_tensorflow.adabound import AdaboundOptimizer
optimizer = AdaboundOptimizer(learning_rate=0.001, final_lr=0.1)
```
## TODO
 - Still to come:
 * [x] Keras version
 * [x] TensorFlow version
 * [ ] Add test demo
 
 ## References
 - Luo, et al. [Adaptive Gradient Methods with Dynamic Bound of Learning Rate](https://openreview.net/forum?id=Bkg3g2R9FX). In Proc. of ICLR 2019.
 - [Original Implementation (PyTorch)](https://github.com/Luolc/AdaBound)
 - [keras.optimizers](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizers.py)
 - [tensorflow.adam](https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/training/adam.py)
