TensorFlow GPU support is needed to run HTNet code. Install instructions can be found in the [tensorflow website](https://www.tensorflow.org/install/gpu) or in [this blog](https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1).

Tested build configurations:

Linux

| NVIDIA GPU Driver | CUDA toolkit | cuDNN SDK | Python | TensorFlow |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 460.32.03 | 11.0 | 8.1.0 | 3.8.8 | 2.4.0 |
| 460.32.03 | 11.0 | ? | 3.7.10 | 2.4.1 |

Use the above configurations to avoid library conflicts.

Always check compatibility:

- [CUDA toolkit](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)
- [TF](https://www.tensorflow.org/install/source#gpu)
