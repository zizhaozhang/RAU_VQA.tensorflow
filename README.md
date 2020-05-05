# RAU_VQA.tensorflow
A tensorflow version of RAU_VQA (https://github.com/HyeonwooNoh/RAU_VQA)


This tensorflow code implements the paper (https://arxiv.org/pdf/1606.03647.pdf). There are some details that have not been added. I will complete them later on. But the current version is able to train.

### How to train
- Follow https://github.com/HyeonwooNoh/RAU_VQA to download the data in the ./data folder

- Deploy the training 

    ```
    python train.py
    ``` 

- Open tensorboard to monitor the training and the loss of multiple hops

    ```
    tensorboard --logdir=./checkpoint/
    ```
    

