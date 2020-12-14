# Modeling

## Deep Learning U-Net Model Reference
[DocProj Project Page](https://xiaoyu258.github.io/projects/docproj/) 

This work was heavily influenced by the DocProj project. Please checkout their code at the link above.

## Dependencies

* Python3
* Windows (and Linux for Dataset Generation)
* CUDA and CuDNN


## Generate Your Own Dataset
Generating your own dataset is a great way to train an existing model. Please follow the instructiosn on the main page for dataset generation.

## Generate Data-Set Patches 
Please run the following pre-processing commands to create local and global patches of your data-set. These patches are necessary inputs for model training. Please change the arguments as necessary for your implementation.

`python local_patch.py`  

`python global_patch.py `

## Training
Please run the follwong command for training, and change the arguments as necessary for your implementation.

`python train.py`

## Evaluate an Image with a Model
Please run the following four command to evaluate the model. An example is given in training.sh. Additionally, the Graphcut.exe application can be found [here](https://drive.google.com/open?id=1QI2v1oWgha0jdcVuj7mzOXpgjBULZ7Mg).

`python eval.py --imgPath [input_image.png] --modelPath [model_to_save.pkl] --saveImgPath [new_resized_image.png] --saveFlowPath [myflow.npy]`

`Graphcut.exe [myflow.npy] [my_new_flow.npy]`

`python resampling.py --img_path [new_resized_image.png] --flow_path [my_new_flow.npy]`

`python.exe eval_illumination.py --imgPath [resamplling_result.png] --savPath [output.png] --modelPath [model_illNet.pkl]`



## Contact
Arnav Sharma - arnavsha@umich.edu

Austin Jeffries - ajeffr@umich.edu

Ali Badreddine - abadredd@umich.edu
