# Paper Flattening
EECS 504 Fall 2020 at University of Michigan Final Project Assignment regarding paper flattening.

[Paper](https://github.com/arnavsharma/the-flatteners/blob/main/Paper/EECS_504_Project_Paper%20-%20The%20Flatteners.pdf) | [Presentation](https://github.com/arnavsharma/the-flatteners/blob/main/Paper/EECS%20504%20Presentation.pdf) | [Web-App Demo](https://shrouded-temple-50673.herokuapp.com/) | [Code](https://github.com/arnavsharma/the-flatteners)

## Dependencies

* Python3
* Windows (and Linux for Dataset Generation)
* CUDA and CuDNN

## Set-up
`pip install -r requirements.txt`

## Demo
To run the demo on your location machine instead of as a web-app with the link above, please run the following code in the main directory:

`python app.py`

## Generate Your Own Dataset
Generating your own dataset is a great way to train an existing model. However, the format of the dataset must match what the model expects.

For the dataset, three different files in three separate folders are required. These are:

* `./img/` (stores the perturbed paper .png files; [850, 850, 3] pixels)

* `./img_msk/` (stores the binary black-white mask of where the paper is and where the background is; [850, 850, 3] pixels)

* `./flow/` (stores the 2D flow maps of the perturbations applied to the source images; [2, 850, 850])

To get a quick look of how the perturbations and realistic lighting look, run and edit accordingly the following code:

`python generate_perturbations.py`

Our sample dataset (2700 images, about 12 GB) is located [here](https://drive.google.com/file/d/1CA6YbR_N1gXBOYSqL5V9Zih7dudRGMLk/view?usp=sharing).

To see the math worksheet solved, please run the following code:

`python generate_answer_key.py --mathWorksheet pathToWorksheet.png`

where the path to the worksheet is a full path.

## Deep Learning U-Net Model Reference
[DocProj Project Page](https://xiaoyu258.github.io/projects/docproj/)

Please fork their repository and follow the instructions there to train and evaluate your dataset/model.

For completing the Transfer Learning portion, read the model_geoNet.pkl file for the model.state_dict dictionary and read that in into the modelGeoNet.py file instead of initializing the weights and biases as seen in Lines 163-169. Utilize the `load_dict` method and its argument is `model.state_dict`. This code can be run using a bash script. Make sure to have downloaded the Graphcut executable file from the forked GitHub project.

## Contact
Arnav Sharma - arnavsha@umich.edu

Austin Jeffries - ajeffr@umich.edu

Ali Badreddine - abadredd@umich.edu
