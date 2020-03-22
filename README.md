# AI4COVID

This project is a result of the [WirVsVirus Hackathon](https://wirvsvirushackathon.org/). Our objective was to assist doctors in diagnosing COVID-19 patients by training a convolutional network to discriminate between patients with and without the disease, based on an X-ray image. We use a DenseNet121 pretrained on the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) dataset and finetune it to the [covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset/tree/master). Furthermore, we implemented a simple prediction service that displays the diagnosis based on an X-ray image uploaded by the user.

![alt text](/res/examples/gui.png)

## How does it work?

1. We first trained the COVID-19 classifier on Google Colab. See the notebook file `covid_19.ipynb`.
2. We then implemented a simple backend server in Flask, that loads the classifier and performs inference on the X-ray image uploaded by the user.
3. Finally, we developed a simple frontend server that allows user to upload an X-ray image and see the results of diagnosis: the predicted probability of patient having COVID-19 and the heatmap of critical regions on the X-ray that contributed to the diagnosis.

## Setup

Requirements:

* python >=3.7.6
* NPM >=6.13.4

Running the application

1. `cd ai4covid` -- Enter project root directory
1. `chmod +x setup.sh run.sh` -- Add execute permission to the scripts
2. `./setup.sh` -- Install dependencies
3. `./run.sh` -- Run the backend and the frontend servers
