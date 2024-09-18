# Pollution-forecasting

Breve descripción del proyecto, qué hace y para qué sirve.

## Table of contents

- [Instalation](#instalation)
- [Structure](#Structure)
- [Usage](#usage)
- [Licencia](#licencia)

## Instalation

1. Clone the repository:

   ```bash
   git clone https://github.com/tu-usuario/nombre-del-proyecto.git

2. Navigate to the project directory:

   ```bash
   cd pollution-forecasting

3. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt

5. Download the data from [this Google Drive link](https://drive.google.com/drive/folders/18CrTp28JYs7zGxPOtbe0yMc6bzcMLqPX?usp=sharing).


## Structure
The project structure must be as follows:

pollution-forecasting/  
├── folder_structure.py  
├── ML_DL/  
│   ├── correspondences/  
│   │   ├── correspondences.csv  
│   │   └── correspondencesPaper.csv  
│   ├── DATA/  
│   │   ├── 2022_24.csv  
│   │   └── 2023_24.csv  
│   ├── datasets/  
│   │   ├── FastPollutionDataset.py  
│   │   └── PaperDataset.py  
│   ├── Mad_Station/  
│   │   ├── Mad_Station_2019.csv  
│   │   └── Mad_Station_2022.csv  
│   ├── methods_paper/  
│   │   ├── infer_models_paper.py  
│   │   ├── ML_paper.py  
│   │   └── train_models_paper.py  
│   ├── ML_methods/  
│   │   └── train_models.py  
│   ├── models/  
│   │   └── NNs.py  
│   ├── NN_paper_results/  
│   │   └── info_results_paper.csv  
│   ├── nns/  
│   │   ├── inference_nn.py  
│   │   ├── train_functions.py  
│   │   └── train_nn.py  
│   ├── utils/  
│   │   ├── cast_to_precision.py  
│   │   ├── generic_utils.py  
│   │   ├── metrics_utils.py  
│   │   ├── normalize.py  
│   │   ├── os_utils.py  
│   │   ├── scikit_utils.py  
│   │   └── torch_utils.py  
│   └── utils_os/  
│       ├── models_saver.py  
│       └── results_saver.py  
├── pollution_DiffModel/  
│   ├── config/  
│   │   └── NO2.py  
│   ├── correspondences/  
│   │   ├── correspondences.csv  
│   │   └── correspondencesPaper.csv  
│   ├── DATA/  
│   │   ├── 2022_24.csv  
│   │   └── 2023_24.csv  
│   ├── dataloaders/  
│   │   ├── DiffusionDataloader.py  
│   │   ├── DiffusionPollutionDataset.py  
│   │   └── PaperDataset.py  
│   ├── diffuser/  
│   │   ├── datasets/  
│   │   │   ├── buffer.py  
│   │   │   ├── d4rl.py  
│   │   │   ├── normalization.py  
│   │   │   ├── preprocessing.py  
│   │   │   └── sequence.py  
│   │   ├── models/  
│   │   │   ├── diffusion.py  
│   │   │   ├── helpers.py  
│   │   │   └── temporal.py  
│   │   ├── sampling/  
│   │   │   ├── functions.py  
│   │   │   ├── guides.py  
│   │   │   └── policies.py  
│   │   └── utils/  
│   │       ├── arrays.py  
│   │       ├── cloud.py  
│   │       ├── colab.py  
│   │       ├── config.py  
│   │       ├── git_utils.py  
│   │       ├── iql.py  
│   │       ├── logger.py  
│   │       ├── progress.py  
│   │       ├── serialization.py  
│   │       ├── setup.py  
│   │       ├── tap.py  
│   │       ├── timer.py  
│   │       ├── training.py  
│   │       ├── transformations.py  
│   │       ├── typing_inspect.py  
│   │       └── video.py  
│   ├── Mad_Station/  
│   │   ├── Mad_Station_2019.csv  
│   │   └── Mad_Station_2022.csv  
│   ├── scripts/  
│   │   ├── generate_NO2_curves.py  
│   │   ├── launch_train.sh  
│   │   └── train_NO2.py  
│   └── setup.py  
├── README.md  
└── requirements.txt  


## Usage
All the code corresponding to the diffusion model is located in the folder **pollution_DiffModel**. To run this model, the file 
   ```bash
   pollution_DiffModel/scripts/train_NO2.py
   ```

must be executed.  

The rest of the ML and DL models are located in the **ML_DL** folder. For ML methods, 
   ```bash
   ML_DL/ML_methods/train_models.py
   ```

must be executed, and for DL neural networks, 
   ```bash
   ML_DL/nns/train_nn.py
   ML_DL/nns/inference_nn.py
   ```

must be executed.

For each of the contaminants, the list of valid locations appears below

| Name                     | Code | Locations                                   |
|--------------------------|------|---------------------------------------------|
| NO                       | 1    | (16, 1), (171, 1), (47, 2), (5, 2), (92, 5) |
| SO<sub>2</sub>           | 6    | (16, 1), (171, 1), (45, 2), (5, 2), (92, 5) |
| CO                       | 7    | (120, 1), (123, 2), (133, 2), (148, 4), (14, 2), (16, 1), (171, 1), (180, 1), (45, 2), (47, 2), (49, 3), (5, 2), (58, 4), (6, 4), (65, 14), (67, 1), (7, 4), (80, 3), (74, 7), (92, 5), (9, 1) |
| NO<sub>2</sub>           | 8    | (123, 2), (133, 2), (14, 2), (148, 4), (16, 1), (171, 1), (180, 1), (45, 2), (47, 2), (49, 3), (5, 2), (58, 4), (6, 4), (65, 14), (67, 1), (7, 4), (74, 7), (80, 3), (9, 1), (92, 5) |
| PM<sub>2.5</sub>         | 9    | (120, 1), (148, 4), (16, 1), (171, 1), (180, 1), (47, 2), (49, 3), (5, 2), (65, 14), (7, 4), (74, 7), (9, 1) |
| PM<sub>10</sub>          | 10   |(120, 1), (123, 2), (133, 2), (14, 2), (148, 4), (16, 1), (171, 1), (45, 2), (49, 3), (5, 2), (58, 4), (65, 14), (6, 4), (67, 1), (74, 7), (80, 3), (92, 5)|
| NO<sub>x</sub>           | 12   |(120, 1), (123, 2), (133, 2), (14, 2), (148, 4), (16, 1), (171, 1), (180, 1), (45, 2), (47, 2), (49, 3), (5, 2), (58, 4), (6, 4), (65, 14), (7, 4), (67, 1), (74, 7), (80, 3), (9, 1), (92, 5)|
| O<sub>3</sub>            | 14   |(120, 1), (123, 2), (133, 2), (148, 4), (16, 1), (171, 1), (180, 1), (45, 2), (47, 2), (49, 3), (5, 2), (58, 4), (6, 4), (65, 14), (67, 1), (7, 4), (74, 7), (80, 3), (9, 1), (92, 5)
| Toluene                  | 20   |(16, 1), (47, 2), (58, 4), (6, 4)|
| Benzene                  | 30   |(16, 1), (47, 2), (58, 4), (6, 4)|
| Total hydrocarbons       | 42   |(16, 1), (47, 2), (58, 4), (6, 4)|
| Non-methane hydrocarbons | 44   |(16, 1), (47, 2), (58, 4), (6, 4)|
