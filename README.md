# ML Final Project
- [Model Download Link](https://drive.google.com/file/d/1w2nkK2jngo0MQT92QuAgzX3ZN5pjwPpy/view?usp=sharing)
- [Kaggle Competition Link](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/leaderboard)

## Project Introduction
- This project is aim to solve the above Kaggle competition
- Score
    - public leaderboard: 0.59126
    - private leaderboard: 0.59057
    ![](https://hedgedoc.linyz.org/uploads/134ea281-45df-4b31-9085-9b2da049523a.png)


### Brief Summary of My Work
- Data Preprocessing
    - Aggregation of features
    - Creation of features
- Model
    - Using Multiple Linear Regression Model to prevent overfitting

### Environment
- python version: `3.6.9`
- requirements
    ```
    joblib==1.1.1
    numpy==1.19.5
    pandas==1.1.5
    scikit_learn==1.2.0
    ```

## Inference -- Reproduce the Result
- To reproduce the results of the result, please following steps below

### Steps
- Summary of steps
    1. download necessary files
    2. create virtual environment
    3. install requirements
    4. run inference code
#### Download necessary files
1. [model](https://drive.google.com/file/d/1w2nkK2jngo0MQT92QuAgzX3ZN5pjwPpy/view?usp=sharing), download link is also write on the top of README
2. Kaggle Competition Dataset, you can also get the dataset by accessing [competition page](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/leaderboard)
3. Clone the repository with following commands
    ```
    git clone https://github.com/lyz508/NYCU-ML-FinalProject
    ```
4. Put datas under the repository directory, structure should be like:
    ![](https://hedgedoc.linyz.org/uploads/69ef8e27-2f2e-43dd-b63c-83fc8647e2dd.png)
    
#### Create virtual environment
1. `cd` into the directory
    ```
    cd NYCU-ML-FinalProject
    ```
2. create virtual environment with `virtualenv`
    ```
    virtualenv -p /usr/bin/python3 virtual
    ```
3. activate the virtual environment
    ```
    source virtual/bin/activate
    ```

#### Install requirements
- install requirements.txt
    ```
    pip3 install -r requirements.txt
    ```

#### Run inference code
- run inference code with `python3`
    ```
    python3 109550129_Final_inference.py
    ```
- it should generate `109550129_submission.csv`
- upload to the competition to check the result
    ![](https://hedgedoc.linyz.org/uploads/d2c672b2-2c69-4230-bbb4-e000d1a06d83.png)