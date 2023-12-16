# Building-a-Simple-Linear-Regression-Machine-Learning-Model-for-CO2-Emission-Analysis
Building a Simple Linear Regression Machine Learning Model for CO2 Emission Analysis

## Introduction

In response to the growing imperative for sustainable practices in the automotive industry, the development of predictive models for carbon dioxide (CO2) emissions has become a pivotal focus. This document presents a comprehensive overview of a Machine Learning (ML) project aimed at predicting CO2 emissions from cars. The project leverages a **linear regression model**, utilizing the latest dataset sourced from the **Government of Canada.**

The core objective of this initiative is to create a tool that aids in **estimating the CO2 emissions of new cars**, with a specific emphasis on key factors influencing environmental impact—**engine size and the number of cylinders**. By employing advanced data analytics and machine learning techniques, this model not only addresses the immediate need for accurate emission predictions but also aligns with broader sustainability goals within the automotive sector.

This documentation serves as a transparent and detailed account of the entire machine learning project, from its conceptualization to the deployment of the predictive model. Through a structured approach encompassing problem definition, data collection, exploratory data analysis, modeling, and beyond, **this document provides insights into the decision-making processes, methodologies employed, and the model's performance.**

**The dataset, sourced directly from the Government of Canada**, ensures the incorporation of the latest and most relevant information, contributing to the model's accuracy and relevance in the context of contemporary automotive emissions standards. As we delve into the intricacies of the project, we aim to foster transparency, reproducibility, and collaboration, promoting a holistic understanding of the methodologies and outcomes achieved.

This initiative represents a step towards informed decision-making within the automotive industry, empowering stakeholders with a tool that aligns with environmental sustainability objectives. Through open documentation and shared knowledge, we hope to contribute to the collective efforts aimed at reducing the carbon footprint of vehicular emissions.

# Problem Definition
  ## Problem Statement

The automotive industry plays a significant role in contributing to environmental challenges, with carbon dioxide (CO2) emissions from vehicles being a primary concern. The problem at hand revolves around the need for a reliable and accurate method to predict CO2 emissions from new cars. As the industry undergoes rapid changes and strives to meet stringent environmental standards, having a robust predictive model becomes crucial for manufacturers, regulators, and consumers alike.

The primary challenge is to develop a machine learning model capable of estimating CO2 emissions with precision, considering key influencing factors. Engine size and the number of cylinders have been identified as pivotal parameters affecting emissions, necessitating a predictive tool that can analyze these variables and provide actionable insights. **The goal is to create a model that not only meets the immediate need for accurate predictions but also aligns with the broader sustainability objectives within the automotive sector.**


# Data Collection

## Data Description

The dataset contains information related to various vehicle models, including details about their make, model, vehicle class, engine specifications, transmission type, fuel type, fuel consumption metrics, and environmental attributes. 
Here's a breakdown of the key features:

1. **Year:**
   - The year in which the vehicle model is designated.

2. **Make:**
   - The brand or manufacturer of the vehicle.

3. **Model:**
   - The specific model designation of the vehicle.

4. **Vehicle Class:**
   - The classification of the vehicle, indicating its size or type (e.g., Full-size, SUV: Small, SUV: Standard).

5. **Engine Size (L):**
   - The size of the vehicle's engine, measured in liters.

6. **Cylinders:**
   - The number of cylinders in the vehicle's engine.

7. **Transmission:**
   - The type of transmission system used in the vehicle (e.g., AV7, AS10, M6).

8. **Fuel Type:**
   - The type of fuel used by the vehicle (e.g., Z).

9. **Fuel Consumption:**
   - - **City (L/100 km):**
        - The fuel consumption rate in liters per 100 kilometers during city driving.
     - **Hwy (L/100 km):**
        - The fuel consumption rate in liters per 100 kilometers during highway driving.
     - **Comb (L/100 km):**
        - The combined fuel consumption rate in liters per 100 kilometers (averaged city and highway).
     - **Comb (mpg):**
        - The combined fuel consumption rate in miles per gallon.

10. **CO2 Emissions:**
    - - **CO2 (g/km):**
        - The carbon dioxide emissions in grams per kilometer.
    - **CO2 Rating:**
        - The rating assigned to the vehicle based on its CO2 emissions.

11. **Smog Rating:**
    - The smog rating assigned to the vehicle.

The dataset appears to provide a detailed overview of vehicle characteristics, fuel efficiency, and environmental impact, making it suitable for analysis and modeling related to carbon emissions and fuel consumption.


# Exploratory Data Analysis (EDA)

### Reading the data in

### Data Exploration
Let's first have a descriptive exploration on our data.

### Now, let's plot each of these features against the Emission, to see how linear their relationship is:



<img width="763" alt="Screenshot 2023-12-16 at 2 42 51 PM" src="https://github.com/git-shashank-hp/Building-a-Simple-Linear-Regression-Machine-Learning-Model-for-CO2-Emission-Analysis/assets/144894099/b5ac5216-4336-4ea3-aecd-4a05d5f5e473">

<img width="726" alt="Screenshot 2023-12-16 at 2 47 44 PM" src="https://github.com/git-shashank-hp/Building-a-Simple-Linear-Regression-Machine-Learning-Model-for-CO2-Emission-Analysis/assets/144894099/8679f72b-0287-47f6-8bce-64dc0e46ba15">

<img width="819" alt="Screenshot 2023-12-16 at 2 47 52 PM" src="https://github.com/git-shashank-hp/Building-a-Simple-Linear-Regression-Machine-Learning-Model-for-CO2-Emission-Analysis/assets/144894099/f6b1961c-c6d0-4707-8819-59002945f86f">


## Creating train and test dataset

Train/Test Split involves splitting the dataset into training and testing sets that are mutually exclusive. After which, you train with the training set and test with the testing set. 
This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that have been used to train the model. Therefore, it gives us a better understanding of how well our model generalizes on new data.

This means that we know the outcome of each data point in the testing dataset, making it great to test with! Since this data has not been used to train the model, the model has no knowledge of the outcome of these data points. So, in essence, it is truly an out-of-sample testing.

Let's split our dataset into train and test sets. 80% of the entire dataset will be used for training and 20% for testing. We create a mask to select random rows using __np.random.rand()__ function: 


### Simple Regression Model
Linear Regression fits a linear model with coefficients B = (B1, ..., Bn) to minimize the 'residual sum of squares' between the actual value y in the dataset, and the predicted value yhat using linear approximation. 

<img width="702" alt="Screenshot 2023-12-16 at 2 49 15 PM" src="https://github.com/git-shashank-hp/Building-a-Simple-Linear-Regression-Machine-Learning-Model-for-CO2-Emission-Analysis/assets/144894099/4295450e-10ae-44e4-8e02-ba03ff2d892c">




## Modeling


### Using sklearn package to model data.


As mentioned before, __Coefficient__ and __Intercept__ in the simple linear regression, are the parameters of the fit line. 
Given that it is a simple linear regression, with only 2 parameters, and knowing that the parameters are the intercept and slope of the line, sklearn can estimate them directly from our data. 
Notice that all of the data must be available to traverse and calculate the parameters.


### Plot outputs


As mentioned before, __Coefficient__ and __Intercept__ in the simple linear regression, are the parameters of the fit line. 
Given that it is a simple linear regression, with only 2 parameters, and knowing that the parameters are the intercept and slope of the line, sklearn can estimate them directly from our data. 
Notice that all of the data must be available to traverse and calculate the parameters.

<img width="748" alt="Screenshot 2023-12-16 at 2 50 41 PM" src="https://github.com/git-shashank-hp/Building-a-Simple-Linear-Regression-Machine-Learning-Model-for-CO2-Emission-Analysis/assets/144894099/67f68308-021b-4ec1-be83-a189212dcd74">


## Evaluation
We compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.

There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set: 
* Mean Absolute Error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.

* Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean Absolute Error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.

* Root Mean Squared Error (RMSE). 

* R-squared is not an error, but rather a popular metric to measure the performance of your regression model. It represents how close the data points are to the fitted regression line. The higher the R-squared value, the better the model fits your data. The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).


### Mean absolute error: 30.52
### Residual sum of squares (MSE): 1589.92
### R2-score: 0.63
