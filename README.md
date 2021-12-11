<img src="https://user-images.githubusercontent.com/83367900/145607610-c5cef8f0-3684-43f3-8887-c4267e692378.png" width="100%" height="100%">

# Hedge Fund X: Financial Modeling Challenge
### Michael S. Bonetti
#### Zicklin School of Business
#### CUNY Bernard M. Baruch College
#
### Brief Description

This project aims to apply classification models, using R, on the Hedge Fund X Dataset, to analyze model performance, runtimes, and variable importance. This dataset is available on Kaggle's website, summarizing a heterogeneous set of financial products, made available via an ML competition through Signate of Japan, circa 2017 / 2018.\
https://www.kaggle.com/datasicencelab/hedge-fund-x-financial-modeling-challenge/version/1#

The dataset contains 10,000 observations, with 91 attributes (all numerical), 2 non-predictive, and 1 target (𝑡𝑎𝑟𝑔𝑒𝑡) variables. For this project, a 10% random sample (RS) was taken, using 1,000 observations and 88 predictive attributes. Two distinct runs were performed: one with 1,000 statistic observations for reproducible results (R1), and another with 1,000 randomly-chosen observations for varying results (R2), whereby I made comparisons between the two runs throughout the PowerPoint slide deck.

#
### Data Pre-Processing & Imbalance Ratio (IR)
Fortunately, as all of the attributes were numerical and the dataset was relatively balanced, not much (if any) data pre-processing was necessary. The imbalance ratio (IR):
* For the original dataset was 𝑛<sub>0</sub> = 4,994, 𝑛<sub>1</sub> = 5,006 ∴ 𝑛<sub>0</sub> / 𝑛<sub>1</sub> = 99.76%
* For Run 1: 𝑛<sub>0</sub><sup>∗</sup> = 512, 𝑛<sub>1</sub> = 488 ∴  𝑛<sub>1</sub> / 𝑛<sub>0</sub> = 95.31%
* For Run 2: 𝑛<sub>0</sub> = 473, 𝑛<sub>1</sub><sup>∗</sup> = 527 ∴  𝑛<sub>0</sub> / 𝑛<sub>1</sub> = 89.75%

Additionally, some preliminary exploratory data analysis (EDA) boxplots were created, to better visualize attribute importance and weight. Furthermore, the target variable y-distribution was taken and appears to be slightly skewed, but otherwise normal.

#
###  Creating Machine Learning (ML) Models

#### AUCs
A 90/10 split of the training and testing set was performed to fit 6 classification models (Logistic, LASSO, Elastic-net (EN), Ridge, Random Forest (RF), and Radial Support Vector Machines (SVM)) 50 times for 50 samples. The AUCs and runtimes were recorded, with boxplots to visualize these results:
* 0.9𝑛 AUC training
  * SVM and RF medians at, or near, 1,
* 0.9𝑛 AUC testing
  * Larger variances overall compared to AUC training boxplots,
  * Logistic, Elnet, LASSO, and Ridge medians are close,
  * SVM and RF higher,
* 0.9𝑛 training errors
  * Near mirror-image of AUC training boxplots, with ν<sub>EN</sub> slightly smaller,
  * SVM and RF medians at, or near, 0,
* 0.9𝑛 testing errors
  * Larger variances overall; medians for SVM and RF are still lower.

<img src="https://user-images.githubusercontent.com/83367900/145630609-76271709-6613-4e8d-bcdf-fc74b23fe0e5.png" width="45%" height="45%"> <img src="https://user-images.githubusercontent.com/83367900/145630912-521940dd-c968-4188-bc6b-5fcd21f3cc33.png" width="36%" height="36%">

#### Cross-validation Curves
For one of the 50 samples, misclassification error 10-fold cross-validation (CV) curves, using EN, LASSO, and Ridge, were done with LASSO performing the best in R1, but Ridge outperforming in R2. Overall, log(λ) and runtimes were generally the same, with EN, LASSO, and Ridge producing similarly-shaped upward-sloping graphs, albeit ungracefully.\
<img src="https://user-images.githubusercontent.com/83367900/145612701-4539299f-9105-435f-86f0-db3492d0db1f.png" width="30.05%" height="30.05%">
<img src="https://user-images.githubusercontent.com/83367900/145612863-40bcfb24-7c72-4eca-9f5e-f97169be783c.png" width="30.05%" height="30.05%">
<img src="https://user-images.githubusercontent.com/83367900/145612913-033bb53b-02fa-45f5-93e2-b12746bce837.png" width="30.05%" height="30.05%">

#
###  Performance and Runtimes
Upon observation, the average performance (training error rate) was about 0.422, processing at fast runtimes, regardless. However, the AUCs yielded no better than 50 / 50, meaning there was a slight trade-off with performance compared to runtime. RF consistently performed the best, albeit with a runtime of +3.50 secs, with SVM taking the longest, and with minimal performance improvement.\
<img src="https://user-images.githubusercontent.com/83367900/145629221-3cfbf1d7-09ff-4a49-b922-a6b4601f7d2a.png" width="40%" height="40%">\
<img src="https://user-images.githubusercontent.com/83367900/145629272-f6b02a0b-f37f-4650-8bbb-3a4439a4b975.png" width="40%" height="40%">

#
###  Variable Importance
Standardizing the estimated coefficients allows for its visualization (RF) and variable importance (LASSO, EN, Ridge) bar plots to be generated. Variables c69, c27, and c80 were the top 3 influencers for RF.\
<img src="https://user-images.githubusercontent.com/83367900/145630136-a33debf3-326a-453a-a48e-72cf0c3f19d2.png" width="45%" height="45%">
<img src="https://user-images.githubusercontent.com/83367900/145629056-6e53cbab-bec4-49b2-a2ec-e0f29cc52a0c.png" width="45%" height="45%">

#
###  Results

#### Variable Importance (Top 3)
* The top 3 positive influencers are c85, c17, and c45
* The top 3 negative influencers are c70, c81, and c80
<img src="https://user-images.githubusercontent.com/83367900/145613386-be596b4d-83b2-4c20-bed2-7dba094741dc.png" width="45%" height="45%">

#### However...
* There are too many unknowns, where it cannot be ascertained which specific funds and/or stocks affected performance,
* Overall, the results are no better than a 50/50 coin toss, except for RF and SVM AUCs.

#### Improvements can be made…
* RF & SVM (best performers), AdaBoost, SGD, kNN, NB,
#### … but the financial market is unpredictable!
* Therefore, this makes financial modeling decidedly more difficult to perform.

#
### Closing Thoughts
While financial behavior can be predicted, to a certain extent to ascertain some insights from the Hedge Fund X dataset, performance may increase under differing classification models, but probably not by much.  As such, although improvements can surely be made, the data _may_ be insufficient to adequately predict the future movement of financial products, unless certain unknowns, such as what the products are and the timeline, are made available.
