# COVID19-Brazil
University group project for a **machine learning** course on a datased of COVID19 cases from Brazil.
## Objective
The aim was to investigate relevant questions such as predicting if a patient is in need of intensinve care unit (ICU) and if dying for covid19 is related to any subject characteristics or health situation and so if it is possible to predict the outcome from the other covariates.
## Dataset features
- *first_smpts:* date of first symptoms  
- *week_smpts:* week of first symptoms  
- *sex:* sex M/F/unknown(I)  
- *date.birth:* date of birth  
- *age:* age in years  
- *pregnant:* 1-2-3 trimester, 4:unknown trimester but pregnant, 5: no, 6:male or age<10y, 0 or 9:unknown  
- *ethnicity:* 1:white 2:black 3:yellow 4:brown 5:indigenous, 9:unknown  
- *region:* Region of residence  
- *fever:* 1:yes, 2:no, 9:unknown  
- *cough:* 1:yes, 2:no, 9:unknown  
- *dyspnea:* 1:yes, 2:no, 9:unknown  
- *resp.disc:* Respiratory Discomfort 1:yes, 2:no, 9:unknown  
- *saturation:* oxygen saturation<95%? 1:yes, 2:no, 9:unknown  
- *diarrhea:* 1:yes, 2:no, 9:unknown  
- *vomit:* 1:yes, 2:no, 9:unknown  
- *abdom.pain:* abdominal pain 1:yes, 2:no, 9:unknown  
- *fatigue:* 1:yes, 2:no, 9:unknown  
- *loss.smell:* loss of smell 1:yes, 2:no, 9:unknown  
- *loss.taste:* loss of taste 1:yes, 2:no, 9:unknown  
- *other.symp:* other symptoms 1:yes, 2:no, 9:unknown  
- *cardio.dis:* chronic cardiovascular disease 1:yes, 2:no, 9:unknown  
- *hepatic.dis:* chronic liver disease 1:yes, 2:no, 9:unknown  
- *pneumo.dis:* Other Chronic Pneumatopathy 1:yes, 2:no, 9:unknown  
- *asthma:* 1:yes, 2:no, 9:unknown  
- *diabetes:* 1:yes, 2:no, 9:unknown  
- *obesity:* 1:yes, 2:no, 9:unknown  
- *hospital:* Hospitalization? 1:yes, 2:no, 9:unknown  
- *ICU:* # admitted to ICU? 1:yes, 2:no, 9:unknown  
- *outcome:* 1:cure 2:death 3:death of other cause, 9: unknown  
## NAs considerations
27,24% of the cells in the dataset are NAs or unknowns.
Since it contains 688040 subjects, it has been decided to remove all the rows (subjects) containing covariates with NAs.
It has been checked that the covariates' distribution doesn't change by doing so, finding that all the covariates keep their original class proportions: the
predominant class in the full dataset is the most prevalent also in the reduced dataset.
After the remotion of all the NAs and unknowns the dataset contains 111801 subjects.
## Data pre-processing
After the NaN and Unknown values removal, the values of the variable "*outcome*" (1,2,3) were unbalanced: class 3,
referring to non-covid-death, occurred in only the 0.4% of the subjects, so it was hard to be
predicted by the models. For this reason it has been decided to focus only on covid-death and to not consider other possible death reasons.  
The next step was to reduce the size of the dataset for computational resons. A quick analysis using reduced dataset with
several proportions (75%,50%,25%) and Logistic Models was performed, comparing the outcomes to ensure that using the reduced dataset wouldn't influence the goodness of the
results. In the end it has been decided to work with 25 % of data, randomly sampled from the original dataset.
## Dataset splitting
The reduced dataset was split into a Training Set (70%) and several Test Sets, containing 100,80,60,40,20 and 10 % of the remaining data of the reduced
dataset respectively.
This decision was made in order to check if a smaller test set can cover
the variability of the original data.
If the performance of different models is comparable in all the test sets, it proves that the variance
is covered. (https://www.baeldung.com/cs/train-test-datasets-ratio)
For each model, mean accuracy, sensitivity and specificity across all the testing sets were computed.
## Inferential analysis
ICU and outcome are the target variables to be predicted using information about the other covariates.
A perfect ideal classificator would predict the outcomes correctly with 100% accuracy, but a real one will for sure make some mistake.
This leads to an important question:
Is it worse predict “yes” (for example that a patient will go into ICU) when instead the true class is
“no” (the patient will not go into ICU) or vice versa?
In one case it would underestimate the real necessity of ICU units, on the other one it woud overestimate
it. Answering this question is not trivial, so it has been decided to maintain a good balance between
the two types of error.
## Model formula selection
The covariates to be included in our models have been selected by creating a formula with all the covariates, fitting a logistic model and selecting just the predictors with a statistically significant estimated coefficient.
On top of that, it has been verified if it was possible to simplify even further the model removing
some covariates and evaluating the different solutions with AIC and BIC index.  
### ICU  
| Model formula | Mean accuracy | Mean sensitivity | Mean specificity | AIC | BIC | Selected |
| ------------- | ------------- | ---------------- | ---------------- | --- | --- | -------- |
| mod.form1.1 | 0.602 | 0.876 | 0.199 | 25894.80 | 25957.85 | no |
| mod.form1.2 | 0.611 | 0.825 | 0.294 | 25639.10 | 25875.59 | no |
| mod.form1.3 | 0.606 | 0.820 | 0.291 | 25629.49 | 25810.77 | no |
| mod.form1.4 | 0.607 | 0.819 | 0.294 | 25644.28 | 25730.97 | yes |
| mod.form1.5 | 0.609 | 0.855 | 0.246 | 25837.57 | 25908.50 | no |
| mod.form1.6 | 0.604 | 0.829 | 0.273 | 25672.03 | 25758.72 | yes |
| mod.form1.7 | 0.601 | 0.880 | 0.189 | 25902.69 | 25949.98 | no |
| mod.form1.8 | 0.603 | 0.867 | 0.215 | 25877.80 | 25932.97 | no |
| by chance | 0.577 ||||||
### OUTCOME  
| Model formula | Mean accuracy | Mean sensitivity | Mean specificity | AIC | BIC | Selected |
| ------------- | ------------- | ---------------- | ---------------- | --- | --- | -------- |
| mod.form2.1 | 0.619 | 0.944 | 0.093 | 25425.21 | 25480.35 | no |
| mod.form2.2 | 0.738 | 0.835 | 0.580 | 20527.60 | 20771.80 | yes |
| mod.form2.3 | 0.733 | 0.828 | 0.572 | 20563.04 | 20752.10 | no |
| mod.form2.4 | 0.735 | 0.832 | 0.577 | 20544.14 | 20693.82 | yes |
| mod.form2.5 | 0.735 | 0.833 | 0.578 | 20567.77 | 20717.44 | no |
| mod.form2.6 | 0.710 | 0.795 | 0.570 | 22365.46 | 22412.72 | no |
| by chance | 0.618 ||||||  

In the end, two formulas have been selected for each target variable (outcome and ICU) and they have been tried in all the other
approaches beyond logistic regression.
## MODEL USED FOR THE ANALYSIS
### LOGISTIC AND MULTINOMIAL MODELS
Logistic model was the first approach that we tried, and a useful information that you can derive
from this model is the variation of the probability of being classified in a certain class respect
another class when the value of a specific covariate changes.
So, we find interesting answering the following questions:
Everyone know that if a patient goes into ICU its heath is really compromised, therefore the risk of
dying increase, however how much does it increase?
We find that the probability of dying of covid19 respect the probability of recovering is 5.44
greater if a patient goes into ICU.
ODDS RATIO of P(outcome = 1)/P(outcome = 0) is 5.44 greater if a patient go into ICU (ICU goes
from 0 to 1).
In order to select the threshold for classify the two classes (0,1), we use the Roc Curve method.
The threshold selected by the ROC is where TPR+FPR is maximized. We decide to use it since, as
said previously, we are looking about a model with a good balance between sensitivity and
specificity.
Doing that, we lose something in the overall accuracy respect other threshold due to the not
symmetric distribution of the two classes.
Even if both ICU and outcome have just 2 classes, we fit a multinomial classification (only on
outcome) just to see if the results were the same of the logistic model.
From the table at point 4.3, you can see that the outcomes are slightly different.
We had looks about the coefficient estimated by the two approaches: they are close.
Therefore, we think that the discrepancy is due to different estimation algorithms.
###  TREE-BASED MODEL
Decision tree is a classification algorithm that has one important feature compared with the other
approaches: it can produce a graphical visualization of the process through which the classification
is performed.
This output can really be useful because gives the possibility to have an easy interpretation on
what is going on to a person that is not an expert of statistic and machine learning.
So, our idea is to use this approach to produce a graph able to tell a physician which are the most
relevant subjects characteristic that influence the probability of going into ICU, for example.
However, to achieve an accuracy comparable with the other methods, the tree has to be made by
one hundred nodes; therefore, a graphical representation of such big tree is not easily
understandable.
So, is it worth using tree-based model? No, since in addiction of not being able of produce a useful
graph, also the classification performance is not good (see point 4.3).
### SVM MODEL
We tried a support vector machine approach since usually it has great performances. Because of
the high computational effort required to run the method with the reduced dataset, we decide to
down-sample the data considering only the 25 % of the whole ones. We use the same proportions
to create the training set and the testing sets. In order to obtain the most suitable model we
perform the tuning of the models using several types of kernels (“linear”, “polynomial”, “radial”)
and different parameters. From the best ones we derive the performance.
### BOOSTING MODELS
Since the performances of the previous models were good but not enough satisfying, we decided
to try out this method to check if a combination of week learners could do better than a single
strong one.
### RANDOM FOREST models
Random forest (RF) is part of the bootstrap aggregation family, which has the goal of improving
the prediction respect standard decision trees and to find a way to deal with a huge number of
predictors.
In particular, random forest selects randomly a subset of covariates to building each tree.
This approach reduces tree similarity, increase flexibility, and hopefully leads to a better
prediction.
Furthermore, RF produces a useful graph indicating which are the variables that better reduce
impurity when used for splitting. The ones resulting from our model are the followings:
At this point, we found interesting to understand if the same predictors were the most important
also for the other models.
So, we build the following table, comparing them to the covariates used in the first splits of the
decision tree model and the ones with the highest absolute value of the coefficients estimated by
the logistic model (of course only if statistically significant for the model).
#### Most important predictors
|| ICU | outcome |
| ------------- | ------------- | ------------- |
| Logistic | sex, obesity, saturation, dyspnea, cough | ICU, pregnant |
| Decision tree | saturation, dyspnea, age | ICU, age |
| Random forest | obesity, cough, saturation | ICU, age |  

Overall, it turns out that almost the same predictors were selected to be the most important,
consequently we can affirm with more certainty which variables are the most relevant.
