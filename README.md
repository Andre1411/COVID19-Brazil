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
ICU  
| Model formula | Mean accuracy | Mean sensitivity | Mean specificity | AIC | BIC | Model selected |
| ------------- | ------------- | ---------------- | ---------------- | --- | --- | ---------------|
OUTCOME  
| Model formula | Mean accuracy | Mean sensitivity | Mean specificity | AIC | BIC | Model selected |
| ------------- | ------------- | ---------------- | ---------------- | --- | --- | ---------------|
