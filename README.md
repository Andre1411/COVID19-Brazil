# COVID19-Brazil
This repo contains files related to a Uni group project on **machine learning** applied to a datased of COVID19 cases from Brazil. The aim was to investigate relevant questions such as predicting if a patient is in need of intensinve care unit (ICU) and if dying for covid19 is related to any subject characteristics or health situation and so if it is possible to predict the outcome from the other covariates.
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
## NAs considerations:
The complete dataset contains 688040 subjects; however, it contains lots of NAs / unknowns
(27,24% of the cells in the data set are NAs / unknowns).
Since it is a big dataset, the first thing we did was to remove all the NAs.
Does removing NAs produce a deviation of the original distribution of the dataset?
We checked the distribution of all the covariates before and after removing the unknowns and we
found that:  
- Almost all the covariates keep the original class proportions, meaning that the
predominant class in the full dataset is also the most prevalent in the reduced one.
- the variable recording the greatest change is cardio.dis, from 33.81% class 1 in the full
dataset to 54.83% class 1 in the reduced one.
Consequently, we can affirm that the reduced dataset is not deviated by NAs removal.
After the remotion of all the NAs and unknowns the dataset contains 111801 subjects.
