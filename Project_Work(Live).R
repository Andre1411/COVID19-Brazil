################################
### 				0. DATA LOADING
rm(list=ls())
setwd("C:/Users/ASUS/OneDrive - Università degli Studi di Padova/uni/Magistrale/1° anno/2° - Machine learning for bioengineering/R_codes/covid19_brazil.data")
covid19 <- read.table('managedData.dat',header = T) # colClasses ='factor')
# A.A. : managedData is a file containing categorical variables with Os (No) and 1s (Yes) levels
################################
###					1. DATA FACTORING
covid19$sex <- as.factor(covid19$sex);covid19$pregnant <- as.factor(covid19$pregnant);covid19$race <- as.factor(covid19$race)
covid19$region <- as.factor(covid19$region);covid19$fever <- as.factor(covid19$fever);covid19$cough <- as.factor(covid19$cough)
covid19$dyspnea <- as.factor(covid19$dyspnea);covid19$resp.disc <- as.factor(covid19$resp.disc);covid19$saturation <- as.factor(covid19$saturation)
covid19$diarrhea <- as.factor(covid19$diarrhea);covid19$vomit <- as.factor(covid19$vomit);covid19$abdom.pain <- as.factor(covid19$abdom.pain)
covid19$fatigue <- as.factor(covid19$fatigue);covid19$loss.smell <- as.factor(covid19$loss.smell);covid19$loss.taste <- as.factor(covid19$loss.taste)
covid19$other.symp <- as.factor(covid19$other.symp);covid19$cardio.dis <- as.factor(covid19$cardio.dis);covid19$hepatic.dis <- as.factor(covid19$hepatic.dis)
covid19$pneumo.dis <- as.factor(covid19$pneumo.dis);covid19$asthma <- as.factor(covid19$asthma);covid19$diabetes <- as.factor(covid19$diabetes)
covid19$obesity <- as.factor(covid19$obesity);covid19$hospital <- as.factor(covid19$hospital);covid19$ICU <- as.factor(covid19$ICU)
covid19$outcome <- as.factor(covid19$outcome);covid19$week_smpts <- as.integer(covid19$week_smpts)
detach(covid19)
attach(covid19)
summary(covid19)
names(covid19)
################################
###					2. MODEL FORMULA (After Model Formula Analysis)
###		ICU PREDICTION
mod.form1.4 <- formula(ICU ~ sex+age+cough+dyspnea+resp.disc+saturation+diarrhea+fatigue+loss.taste+obesity)
mod.form1.6 <- formula(ICU~sex+age+cough+dyspnea+resp.disc+saturation+diarrhea+fatigue+obesity+pneumo.dis)
###		OUTCOME PREDICTION
mod.form2.2 <- formula(outcome ~ sex+age+pregnant+race+fever+cough+dyspnea+resp.disc+saturation +diarrhea+vomit+abdom.pain+fatigue+loss.smell+loss.taste+other.symp+cardio.dis+hepatic.dis+pneumo.dis+asthma+diabetes+obesity+ICU)		
mod.form2.4 <- formula(outcome ~ age+race+cough+dyspnea+resp.disc+saturation +diarrhea+fatigue+loss.taste+other.symp+cardio.dis+hepatic.dis+pneumo.dis+asthma+ICU)
################################
###					3. DATASET DIVISION
set.seed(123)
DATASET1 <- covid19[(sample(nrow(covid19),round(0.25*nrow(covid19)))),] # 25 % of the Entire Dataset
### Managing the CLASS 3 of outcome
#DATASET1$outcome <- as.factor(ifelse(DATASET1$outcome==2,2,1)) # 			Method 2 (V)	
####### A.A.: If not specified , we are considering the method 3 for binarize the class outcome #######
# 			Method 3 (V)
DATASET1 <- DATASET1[DATASET1$outcome != 3,]
#A.A. BINARIZATION OF OUTCOME
DATASET1$outcome <- as.factor(ifelse(DATASET1$outcome==1,0,1)) # CURE:0,DEATH for COVID19:1
print(' We are using the Method 3')
################################
###		Tranig/Testing Sets for DATASET1
set.seed(12)
idx1 <- sample(nrow(DATASET1),0.7*nrow(DATASET1))
TRN1 = DATASET1[idx1,]
TST1 = DATASET1[-idx1,]
TST1s <- list('NULL')
TSTsProp <- c(1,0.8,0.6,0.4,0.2,0.1)
TSTs1Dims <- round(TSTsProp*dim(TST1)[1],digits=0)
set.seed(10)
for (i in 1:length(TSTs1Dims)){
	TST1s[[i]] <- TST1[sample(nrow(TST1),round(TSTsProp[i]*nrow(TST1))),] 
}
################################
###					4.MODELS SELECTION (a ~ mod_form1 ,b ~ mod_form2)
###		PACKAGES
library(gbm)
library(caret)
library(randomForest)
library(pROC)
library(tree)
library(e1071)
library(nnet)
################################
###		PREDICTION BY CHANCE
### ICU:
Pred.Chance.ICU <- Pred.Chance.outcome<- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
for (i in 1: length(TSTs1Dims)){
	Pred.Chance.ICU[i] <- round(((sum(TST1s[[i]]$ICU==0))/TSTs1Dims[i]),digits=3)
}
print(round(mean(Pred.Chance.ICU),digits=3))
### Mean Pred by Chance(%): 	0.577 
###	Outcome:
for (i in 1: length(TSTs1Dims)){
	Pred.Chance.outcome[i] <- round(((sum(TST1s[[i]]$outcome==1))/TSTs1Dims[i]),digits=3)
}
print(round(mean(Pred.Chance.outcome),digits=3))
### Mean Pred by Chance(%): 	0.382 

##################################################################################
################################# LOGISTIC MODEL #################################
##################################################################################

################################
###		1.1.4 LOGISTIC MODEL based on DATASET1 (27950 samples) 
logistic1.1.4 <- glm(mod.form1.4,binomial(link='logit'),data= TRN1)
Acc1.1.4 <- Spec1.1.4 <- Sens1.1.4<- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
ans <- predict(logistic1.1.4,newdata = TST1s[[1]],type='response')
roc(TST1s[[1]]$ICU ~ ans, plot = T, print.auc = T,print.thres="best")
#### Model Formula: 	AUC 		Threshold 								
### mod.form1.4			0.623		  0.401										
th <- 0.401
for (i in 1: length(TSTs1Dims)){
	ans <- (predict(logistic1.1.4,newdata = TST1s[[i]],type='response') > th) 
	ans1 <- as.factor(ifelse(ans == 1,1,0))
	Acc1.1.4[i] <- round(sum(ans1 == TST1s[[i]]$ICU)/TSTs1Dims[i],digits=3)
	Sens1.1.4[i] <- round(sensitivity(ans1,TST1s[[i]]$ICU),digits=3)	
	Spec1.1.4[i] <- round(specificity(ans1,TST1s[[i]]$ICU),digits=3)		
}
print(round(mean(Acc1.1.4),digits=3))
print(round(mean(Sens1.1.4),digits=3))
print(round(mean(Spec1.1.4),digits=3))
### Model Formula:		Mean Accuracy	 	Mean Sensitivity 		Mean Specificity 
### mod.form1.4			      0.583				0.535					 0.648
### Mean Pred by Chance(%): 	0.577 
################################
###		1.1.6 LOGISTIC MODEL based on DATASET1 (27950 samples) 
logistic1.1.6 <- glm(mod.form1.6,binomial(link='logit'),data= TRN1)
Acc1.1.6 <- Spec1.1.6 <- Sens1.1.6<- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
ans <- predict(logistic1.1.6,newdata = TST1s[[1]],type='response')
roc(TST1s[[1]]$ICU ~ ans, plot = T, print.auc = T,print.thres="best")
#### Model Formula: 	AUC 		Threshold 												
### mod.form1.6			0.623			0.408					
th <- 0.408
for (i in 1: length(TSTs1Dims)){
	ans <- (predict(logistic1.1.6,newdata = TST1s[[i]],type='response') > th) 
	ans1 <- as.factor(ifelse(ans == 1,1,0))
	Acc1.1.6[i] <- round(sum(ans1 == TST1s[[i]]$ICU)/TSTs1Dims[i],digits=3)
	Sens1.1.6[i] <- round(sensitivity(ans1,TST1s[[i]]$ICU),digits=3)	
	Spec1.1.6[i] <- round(specificity(ans1,TST1s[[i]]$ICU),digits=3)		
}
print(round(mean(Acc1.1.6),digits=3))
print(round(mean(Sens1.1.6),digits=3))
print(round(mean(Spec1.1.6),digits=3))
### Model Formula:		Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form1.6			    0.591			0.569					  0.623
### Mean Pred by Chance(%): 	0.577 
################################
### 	1.2.2 LOGISTIC MODEL based on DATASET1 (27950 samples) 
logistic1.2.2 <- glm(mod.form2.2,binomial(link='logit'),data= TRN1)
Acc1.2.2 <- Sens1.2.2 <- Spec1.2.2 <- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
ans <- predict(logistic1.2.2,newdata = TST1s[[1]],type='response')
roc(TST1s[[1]]$ICU ~ ans, plot = T, print.auc = T,print.thres="best")
#### Model Formula: 	AUC 		Threshold 								
### mod.form2.2			0.946			0.418					
th <- 0.418					
for (i in 1: length(TSTs1Dims)){
	ans <- (predict(logistic1.2.2,newdata = TST1s[[i]],type='response') > th)
	ans1 <- as.factor(ifelse(ans == 1,1,0))
	Acc1.2.2[i] <- round(sum(ans1 == TST1s[[i]]$outcome)/TSTs1Dims[i],digits=3)
	Sens1.2.2[i] <- round(sensitivity(ans1,TST1s[[i]]$outcome),digits=3)	
	Spec1.2.2[i] <- round(specificity(ans1,TST1s[[i]]$outcome),digits=3)	
}
print(round(mean(Acc1.2.2),digits=3)) 
print(round(mean(Sens1.2.2),digits=3))
print(round(mean(Spec1.2.2),digits=3))
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form2.2			0.728				0.755					0.682
### Mean Pred by Chance(%): 	0.382 
################################
### 	1.2.4 LOGISTIC MODEL based on DATASET1 (27950 samples) 
logistic1.2.4 <- glm(mod.form2.4,binomial(link='logit'),data= TRN1)
Acc1.2.4 <- Sens1.2.4 <- Spec1.2.4<- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
ans <- predict(logistic1.2.4,newdata = TST1s[[1]],type='response')
roc(TST1s[[1]]$ICU ~ ans, plot = T, print.auc = T,print.thres="best")
#### Model Formula: 	AUC 		Threshold 												
### mod.form2.4			0.9476			0.417	
th <- 0.417		
for (i in 1: length(TSTs1Dims)){
	ans <- (predict(logistic1.2.4,newdata = TST1s[[i]],type='response') > th)
	ans1 <- as.factor(ifelse(ans == 1,1,0))
	Acc1.2.4[i] <- round(sum(ans1 == TST1s[[i]]$outcome)/TSTs1Dims[i],digits=3)
	Sens1.2.4[i] <- round(sensitivity(ans1,TST1s[[i]]$outcome),digits=3)	
	Spec1.2.4[i] <- round(specificity(ans1,TST1s[[i]]$outcome),digits=3)	
}
print(round(mean(Acc1.2.4),digits=3)) 
print(round(mean(Sens1.2.4),digits=3))
print(round(mean(Spec1.2.4),digits=3))
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form2.4			0.727				0.756					0.681
### Mean Pred by Chance(%): 	0.382 

#####################################################################################
################################# MULTINOMIAL MODEL #################################
#####################################################################################

################################
### 	2.2.2 MULTINOMIAL MODEL based on DATASET1 (27950 samples) 
logistic2.2.2 <- multinom(mod.form2.2, data= TRN1)
Acc2.2.2 <- Sens2.2.2 <- Spec2.2.2<- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
for (i in 1: length(TSTs1Dims)){
	ans <- as.factor(predict(logistic2.2.2,newdata = TST1s[[i]]))
	Acc2.2.2[i] <- round(sum(ans == TST1s[[i]]$outcome)/TSTs1Dims[i],digits=3)
	Sens2.2.2[i] <- round(sensitivity(ans,TST1s[[i]]$outcome),digits=3)	
	Spec2.2.2[i] <- round(specificity(ans,TST1s[[i]]$outcome),digits=3)	
}
print(round(mean(Acc2.2.2),digits=3)) 
print(round(mean(Sens2.2.2),digits=3))
print(round(mean(Spec2.2.2),digits=3))
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form2.2			0.735				0.824					0.591
### Mean Pred by Chance(%): 	0.382 
################################
### 	2.2.4 MULTINOMIAL MODEL based on DATASET1 (27950 samples) 
logistic2.2.4 <- multinom(mod.form2.4, data= TRN1)
Acc2.2.4 <- Sens2.2.4 <- Spec2.2.4<- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
for (i in 1: length(TSTs1Dims)){
	ans <- as.factor(predict(logistic2.2.4 ,newdata = TST1s[[i]]))
	Acc2.2.4[i] <- round(sum(ans == TST1s[[i]]$outcome)/TSTs1Dims[i],digits=3)
	Sens2.2.4[i] <- round(sensitivity(ans,TST1s[[i]]$outcome),digits=3)	
	Spec2.2.4[i] <- round(specificity(ans,TST1s[[i]]$outcome),digits=3)	
}
print(round(mean(Acc2.2.4),digits=3)) 
print(round(mean(Sens2.2.4),digits=3))
print(round(mean(Spec2.2.4),digits=3))
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form2.4			0.732				0.82					0.591
### Mean Pred by Chance(%): 	0.382 

#################################################################################
################################# DECISION TREE #################################
#################################################################################

################################
### 	 3.1.4 DECISION TREE MODEL on DATASET1 (27950 samples)
ctrl1.a <- tree.control(nobs=nrow(TRN1),mincut=50,minsize=100,mindev=0.00001)
tree3.1.4 <- tree(mod.form1.4,data=TRN1,split='deviance',control=ctrl1.a)
dev.new()
plot(tree3.1.4)
text(tree3.1.4 ,pretty=0)
summary(tree3.1.4 )
# MISCLASSIFICATION
misclass = prune.misclass(tree3.1.4)
print(misclass)
tree3.1.4.cut <- prune.misclass(tree3.1.4,best=93) 
# Not good method to manage this type of data
Acc3.1.4 <- Acc3.1.4.cut <- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
Spec3.1.4 <- Spec3.1.4.cut<- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
Sens3.1.4<- Sens3.1.4.cut <- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
for (i in 1: length(TSTs1Dims)){
	ans1 <- predict(tree3.1.4 ,newdata = TST1s[[i]], type="class")
	ans2 <- predict(tree3.1.4.cut ,newdata = TST1s[[i]], type="class")
	Acc3.1.4[i] <-round(sum(ans1 == TST1s[[i]]$ICU)/TSTs1Dims[i],digits=3)
	Acc3.1.4.cut[i] <-round(sum(ans2 == TST1s[[i]]$ICU)/TSTs1Dims[i],digits=3)
	Sens3.1.4[i] <- round(sensitivity(ans1,TST1s[[i]]$ICU),digits=3)
	Sens3.1.4.cut[i] <- round(sensitivity(ans2,TST1s[[i]]$ICU),digits=3)
	Spec3.1.4[i] <- round(specificity(ans1,TST1s[[i]]$ICU),digits=3)
	Spec3.1.4.cut[i] <- round(specificity(ans2,TST1s[[i]]$ICU),digits=3)	
}
print(round(mean(Acc3.1.4),digits=3))
print(round(mean(Sens3.1.4),digits=3))
print(round(mean(Spec3.1.4),digits=3))
### 					WITHOUT PRUNING
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity
### mod.form1.4		   0.599			  0.808					   0.313
### Mean Pred by Chance(%): 	0.577 
print(round(mean(Acc3.1.4.cut),digits=3))
print(round(mean(Sens3.1.4.cut),digits=3))
print(round(mean(Spec3.1.4.cut),digits=3))
### 						PRUNING
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity
### mod.form1.4			0.599				0.809					0.312
### Mean Pred by Chance(%): 	0.577 
################################
### 	 3.1.6 DECISION TREE MODEL on DATASET1 (27950 samples)
ctrl1.a <- tree.control(nobs=nrow(TRN1),mincut=50,minsize=100,mindev=0.00001)
tree3.1.6 <- tree(mod.form1.6,data=TRN1,split='deviance',control=ctrl1.a)
dev.new()
plot(tree3.1.6)
text(tree3.1.6 ,pretty=0)
summary(tree3.1.6)
# MISCLASSIFICATION
misclass = prune.misclass(tree3.1.6)
print(misclass)
tree3.1.6.cut <- prune.misclass(tree3.1.6,best=89) 
# Not good method to manage this type of data
Acc3.1.6 <- Acc3.1.6.cut <- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
Spec3.1.6 <- Spec3.1.6.cut<- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
Sens3.1.6 <- Sens3.1.6.cut <- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
for (i in 1: length(TSTs1Dims)){
	ans1 <- predict(tree3.1.6 ,newdata = TST1s[[i]], type="class")
	ans2 <- predict(tree3.1.6.cut ,newdata = TST1s[[i]], type="class")
	Acc3.1.6[i] <-round(sum(ans1 == TST1s[[i]]$ICU)/TSTs1Dims[i],digits=3)
	Acc3.1.6.cut[i] <-round(sum(ans2 == TST1s[[i]]$ICU)/TSTs1Dims[i],digits=3)
	Sens3.1.6[i] <- round(sensitivity(ans1,TST1s[[i]]$ICU),digits=3)
	Sens3.1.6.cut[i] <- round(sensitivity(ans2,TST1s[[i]]$ICU),digits=3)
	Spec3.1.6[i] <- round(specificity(ans1,TST1s[[i]]$ICU),digits=3)
	Spec3.1.6.cut[i] <- round(specificity(ans2,TST1s[[i]]$ICU),digits=3)	
}
print(round(mean(Acc3.1.6),digits=3))
print(round(mean(Sens3.1.6),digits=3))
print(round(mean(Spec3.1.6),digits=3))
### 					WITHOUT PRUNING
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity
### mod.form1.6			  0.597		    0.808				      0.31
### Mean Pred by Chance(%): 	0.577 
print(round(mean(Acc3.1.6.cut),digits=3))
print(round(mean(Sens3.1.6.cut),digits=3))
print(round(mean(Spec3.1.6.cut),digits=3))
### 						PRUNING
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity
### mod.form1.6			0.603				0.823					0.301
### Mean Pred by Chance(%): 	0.577 
################################
### 	 3.2.2 DECISION TREE MODEL on DATASET1 (27950 samples)
ctrl1.b <- tree.control(nobs=nrow(TRN1),mincut=50,minsize=100,mindev=0.00001)
tree3.2.2 <- tree(mod.form2.2,data=TRN1,split='deviance',control=ctrl1.b)
dev.new()
plot(tree3.2.2)
text(tree3.2.2,pretty=0)
summary(tree3.2.2)
# MISCLASSIFICATION
misclass = prune.misclass(tree3.2.2)
print(misclass)
tree3.2.2.cut <- prune.misclass(tree3.2.2,best=45) 
Acc3.2.2  <- Acc3.2.2.cut  <- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
Spec3.2.2 <- Spec3.2.2.cut <- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
Sens3.2.2 <- Sens3.2.2.cut <- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
for (i in 1: length(TSTs1Dims)){
	ans1 <- predict(tree3.2.2 ,newdata = TST1s[[i]], type="class")
	ans2 <- predict(tree3.2.2.cut ,newdata = TST1s[[i]], type="class")
	Acc3.2.2[i] <-round(sum(ans1 == TST1s[[i]]$outcome)/TSTs1Dims[i],digits=3)
	Acc3.2.2.cut[i] <-round(sum(ans2 == TST1s[[i]]$outcome)/TSTs1Dims[i],digits=3)
	Sens3.2.2[i] <- round(sensitivity(ans1,TST1s[[i]]$outcome),digits=3)
	Sens3.2.2.cut[i] <- round(sensitivity(ans2,TST1s[[i]]$outcome),digits=3)
	Spec3.2.2[i] <- round(specificity(ans1,TST1s[[i]]$outcome),digits=3)
	Spec3.2.2.cut[i] <- round(specificity(ans2,TST1s[[i]]$outcome),digits=3)	
}
print(round(mean(Acc3.2.2),digits=3))
print(round(mean(Sens3.2.2),digits=3))
print(round(mean(Spec3.2.2),digits=3))
### 					WITHOUT PRUNING(Method3)
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form2.2			0.725				0.806				 0.594
### Mean Pred by Chance(%): 	0.382 
print(round(mean(Acc3.2.2.cut),digits=3))
print(round(mean(Sens3.2.2.cut),digits=3))
print(round(mean(Spec3.2.2.cut),digits=3))
### 						PRUNING(Method3)
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form2.2			0.728				0.806				0.601
### Mean Pred by Chance(%): 	0.382 
################################
### 	 3.2.4 DECISION TREE MODEL on DATASET1 (27950 samples)
ctrl1.b <- tree.control(nobs=nrow(TRN1),mincut=50,minsize=100,mindev=0.00001)
tree3.2.4 <- tree(mod.form2.4,data=TRN1,split='deviance',control=ctrl1.b)
dev.new()
plot(tree3.2.4)
text(tree3.2.4,pretty=0)
summary(tree3.2.4)
# MISCLASSIFICATION
misclass = prune.misclass(tree3.2.4)
print(misclass)
tree3.2.4.cut <- prune.misclass(tree3.2.4,best=37) 
Acc3.2.4  <- Acc3.2.4.cut  <- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
Spec3.2.4 <- Spec3.2.4.cut <- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
Sens3.2.4 <- Sens3.2.4.cut <- matrix(rep(0,length(TSTs1Dims)),1,length(TSTs1Dims))
for (i in 1: length(TSTs1Dims)){
	ans1 <- predict(tree3.2.4 ,newdata = TST1s[[i]], type="class")
	ans2 <- predict(tree3.2.4.cut ,newdata = TST1s[[i]], type="class")
	Acc3.2.4[i] <-round(sum(ans1 == TST1s[[i]]$outcome)/TSTs1Dims[i],digits=3)
	Acc3.2.4.cut[i] <-round(sum(ans2 == TST1s[[i]]$outcome)/TSTs1Dims[i],digits=3)
	Sens3.2.4[i] <- round(sensitivity(ans1,TST1s[[i]]$outcome),digits=3)
	Sens3.2.4.cut[i] <- round(sensitivity(ans2,TST1s[[i]]$outcome),digits=3)
	Spec3.2.4[i] <- round(specificity(ans1,TST1s[[i]]$outcome),digits=3)
	Spec3.2.4.cut[i] <- round(specificity(ans2,TST1s[[i]]$outcome),digits=3)	
}
print(round(mean(Acc3.2.4),digits=3))
print(round(mean(Sens3.2.4),digits=3))
print(round(mean(Spec3.2.4),digits=3))
### 					WITHOUT PRUNING(Method3)
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form2.4			0.726				0.805					0.597
### Mean Pred by Chance(%): 	0.382 
print(round(mean(Acc3.2.4.cut),digits=3))
print(round(mean(Sens3.2.4.cut),digits=3))
print(round(mean(Spec3.2.4.cut),digits=3))
### 						PRUNING(Method3)
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form2.4			0.73				0.806					0.608
### Mean Pred by Chance(%): 	0.382 

#######################################################################
################################# SVM #################################
#######################################################################

################################ 
###		REDUCED DATASET FOR SVM APPLICATION (Only the 25% of of the Training/Testinng Set) (4891 samples)
set.seed(69)
TRN1red <- TRN1[sample(nrow(TRN1),round(0.25*nrow(TRN1))),]
TST1red <- TST1[sample(nrow(TST1),round(0.25*nrow(TST1))),]
TST1sred <- list('NULL')
TSTs1DimsRed <- round(TSTsProp*dim(TST1red)[1],digits=0)
set.seed(70)
for (i in 1:length(TSTs1DimsRed)){
	TST1sred[[i]] <- TST1red[sample(nrow(TST1red),round(TSTsProp[i]*nrow(TST1red))),] 
}
################################ 	
###		4.1.4 SVM MODEL on DATASET1 REDUCED
### 			LINEAR KERNEL
svm4.1.4.l <- svm(mod.form1.4, data = TRN1red, kernel="linear")
Acc4.1.4.p <- Acc4.1.4.r <- Acc4.1.4.l  <- matrix(rep(0,length(TSTs1DimsRed)),1,length(TSTs1DimsRed))
Spec4.1.4.p <- Spec4.1.4.r <- Spec4.1.4.l <- matrix(rep(0,length(TSTs1DimsRed)),1,length(TSTs1DimsRed))
Sens4.1.4.p <- Sens4.1.4.r <- Sens4.1.4.l <- matrix(rep(0,length(TSTs1DimsRed)),1,length(TSTs1DimsRed))
for (i in 1: length(TSTs1DimsRed)){
	ans1 <- predict(svm4.1.4.l,newdata = TST1sred[[i]], type="class")
	Acc4.1.4.l[i] <-round(sum(ans1 == TST1sred[[i]]$ICU)/TSTs1DimsRed[i],digits=3)
	Sens4.1.4.l[i] <- round(sensitivity(ans1, TST1sred[[i]]$ICU),digits=3)
	Spec4.1.4.l[i] <- round(specificity(ans1, TST1sred[[i]]$ICU),digits=3)	
}
print(round(mean(Acc4.1.4.l),digits=3))
print(round(mean(Sens4.1.4.l),digits=3))
print(round(mean(Spec4.1.4.l),digits=3))
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form1.4			0.579				0.889					0.163
### Mean Pred by Chance(%): 	0.577 
### 			POLYNOMIAL KERNEL
tune_out = tune(svm,mod.form1.4,data = TRN1red,kernel='polynomial',ranges = list(cost=c(1,10,100),degree = c(3,4,5)))
summary(tune_out)
### Model Formula:  Cost  	Degree    	Error		Accuracy
### mod.form1.4  	10.0      3       0.3934725		0.6065275
svm4.1.4.p <- tune_out$best.model
for (i in 1: length(TSTs1DimsRed)){
	ans1 <- predict(svm4.1.4.p,newdata = TST1sred[[i]], type="class")
	Acc4.1.4.p[i] <-round(sum(ans1 == TST1sred[[i]]$ICU)/TSTs1DimsRed[i],digits=3)
	Sens4.1.4.p[i] <- round(sensitivity(ans1, TST1sred[[i]]$ICU),digits=3)
	Spec4.1.4.p[i] <- round(specificity(ans1, TST1sred[[i]]$ICU),digits=3)	
}
print(round(mean(Acc4.1.4.p),digits=3))
print(round(mean(Sens4.1.4.p),digits=3))
print(round(mean(Spec4.1.4.p),digits=3))
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form1.4			0.609			    0.875				  0.25
### Mean Pred by Chance(%): 	0.577 
### 			RADIAL KERNEL
tune_out1 = tune(svm,mod.form1.4,data = TRN1red,kernel='radial',ranges = list(cost=c(1,0.01,0.1,5),gamma = c(0.1,0.5,1)))
summary(tune_out1)
### Model Formula:  Cost  	Gamma         Error	 	  Accuracy
### mod.form1.4  	  5       0.1		 0.3951161	  0.6048839
svm4.1.4.r <- tune_out1$best.model
for (i in 1: length(TSTs1DimsRed)){
	ans1 <- predict(svm4.1.4.r,newdata = TST1sred[[i]], type="class")
	Acc4.1.4.r[i] <-round(sum(ans1 == TST1sred[[i]]$ICU)/TSTs1DimsRed[i],digits=3)
	Sens4.1.4.r[i] <- round(sensitivity(ans1, TST1sred[[i]]$ICU),digits=3)
	Spec4.1.4.r[i] <- round(specificity(ans1, TST1sred[[i]]$ICU),digits=3)	
}
print(round(mean(Acc4.1.4.r),digits=3))
print(round(mean(Sens4.1.4.r),digits=3))
print(round(mean(Spec4.1.4.r),digits=3))
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form1.4		   0.602			  0.871				      	0.24
### Mean Pred by Chance(%): 	0.577 
################################ 	
###		4.1.6 SVM MODEL on DATASET1 REDUCED
### 			LINEAR KERNEL
svm4.1.6.l <- svm(mod.form1.6, data = TRN1red, kernel="linear")
Acc4.1.6.p <- Acc4.1.6.r <- Acc4.1.6.l  <- matrix(rep(0,length(TSTs1DimsRed)),1,length(TSTs1DimsRed))
Spec4.1.6.p <- Spec4.1.6.r <- Spec4.1.6.l <- matrix(rep(0,length(TSTs1DimsRed)),1,length(TSTs1DimsRed))
Sens4.1.6.p <- Sens4.1.6.r <- Sens4.1.6.l <- matrix(rep(0,length(TSTs1DimsRed)),1,length(TSTs1DimsRed))
for (i in 1: length(TSTs1DimsRed)){
	ans1 <- predict(svm4.1.6.l,newdata = TST1sred[[i]], type="class")
	Acc4.1.6.l[i] <-round(sum(ans1 == TST1sred[[i]]$ICU)/TSTs1DimsRed[i],digits=3)
	Sens4.1.6.l[i] <- round(sensitivity(ans1, TST1sred[[i]]$ICU),digits=3)
	Spec4.1.6.l[i] <- round(specificity(ans1, TST1sred[[i]]$ICU),digits=3)	
}
print(round(mean(Acc4.1.6.l),digits=3))
print(round(mean(Sens4.1.6.l),digits=3))
print(round(mean(Spec4.1.6.l),digits=3))
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity to 
### mod.form1.6			0.579				0.889					0.163
### Mean Pred by Chance(%): 	0.577 
### 			POLYNOMIAL KERNEL
tune_out = tune(svm,mod.form1.6,data = TRN1red,kernel='polynomial',ranges = list(cost=c(1,10,100),degree = c(3,4,5)))
summary(tune_out)
### Model Formula:  Cost  	Degree    	Error		Accuracy
### mod.form1.6  	10        5       0.3971669    0.6028331
svm4.1.6.p <- tune_out$best.model
for (i in 1: length(TSTs1DimsRed)){
	ans1 <- predict(svm4.1.6.p,newdata = TST1sred[[i]], type="class")
	Acc4.1.6.p[i] <-round(sum(ans1 == TST1sred[[i]]$ICU)/TSTs1DimsRed[i],digits=3)
	Sens4.1.6.p[i] <- round(sensitivity(ans1, TST1sred[[i]]$ICU),digits=3)
	Spec4.1.6.p[i] <- round(specificity(ans1, TST1sred[[i]]$ICU),digits=3)	
}
print(round(mean(Acc4.1.6.p),digits=3))
print(round(mean(Sens4.1.6.p),digits=3))
print(round(mean(Spec4.1.6.p),digits=3))
### Model Formula:	Mean Accuracy	 Mean Sensitivity 	Mean Specificity 
### mod.form1.6		     	0.605	    0.916			    0.186 
### Mean Pred by Chance(%): 	0.577 
### 			RADIAL KERNEL
### Model Formula:  Cost  	Gamma         Error	 	  Accuracy
### mod.form1.6  	 0.1      0.5      0.4010696	 0.5989304
svm4.1.6.r <- tune_out1$best.model
for (i in 1: length(TSTs1DimsRed)){
	ans1 <- predict(svm4.1.6.r,newdata = TST1sred[[i]], type="class")
	Acc4.1.6.r[i] <-round(sum(ans1 == TST1sred[[i]]$ICU)/TSTs1DimsRed[i],digits=3)
	Sens4.1.6.r[i] <- round(sensitivity(ans1, TST1sred[[i]]$ICU),digits=3)
	Spec4.1.6.r[i] <- round(specificity(ans1, TST1sred[[i]]$ICU),digits=3)	
}
print(round(mean(Acc4.1.6.r),digits=3))
print(round(mean(Sens4.1.6.r),digits=3))
print(round(mean(Spec4.1.6.r),digits=3))
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form1.6			  0.605			   0.746				    0.396
### Mean Pred by Chance(%): 	0.577 
################################
###		4.2.2 SVM MODEL on DATASET1 REDUCED 
### 			LINEAR KERNEL
svm4.2.2.l <- svm(mod.form2.2, data = TRN1red, kernel="linear")
Acc4.2.2.p <- Acc4.2.2.r <- Acc4.2.2.l  <- matrix(rep(0,length(TSTs1DimsRed)),1,length(TSTs1DimsRed))
Spec4.2.2.p <- Spec4.2.2.r  <- Spec4.2.2.l <- matrix(rep(0,length(TSTs1DimsRed)),1,length(TSTs1DimsRed))
Sens4.2.2.p <- Sens4.2.2.r  <- Sens4.2.2.l <- matrix(rep(0,length(TSTs1DimsRed)),1,length(TSTs1DimsRed))
for (i in 1: length(TSTs1DimsRed)){
	ans1 <- predict(svm4.2.2.l,newdata = TST1sred[[i]], type="class")
	Acc4.2.2.l[i] <-round(sum(ans1 == TST1sred[[i]]$outcome)/TSTs1DimsRed[i],digits=3)
	Sens4.2.2.l[i] <- round(sensitivity(ans1, TST1sred[[i]]$outcome),digits=3)
	Spec4.2.2.l[i] <- round(specificity(ans1, TST1sred[[i]]$outcome),digits=3)	
}
print(round(mean(Acc4.2.2.l),digits=3))
print(round(mean(Sens4.2.2.l),digits=3))
print(round(mean(Spec4.2.2.l),digits=3))
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form2.2			0.696				0.718					0.661
### Mean Pred by Chance(%): 	0.382 
### 			POLYNOMIAL KERNEL
tune_out = tune(svm,mod.form2.2,data = TRN1red,kernel='polynomial',ranges = list(cost=c(0.1,1,10),degree = c(2,3,4)))
summary(tune_out)
### Model Formula:  Cost  	Degree    	Error		Accuracy
### mod.form2.2  	10        2	  	  0.269492		0.730508
svm4.2.2.p <- tune_out$best.model
for (i in 1: length(TSTs1DimsRed)){
	ans1 <- predict(svm4.2.2.p,newdata = TST1sred[[i]], type="class")
	Acc4.2.2.p[i] <-round(sum(ans1 == TST1sred[[i]]$outcome)/TSTs1DimsRed[i],digits=3)
	Sens4.2.2.p[i] <- round(sensitivity(ans1, TST1sred[[i]]$outcome),digits=3)
	Spec4.2.2.p[i] <- round(specificity(ans1, TST1sred[[i]]$outcome),digits=3)	
}
print(round(mean(Acc4.2.2.p),digits=3))
print(round(mean(Sens4.2.2.p),digits=3))
print(round(mean(Spec4.2.2.p),digits=3))
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form2.2			0.725				0.81					0.587
### Mean Pred by Chance(%): 	0.382 
### 			RADIAL KERNEL
tune_out1 = tune(svm,mod.form2.2,data = TRN1red,kernel='radial',ranges = list(cost=c(0.1,1,10),gamma = c(0.5,1,2)))
summary(tune_out1)
### Model Formula:  Cost  	Gamma    	Error		Accuracy
### mod.form2.2  	 1       0.5	   0.2742148   0.7257852
svm4.2.2.r  <- tune_out1$best.model
for (i in 1: length(TSTs1DimsRed)){
	ans1 <- predict(svm4.2.2.r,newdata = TST1sred[[i]], type="class")
	Acc4.2.2.r[i] <-round(sum(ans1 == TST1sred[[i]]$outcome)/TSTs1DimsRed[i],digits=3)
	Sens4.2.2.r[i] <- round(sensitivity(ans1, TST1sred[[i]]$outcome),digits=3)
	Spec4.2.2.r[i] <- round(specificity(ans1, TST1sred[[i]]$outcome),digits=3)	
}
print(round(mean(Acc4.2.2.r ),digits=3))
print(round(mean(Sens4.2.2.r ),digits=3))
print(round(mean(Spec4.2.2.r ),digits=3))
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form2.2			0.732				0.825					0.584
### Mean Pred by Chance(%): 	0.382 
################################
###		4.2.4 SVM MODEL on DATASET1 REDUCED 
### 			LINEAR KERNEL
svm4.2.4.l <- svm(mod.form2.4, data = TRN1red, kernel="linear")
Acc4.2.4.p <- Acc4.2.4.r <- Acc4.2.4.l  <- matrix(rep(0,length(TSTs1DimsRed)),1,length(TSTs1DimsRed))
Spec4.2.4.p <- Spec4.2.4.r  <- Spec4.2.4.l <- matrix(rep(0,length(TSTs1DimsRed)),1,length(TSTs1DimsRed))
Sens4.2.4.p <- Sens4.2.4.r  <- Sens4.2.4.l <- matrix(rep(0,length(TSTs1DimsRed)),1,length(TSTs1DimsRed))
for (i in 1: length(TSTs1DimsRed)){
	ans1 <- predict(svm4.2.4.l,newdata = TST1sred[[i]], type="class")
	Acc4.2.4.l[i] <-round(sum(ans1 == TST1sred[[i]]$outcome)/TSTs1DimsRed[i],digits=3)
	Sens4.2.4.l[i] <- round(sensitivity(ans1, TST1sred[[i]]$outcome),digits=3)
	Spec4.2.4.l[i] <- round(specificity(ans1, TST1sred[[i]]$outcome),digits=3)	
}
print(round(mean(Acc4.2.4.l),digits=3))
print(round(mean(Sens4.2.4.l),digits=3))
print(round(mean(Spec4.2.4.l),digits=3))
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form2.4			0.696				0.718					0.661
### Mean Pred by Chance(%): 	0.382 
### 			POLYNOMIAL KERNEL
tune_out = tune(svm,mod.form2.4,data = TRN1red,kernel='polynomial',ranges = list(cost=c(0.1,1,10),degree = c(2,3,4)))
summary(tune_out)
### Model Formula:  Cost  	Degree    	Error		Accuracy
### mod.form2.4  	10        2	      0.2725818		0.7274182
svm4.2.4.p <- tune_out$best.model
for (i in 1: length(TSTs1DimsRed)){
	ans1 <- predict(svm4.2.4.p,newdata = TST1sred[[i]], type="class")
	Acc4.2.4.p[i] <-round(sum(ans1 == TST1sred[[i]]$outcome)/TSTs1DimsRed[i],digits=3)
	Sens4.2.4.p[i] <- round(sensitivity(ans1, TST1sred[[i]]$outcome),digits=3)
	Spec4.2.4.p[i] <- round(specificity(ans1, TST1sred[[i]]$outcome),digits=3)	
}
print(round(mean(Acc4.2.4.p),digits=3))
print(round(mean(Sens4.2.4.p),digits=3))
print(round(mean(Spec4.2.4.p),digits=3))
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form2.4			0.723				0.807					0.587
### Mean Pred by Chance(%): 	0.382 
### 			RADIAL KERNEL
tune_out1 = tune(svm,mod.form2.4,data = TRN1red,kernel='radial',ranges = list(cost=c(0.1,1,10),gamma = c(0.5,1,2)))
summary(tune_out1)
### Model Formula:  Cost  	Gamma    	Error		Accuracy
### mod.form2.4  	 1       0.5	   0.2660126   0.7339874
svm4.2.4.r  <- tune_out1$best.model
for (i in 1: length(TSTs1DimsRed)){
	ans1 <- predict(svm4.2.4.r,newdata = TST1sred[[i]], type="class")
	Acc4.2.4.r[i] <-round(sum(ans1 == TST1sred[[i]]$outcome)/TSTs1DimsRed[i],digits=3)
	Sens4.2.4.r[i] <- round(sensitivity(ans1, TST1sred[[i]]$outcome),digits=3)
	Spec4.2.4.r[i] <- round(specificity(ans1, TST1sred[[i]]$outcome),digits=3)	
}
print(round(mean(Acc4.2.4.r ),digits=3))
print(round(mean(Sens4.2.4.r ),digits=3))
print(round(mean(Spec4.2.4.r ),digits=3))
### Model Formula:	Mean Accuracy	 Mean Sensitivity 		Mean Specificity 
### mod.form2.4			0.728				0.802					0.607
### Mean Pred by Chance(%): 	0.382 

################################################################################
################################### BOOSTING ###################################
################################################################################

################################
###		5.1.4 BOOSTING MODEL on DATASET1 

gbm5.1.4 <- gbm( mod.form1.4 ,
               distribution = "multinomial" ,
               verbose = T ,
               data = TRN1 ,
               n.trees = 100 ,
               train.fraction = 0.8 ,
               cv.folds = 5 ,
               bag.fraction = 0.5 ,
               shrinkage = 0.05 )

best_iter5.1.4 <- gbm.perf( gbm1.4 , method = "cv" )

perf_gbm5.1.4 <- matrix(rep(0,6*3),6,3)
colnames(perf_gbm5.1.4) <- c("sensitivity","specificity","accuracy")
rownames(perf_gbm5.1.4) <- c("TST1s1","TST1s2","TST1s3","TST1s4","TST1s5","TST1s6")


for (j in 1: length(TSTs1Dims)){
  
  pred5.1.4 <- predict( gbm5.1.4 , newdata = TST1s[[j]] , n.trees = best_iter5.1.4 , type = "response" )
  pred5.1.4 <- as.matrix(as.data.frame(pred5.1.4))
  prediction5.1.4 <- rep(0,nrow(pred5.1.4))
  for ( i in 1:nrow( pred5.1.4 ) ) {
    prediction5.1.4[ i ] = which( pred5.1.4[ i , ] == max( pred5.1.4[ i , ] ) )-1
  }
  
  x <- table( prediction5.1.4 , TST1s[[j]]$ICU )
  
  perf_gbm5.1.4[j,1] <- sensitivity(as.factor(prediction5.1.4),TST1s[[j]]$ICU)
  perf_gbm5.1.4[j,2] <- specificity(as.factor(prediction5.1.4),TST1s[[j]]$ICU)
  perf_gbm5.1.4[j,3] <- (x[1,1]+x[2,2])/sum(x)
  
}

perf_gbm5.1.4 <- round(100*perf_gbm5.1.4,digit=2)
#        sensitivity specificity accuracy
# TST1s1       88.23       22.95    61.05
# TST1s2       88.17       22.67    60.81
# TST1s3       88.28       22.35    60.97
# TST1s4       88.72       23.73    61.84
# TST1s5       88.42       20.77    59.94
# TST1s6       87.75       20.36    61.20
mean( perf_gbm5.1.4[,1] )
mean( perf_gbm5.1.4[,2] ) 
mean( perf_gbm5.1.4[,3] )



################################
###		5.1.6 BOOSTING MODEL on DATASET1 

gbm5.1.6 <- gbm( mod.form1.6 ,
               distribution = "multinomial" ,
               verbose = T ,
               data = TRN1 ,
               n.trees = 100 ,
               train.fraction = 0.8 ,
               cv.folds = 5 ,
               bag.fraction = 0.5 ,
               shrinkage = 0.05 )

best_iter5.1.6 <- gbm.perf( gbm5.1.6 , method = "cv" )

perf_gbm5.1.6 <- matrix(rep(0,6*3),6,3)
colnames(perf_gbm5.1.6) <- c("sensitivity","specificity","accuracy")
rownames(perf_gbm5.1.6) <- c("TST1s1","TST1s2","TST1s3","TST1s4","TST1s5","TST1s6")


for (j in 1: length(TSTs1Dims)){
  
  pred5.1.6 <- predict( gbm5.1.6 , newdata = TST1s[[j]] , n.trees = best_iter5.1.6 , type = "response" )
  pred5.1.6 <- as.matrix(as.data.frame(pred5.1.6))
  prediction5.1.6 <- rep(0,nrow(pred5.1.6))
  for ( i in 1:nrow( pred1.6 ) ) {
    prediction5.1.6[ i ] = which( pred5.1.6[ i , ] == max( pred5.1.6[ i , ] ) )-1
  }
  
  x <- table( prediction5.1.6 , TST1s[[j]]$ICU )
  
  perf_gbm5.1.6[j,1] <- sensitivity(as.factor(prediction5.1.6),TST1s[[j]]$ICU)
  perf_gbm5.1.6[j,2] <- specificity(as.factor(prediction5.1.6),TST1s[[j]]$ICU)
  perf_gbm5.1.6[j,3] <- (x[1,1]+x[2,2])/sum(x)
  
}

perf_gbm5.1.6 <- round(100*perf_gbm5.1.6,digit=2)
#              sensitivity specificity accuracy
# TST1s1       88.82       20.48    60.37
# TST1s2       88.66       20.70    60.27
# TST1s3       88.72       19.70    60.13
# TST1s4       89.28       21.35    61.18
# TST1s5       89.25       19.35    59.82
# TST1s6       88.14       17.63    60.36
mean( perf_gbm5.1.6[,1] )
mean( perf_gbm5.1.6[,2] ) 
mean( perf_gbm5.1.6[,3] )



################################
###		5.2.2 BOOSTING MODEL on DATASET1 

gbm5.2.2 <- gbm( mod.form2.2 ,
               distribution = "multinomial" ,
               verbose = T ,
               data = TRN1 ,
               n.trees = 100 ,
               train.fraction = 0.8 ,
               cv.folds = 5 ,
               bag.fraction = 0.5 ,
               shrinkage = 0.05 )

best_iter5.2.2 <- gbm.perf( gbm5.2.2 , method = "cv" )

perf_gbm5.2.2 <- matrix(rep(0,6*3),6,3)
colnames(perf_gbm5.2.2) <- c("sensitivity","specificity","accuracy")
rownames(perf_gbm5.2.2) <- c("TST1s1","TST1s2","TST1s3","TST1s4","TST1s5","TST1s6")


for (j in 1: length(TSTs1Dims)){
  
  pred5.2.2 <- predict( gbm5.2.2 , newdata = TST1s[[j]] , n.trees = best_iter5.2.2 , type = "response" )
  pred5.2.2 <- as.matrix(as.data.frame(pred2.2))
  prediction5.2.2 <- rep(0,nrow(pred5.2.2))
  for ( i in 1:nrow( pred5.2.2 ) ) {
    prediction2.2[ i ] = which( pred5.2.2[ i , ] == max( pred5.2.2[ i , ] ) )-1
  }
  
  x <- table( prediction5.2.2 , TST1s[[j]]$ICU )
  
  perf_gbm5.2.2[j,1] <- sensitivity(as.factor(prediction5.2.2),TST1s[[j]]$ICU)
  perf_gbm5.2.2[j,2] <- specificity(as.factor(prediction5.2.2),TST1s[[j]]$ICU)
  perf_gbm5.2.2[j,3] <- (x[1,1]+x[2,2])/sum(x)
  
}

perf_gbm5.2.2 <- round(100*perf_gbm5.2.2,digit=2)
# sensitivity specificity accuracy
# TST1s1       98.95       72.53    87.95
# TST1s2       98.97       71.88    87.65
# TST1s3       98.91       71.63    87.61
# TST1s4       98.98       72.79    88.15
# TST1s5       99.28       72.69    88.08
# TST1s6       99.21       69.00    87.31
mean( perf_gbm5.2.2[,1] )
mean( perf_gbm5.2.2[,2] ) 
mean( perf_gbm5.2.2[,3] )



################################
###		5.2.4 BOOSTING MODEL on DATASET1 

gbm5.2.4 <- gbm( mod.form5.2.4 ,
               distribution = "multinomial" ,
               verbose = T ,
               data = TRN1 ,
               n.trees = 100 ,
               train.fraction = 0.8 ,
               cv.folds = 5 ,
               bag.fraction = 0.5 ,
               shrinkage = 0.05 )

best_iter5.2.4 <- gbm.perf( gbm5.2.4 , method = "cv" )

perf_gbm5.2.4 <- matrix(rep(0,6*3),6,3)
colnames(perf_gbm5.2.4) <- c("sensitivity","specificity","accuracy")
rownames(perf_gbm5.2.4) <- c("TST1s1","TST1s2","TST1s3","TST1s4","TST1s5","TST1s6")

for (j in 1: length(TSTs1Dims)){
  
  pred5.2.4 <- predict( gbm5.2.4 , newdata = TST1s[[j]] , n.trees = best_iter5.2.4 , type = "response" )
  pred5.2.4 <- as.matrix(as.data.frame(pred5.2.4))
  prediction5.2.4 <- rep(0,nrow(pred5.2.4))
  for ( i in 1:nrow( pred2.4 ) ) {
    prediction5.2.4[ i ] = which( pred5.2.4[ i , ] == max( pred5.2.4[ i , ] ) )-1
  }
  
  x <- table( prediction5.2.4 , TST1s[[j]]$ICU )
  
  perf_gbm5.2.4[j,1] <- sensitivity(as.factor(prediction5.2.4),TST1s[[j]]$ICU)
  perf_gbm5.2.4[j,2] <- specificity(as.factor(prediction5.2.4),TST1s[[j]]$ICU)
  perf_gbm5.2.4[j,3] <- (x[1,1]+x[2,2])/sum(x)
  
}

perf_gbm5.2.4 <- round(100*perf_gbm5.2.4,digit=2)
# sensitivity specificity accuracy
# TST1s1       99.22       71.44    87.66
# TST1s2       99.23       70.74    87.32
# TST1s3       99.22       70.81    87.45
# TST1s4       99.08       71.78    87.79
# TST1s5       99.59       71.41    87.72
# TST1s6       99.60       67.17    86.83
mean( perf_gbm5.2.4[,1] )
mean( perf_gbm5.2.4[,2] ) 
mean( perf_gbm5.2.4[,3] )



#################################################################################
################################# RANDOM FOREST #################################
#################################################################################

################################
###		6.1.4 RANDOM FOREST MODEL on DATASET1 
rf6.1.4 <- randomForest( mod.form1.4 , data = TRN1 , importance = T )
plot(rf6.1.4)
# Number of trees: 500
# No. of variables tried at each split: 3
# OOB estimate of  error rate: 38.96%
# Confusion matrix:
#   0    1 class.error
# 0 9942 1632   0.1410057
# 1 5960 1954   0.7530958
tuningrf6.1.4 <- tuneRF( TRN1[,c(3,5,10:14,17,19,26) ],TRN1[ , 28 ],mtryStart = 3,ntreeTry = 200,stepFactor = 0.9,improve = 0.05,trace = T,plot = T )
rf6.1.4.tune <- randomForest( mod.form1.4 , data = TRN1 , importance = T ,ntree = 200 , mtry = 3 )
# Number of trees: 200
# No. of variables tried at each split: 3
# OOB estimate of  error rate: 39.01%
# Confusion matrix:
#   0    1 class.error
# 0 9956 1618   0.1397961
# 1 5985 1929   0.7562547

perf_rf6.1.4 <- matrix(rep(0,6*3),6,3)
colnames(perf_rf6.1.4) <- c("sensitivity","specificity","accuracy")
rownames(perf_rf6.1.4) <- c("TST1s1","TST1s2","TST1s3","TST1s4","TST1s5","TST1s6")

for (j in 1: length(TSTs1Dims)){
  
  pred.rf6.1.4 <- predict(rf6.1.4.tune , newdata = TST1s[[j]] , type = "response" ,
                        norm.votes = T , predict.all = F , proximity = F , nodes = F )
  
  x <- table(pred.rf6.1.4,TST1s[[j]]$ICU)
  perf_rf6.1.4[j,1] <- sensitivity(pred.rf6.1.4,TST1s[[j]]$ICU)
  perf_rf6.1.4[j,2] <- specificity(pred.rf6.1.4,TST1s[[j]]$ICU)
  perf_rf6.1.4[j,3] <- (x[1,1]+x[2,2])/sum(x)
  
}

perf_rf6.1.4 <- round(100*perf_rf6.1.4,digit=2)
#        sensitivity specificity accuracy
# TST1s1       86.34       24.19    60.46
# TST1s2       86.35       24.28    60.42
# TST1s3       85.42       22.78    59.47
# TST1s4       86.37       25.83    61.33
# TST1s5       86.04       23.61    59.76
# TST1s6       85.38       19.45    59.40
mean( perf_rf6.1.4[,1] )
mean( perf_rf6.1.4[,2] ) 
mean( perf_rf6.1.4[,3] )


################################
###		6.1.6 RANDOM FOREST MODEL on DATASET1 

rf6.1.6 <- randomForest( mod.form1.6 , data = TRN1 , importance = T )
plot( rf6.1.6 )
# Number of trees: 500
# No. of variables tried at each split: 3
# OOB estimate of  error rate: 39.1%
# Confusion matrix:
#   0    1 class.error
# 0 10087 1487   0.1284776
# 1  6133 1781   0.7749558

tuningrf6.1.6 <- tuneRF( TRN1[ , c(3,5,10:14,17,23,26) ] , TRN1[ , 28 ] ,
                       mtryStart = 3 , ntreeTry = 100 ,
                       stepFactor = 0.9 , improve = 0.05 ,
                       trace = T , plot = T )

rf6.1.6.tune <- randomForest( mod.form1.6 , data = TRN1 , importance = T ,
                            ntree = 100 , mtry = 3 )
# Number of trees: 100
# No. of variables tried at each split: 3
# OOB estimate of  error rate: 39.29%
# Confusion matrix:
#   0    1 class.error
# 0 10094 1480   0.1278728
# 1  6176 1738   0.7803892

perf_rf6.1.6 <- matrix(rep(0,6*3),6,3)
colnames(perf_rf6.1.6) <- c("sensitivity","specificity","accuracy")
rownames(perf_rf6.1.6) <- c("TST1s1","TST1s2","TST1s3","TST1s4","TST1s5","TST1s6")

for (j in 1: length(TSTs1Dims)){
  
  pred.rf6.1.6 <- predict(rf6.1.6.tune , newdata = TST1s[[j]] , type = "response" ,
                        norm.votes = T , predict.all = F , proximity = F , nodes = F )
  
  x <- table(pred.rf6.1.6,TST1s[[j]]$ICU)
  perf_rf6.1.6[j,1] <- sensitivity(pred.rf6.1.6,TST1s[[j]]$ICU)
  perf_rf6.1.6[j,2] <- specificity(pred.rf6.1.6,TST1s[[j]]$ICU)
  perf_rf6.1.6[j,3] <- (x[1,1]+x[2,2])/sum(x)
  
}

perf_rf6.1.6 <- round(100*perf_rf6. and 1.6,digit=2)
#              sensitivity specificity accuracy
# TST1s1       87.49       23.07    60.67
# TST1s2       87.53       23.39    60.73
# TST1s3       87.29       22.16    60.31
# TST1s4       87.70       24.67    61.63
# TST1s5       87.59       21.05    59.58
# TST1s6       86.96       19.76    60.48
mean( perf_rf6.1.6[,1] )
mean( perf_rf6.1.6[,2] ) 
mean( perf_rf6.1.6[,3] )


################################
###		6.2.2 RANDOM FOREST MODEL on DATASET1 
rf6.2.2 <- randomForest( mod.form2.2 , data = TRN1 , importance = T )
plot( rf6.2.2 )
# Number of trees: 500
# No. of variables tried at each split: 4
# OOB estimate of  error rate: 27.19%
# Confusion matrix:
#   0    1 class.error
# 0 10255 1935   0.1587367
# 1  3363 3935   0.4608112
tuningrf6.2.2 <- tuneRF( TRN1[ , -c(1,2,4,7,8,27,29) ] , TRN1[ , 29 ] ,
                       mtryStart = 4 , ntreeTry = 100 ,
                       stepFactor = 0.5 , improve = 0.05 ,
                       trace = T , plot = T )
rf6.2.2.tune <- randomForest( mod.form2.2 , data = TRN1 , importance = T ,
                            ntree = 100 , mtry = 4 )
# Number of trees: 100
# No. of variables tried at each split: 4
# OOB estimate of  error rate: 27.31%
# Confusion matrix:
#   0    1 class.error
# 0 10221 1969   0.1615258
# 1  3353 3945   0.4594409

perf_rf6.2.2 <- matrix(rep(0,6*3),6,3)
colnames(perf_rf6.2.2) <- c("sensitivity","specificity","accuracy")
rownames(perf_rf6.2.2) <- c("TST1s1","TST1s2","TST1s3","TST1s4","TST1s5","TST1s6")

for (j in 1: length(TSTs1Dims)){
  
  pred.rf6.2.2 <- predict(rf6.2.2.tune , newdata = TST1s[[j]] , type = "response" ,
                        norm.votes = T , predict.all = F , proximity = F , nodes = F )
  
  x <- table(pred.rf6.2.2,TST1s[[j]]$ICU)
  perf_rf6.2.2[j,1] <- sensitivity(pred.rf6.2.2,TST1s[[j]]$ICU)
  perf_rf6.2.2[j,2] <- specificity(pred.rf6.2.2,TST1s[[j]]$ICU)
  perf_rf6.2.2[j,3] <- (x[1,1]+x[2,2])/sum(x)
  
}

perf_rf6.2.2 <- round(100*perf_rf6.2.2,digit=2)
# sensitivity specificity accuracy
# TST1s1       95.18       69.83    84.63
# TST1s2       95.37       69.59    84.60
# TST1s3       95.26       68.50    84.17
# TST1s4       95.15       70.41    84.91
# TST1s5       95.97       70.55    85.27
# TST1s6       95.65       67.78    84.67
mean( perf_rf6.2.2[,1] )
mean( perf_rf6.2.2[,2] ) 
mean( perf_rf6.2.2[,3] )



################################
###		6.2.4 RANDOM FOREST MODEL on DATASET1 
rf6.2.4 <- randomForest( mod.form2.4 , data = TRN1 , importance = T )
plot( rf6.2.4 )
# Number of trees: 500
# No. of variables tried at each split: 3
# OOB estimate of  error rate: 26.64%
# Confusion matrix:
#   0    1 class.error
# 0 10296 1894   0.1553733
# 1  3298 4000   0.4519046

tuningrf6.2.4 <- tuneRF( TRN1[ , c(5,7,10:14,17,18,20:24,28) ] , TRN1[ , 29 ] ,
                       mtryStart = 3 , ntreeTry = 100 ,
                       stepFactor = 0.5 , improve = 0.05 ,
                       trace = T , plot = T )

rf6.2.4.tune <- randomForest( mod.form2.4 , data = TRN1 , importance = T ,
                            ntree = 100 , mtry = 3 )
# Number of trees: 100
# No. of variables tried at each split: 3
# OOB estimate of  error rate: 26.83%
# Confusion matrix:
#   0    1 class.error
# 0 10271 1919   0.1574241
# 1  3310 3988   0.4535489

perf_rf6.2.4 <- matrix(rep(0,6*3),6,3)
colnames(perf_rf6.2.4) <- c("sensitivity","specificity","accuracy")
rownames(perf_rf6.2.4) <- c("TST1s1","TST1s2","TST1s3","TST1s4","TST1s5","TST1s6")

for (j in 1: length(TSTs1Dims)){
  
  pred.rf6.2.4 <- predict(rf6.2.4.tune , newdata = TST1s[[j]] , type = "response" ,
                        norm.votes = T , predict.all = F , proximity = F , nodes = F )
  
  x <- table(pred.rf6.2.4,TST1s[[j]]$ICU)
  perf_rf6.2.4[j,1] <- sensitivity(pred.rf6.2.4,TST1s[[j]]$ICU)
  perf_rf6.2.4[j,2] <- specificity(pred.rf6.2.4,TST1s[[j]]$ICU)
  perf_rf6.2.4[j,3] <- (x[1,1]+x[2,2])/sum(x)
  
}

perf_rf6.2.4 <- round(100*perf_rf6.2.4,digit=2)
#              sensitivity specificity accuracy
# TST1s1       97.48       71.41    86.63
# TST1s2       97.63       70.59    86.34
# TST1s3       97.31       70.86    86.35
# TST1s4       97.40       72.36    87.04
# TST1s5       98.04       71.12    86.71
# TST1s6       97.63       68.69    86.23
mean( perf_rf6.2.4[,1] )
mean( perf_rf6.2.4[,2] ) 
mean( perf_rf6.2.4[,3] )


