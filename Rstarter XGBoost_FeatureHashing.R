## Script directly from Kaggle Kernel

library(data.table)
library(FeatureHashing)
library(Matrix)
library(xgboost)

############################ data preparation ########################
setwd("C:/Users/wenbma/Desktop/Others/Kaggle/RedHat/people.csv")
people = fread('people.csv',showProgress = F)

p_logi = names(people)[which(sapply(people,is.logical))]
for (col in p_logi) set(people,j=col,value=as.integer(people[[col]]))

setwd("C:/Users/wenbma/Desktop/Others/Kaggle/RedHat/act_train.csv")
train = fread("act_train.csv",showProgress = F)
d1 = merge(train,people,by="people_id",all.x = T)

Y = d1$outcome
d1[,outcome:=NULL]

########################### process categorical features via feature hashing ######################
b = 2^22
f = ~.-people_id-activity_id-date.x-date.y-1

X_train = hashed.model.matrix(f,d1,hash.size=b)

sum(colSums(X_train)>0)

########################### validate xgboost model ##################################
set.seed(75786)
unique_p = unique(d1$people_id)
valid_p = unique_p[sample(1:length(unique_p),30000)]
valid = which(d1$people_id %in% valid_p)
model = (1:length(d1$people_id))[-valid]
param = list(objective ="binary:logistic",eval_metric="auc",booster="gblinear",eta=0.03)

dmodel = xgb.DMatrix(X_train[model,],label=Y[model])
dvalid = xgb.DMatrix(X_train[valid,],label=Y[valid])

m1 = xgb.train(data=dmodel,param,nrounds = 100,watchlist=list(dmodel,valid=dvalid),print_every_n=10)

########################## retain on all data and predict for test set
dtrain = xgb.DMatrix(X_train,label=Y)
m2 = xgb.train(data=dtrain,param,nrounds=100,watchlist=list(train=dtrain),print_every_n=10)

setwd("C:/Users/wenbma/Desktop/Others/Kaggle/RedHat/act_test.csv")
test = fread("act_test.csv",showProgress = F)
d2 = merge(test,people,by='people_id',all.x=F)

X_test = hashed.model.matrix(f,d2,hash.size = b)
dtest = xgb.DMatrix(X_test)
out = predict(m2,dtest)
sub = data.frame(activity_id=d2$activity_id,outcome=out)
write.csv(sub,file="sub.csv",row.names = F)
summary(sub$outcome)
