library(data.table)

############################ read data ###################################
setwd("C:/Users/wenbma/Desktop/Others/Kaggle/RedHat/people.csv")
people = fread("people.csv",showProgress = F)
setwd("C:/Users/wenbma/Desktop/Others/Kaggle/RedHat/act_train.csv")
train = fread("act_train.csv",showProgress = F)

########################### merge(left join) dataset by 'people_id'#######
train_people = merge(train,people,all.x = T,by="people_id")

########################### convert boolean variable to 0-1 ##############
logical_var = names(train_people)[which(sapply(train_people,is.logical))]

for (col in logical_var) set(train_people,j=col,value=as.integer(train_people[[col]]))

########################### explore categorical variable #################
char_var = names(train_people)[which(sapply(train_people,is.character))]

##barplot(table(train_people$people_id))

i=1
for (col in names(train_people)) 
{
  print (i)
  print(length(unique(train_people[[col]])))
  i = i+1
}

########################### variable 1,2,14,16,18 have too many levels. I will exclude them in the first model#########

########################### In addition, variable 3 "date.x" is a date variable which may not have predictive power####
########################### Let me exclude it in the first model as well ##############################################

## drops variable 1,2,3,14,17,19

drops = c(1,2,3,17,19) # not remove 19
keeps = names(train_people)[-drops]

train_people_first = subset(train_people,select=keeps)

p=0
for (col in names(train_people_first)) 
{
  p=p+length(unique(train_people_first[[col]]))
  
}




Y = train_people_first$outcome
X_train_people_first = train_people_first[,outcome:=NULL]
########################## Set Validation set and implement xgboost ###################################################
set.seed(25000)
valid_ind = sample(1:length(Y),200000)
Y_train = Y[-valid_ind]
X_train = X_train_people_first[-valid_ind,]
Y_valid = Y[valid_ind]
X_valid = X_train_people_first[valid_ind,]

library(CatEncoders)
oenc=OneHotEncoder.fit(train_people_first)
X_train_matrix = transform(oenc,X_train,sparse=TRUE)
X_valid_matrix = transform(oenc,X_valid,sparse=TRUE)

########################## Currently there are 31552 variables after one-hot encoding ###############


library(xgboost)
train_matrix = xgb.DMatrix(X_train_matrix,label=Y_train)
valid_matrix = xgb.DMatrix(X_valid_matrix,label=Y_valid)

param = list(objective="binary:logistic",eval_metric="auc",booster="gbtree",eta=0.03)
m1 = xgb.train(data=train_matrix,param,nrounds=100,watchlist=list(train_matrix,valid=valid_matrix),print.every.n = 10)

"Notice that in m1, the training auc and validation auc are very close which means the model
is less likely suffering from high variance. Thus if we still want to improve the model, we'd add
more feature to see if it works"

"In the previous data preprocessing step, we remove some features to make certain the one-hot encode
is doable. I am here to add back them one by one to see if it improves the model's performance"

"Among all the removed variables, variable 'char_2.y' has the least number of categories, I will try
to add it back first"

drops = c(1,2,3,14,16) ## not remove 18
keeps = names(train_people)[-drops]

train_people_first = subset(train_people,select=keeps)

Y = train_people_first$outcome
X_train_people_first = train_people_first[,outcome:=NULL]
########################## Set Validation set and implement xgboost ###################################################
set.seed(25000)
valid_ind = sample(1:length(Y),200000)
Y_train = Y[-valid_ind]
X_train = X_train_people_first[-valid_ind,]
Y_valid = Y[valid_ind]
X_valid = X_train_people_first[valid_ind,]

library(CatEncoders)
oenc=OneHotEncoder.fit(train_people_first)
X_train_matrix = transform(oenc,X_train,sparse=TRUE)
X_valid_matrix = transform(oenc,X_valid,sparse=TRUE)

########################## Currently there are 31552 variables after one-hot encoding ###############


library(xgboost)
train_matrix = xgb.DMatrix(X_train_matrix,label=Y_train)
valid_matrix = xgb.DMatrix(X_valid_matrix,label=Y_valid)

param = list(objective="binary:logistic",eval_metric="auc",booster="gbtree",eta=0.03)
m1 = xgb.train(data=train_matrix,param,nrounds=100,watchlist=list(train_matrix,valid=valid_matrix),print.every.n = 10)
