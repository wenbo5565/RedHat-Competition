library(data.table)
library(dplyr)
library(ggplot2)
library(knitr)

setwd("C:/Users/wenbma/Desktop/Others/Kaggle/RedHat/act_train.csv")
train = read.csv("act_train.csv")
kable(sample_n(train,4))
str(train)

train %>%
  count(outcome) %>% 
  ggplot(aes(x=outcome, y=n)) + 
  geom_bar(stat = "identity", width=0.6, fill="light blue") + 
  ggtitle("Outcomes")

train %>%
  ggplot(aes(x=outcome,fill=activity_category)) +
  geom_bar(width=0.6,position="fill")+
  ggtitle("Outcome by activity category")

train %>%
  ggplot(aes(x=outcome,fill=char_5)) + 
  geom_bar(width=0.6,position="fill") +
  ggtitle("Outcome by char_5")

train %>%
  filter(char_5!="") %>%
  ggplot(aes(x=outcome,fill=char_5))+
  geom_bar(width=0.6,position="fill")+
    ggtitle("Outcome by char 5  where char 5 not blank")

train %>%
  count(char_5) %>%
  ggplot(aes(x=reorder(char_5,n),y=n)) + 
  geom_bar(stat='identity',fill='light blue') +
  coord_flip()+
  ggtitle("Count of char_5")

train %>%
  count(char_10,sort=TRUE) -> count_char_10
dim(count_char_10)

kable(head(count_char_10))

train %>%
  count(char_10,sort=TRUE) %>%
  filter(n>8000) %>%
  ggplot(aes(x=reorder(char_10,n),y=n)) +
  geom_bar(stat="identity") +
  coord_flip() + 
  ggtitle("Distribution of char_10")

plot(cumsum(count_char_10$n[1:1000])/sum(count_char_10$n),
     type="b",pch=".",main="Cumulative Percent by types of char_10",ylab="cumulative percent")

popular = count_char_10$char_10[1:15]
train %>%
  filter(char_10 %in% popular) %>%
  ggplot(aes(x=outcome,fill=char_10)) +
  geom_bar(width=0.6,position="fill") +
    ggtitle("Outcome by char_10")

train %>%
  filter(char_10 %in% popular) %>%
  ggplot(aes(x=char_10,fill=char_10)) +
  geom_bar() +
  facet_wrap(~outcome) + 
  coord_flip() +
  ggtitle("Outcome by char_10")

train %>%
  count(people_id,sort=TRUE) -> people_counts

people_counts %>%
  ggplot(aes(x=n)) + geom_histogram(color="grey")

train %>%
  filter(people_id %in% people_counts$people_id[1:10]) %>%
  ggplot(aes(x=outcome,fill=people_id)) + 
  geom_bar(position="fill")

train %>%
  filter(people_id %in% people_counts$people_id[1:20]) %>%
  ggplot(aes(x=people_id)) +
  geom_bar() +
  facet_wrap(~outcome) + 
  coord_flip() +
  ggtitle("Outcome by people_id")

train %>%
  filter(people_id %in% people_counts$people_id[1:10000]) %>%
  group_by(people_id) %>%
  summarize(cont=n(),outcome_pos=sum(outcome)) %>%
  mutate(frac_pos = outcome_pos/cont) %>%
  ggplot(aes(x=frac_pos))+geom_histogram() +
  ggtitle("Fraction of Positive Outcomes by person")

setwd("C:/Users/wenbma/Desktop/Others/Kaggle/RedHat/act_test.csv")
test = read.csv("act_test.csv")
  
test %>%
  filter(people_id %in% people_counts$people_id[1:100])

intersect(test$people_id,train$people_id)

setwd("C:/Users/wenbma/Desktop/Others/Kaggle/RedHat/people.csv")
people = read.csv("people.csv")
train %>%
  merge(people,all.x=TRUE,by="people_id") -> train

kable(head(train))

################### how many unique values we have for feature ################
sapply(train, function(x) length(unique(x)))

train %>%
  count(group_1,sort=TRUE) -> group_count %>%

train %>%
  ggplot(aes(x=char_38)) + 
  geom_histogram()
  ggtitle("Distribution of char_38")
  
train %>%
  ggplot(aes(x=char_38,fill=as.factor(outcome)))+
  geom_histogram() + 
  ggtitle("Distribution of char_38")+
  scale_fill_brewer(palette="Set1")+
  facet_wrap(~outcome)+
  ggtitle("Distribution of char_38 faceted by outcome")

train %>%
  ggplot(aes(x=char_38,fill=as.factor(outcome)))+
  geom_histogram(position="fill") +
  scale_fill_brewer(palette="Set1") +
  ggtitle("Proportion of outcomes by value of char_38")
