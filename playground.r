library(ggplot2)
library("dplyr")


score<-list.files(path = "data/", pattern = NULL, all.files = TRUE, full.names = FALSE, recursive = FALSE, ignore.case = FALSE, include.dirs = FALSE, no.. = FALSE)
toxicity<-substr(score,1,3)
toxicity<-toxicity[5:length(toxicity)]
score<- data.frame(toxicity)

head(score)

ggplot(score ,aes(x=toxicity))+   stat_count(fill="#69b3a2", color="#e9ecef",)
