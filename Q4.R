library(tidyverse)
library(mosaic)
library(cluster) # Gap-Statistic
library(corrplot) 
library(foreach)
library(ggplot2)
library(factoextra)
library(NbClust)
library(Hmisc)
library(gridExtra)
library(LICORS) # for kmeans++

# Step 1: Correlation between variables
NH20 = read.csv('./social_marketing.csv', header=TRUE)

NH20_num_data <- NH20[, sapply(NH20, is.numeric)]
summary(NH20_num_data)
NH20_num_data.cor = cor(NH20_num_data)
corrplot(NH20_num_data.cor, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

NH20_num_data.rcorr = rcorr(as.matrix(NH20_num_data))
corrplot(NH20_num_data.rcorr$r, type="upper", order="hclust", 
         NH20_num_data.p = NH20_num_data.rcorr$P , sig.level = 0.01, insig = "blank")

# Step 2: Clustering
#Scaling and Centering
NH20_num_data_scaled = scale(NH20_num_data[,1:36], center=TRUE, scale=TRUE)
mu = attr(NH20_num_data_scaled,"scaled:center")
sigma = attr(NH20_num_data_scaled,"scaled:scale")

#Clustering
#Finding the optimal K

#1. Elbow
k_grid = seq(2, 20, by=1)
SSE_grid = foreach(k = k_grid, .combine='c') %do% {
  cluster_k = kmeans(NH20_num_data_scaled, k, nstart=50,, algorithm = c("Lloyd"))
  cluster_k$tot.withinss
}
plot(k_grid, SSE_grid)
#K at 10

#2. CH index
N = nrow(NH20_num_data_scaled)
CH_grid = foreach(k = k_grid, .combine='c') %do% {
  cluster_k = kmeans(NH20_num_data_scaled, k, iter.max = 100, nstart=50, algorithm = c("Lloyd"))
  W = cluster_k$tot.withinss
  B = cluster_k$betweenss
  CH = (B/W)*((N-k)/(k-1))
  CH
}
plot(k_grid, CH_grid)
#K at 2

#Gap Statistic
NH20_num_data_scaled_gap = clusGap(NH20_num_data_scaled, FUN = kmeans, iter.max = 10000, nstart = 50, K.max = 10, B = 100, algorithm = c("Lloyd"))
plot(NH20_num_data_scaled_gap)
#K at 10

#Clustering
clust1 = kmeans(NH20_num_data_scaled, 10, nstart=25)

clust1$center[1,] #Art seekers - tv-film, music, art 
clust1$center[2,] #The Dark end - adult, spam
clust1$center[3,] #Family/ Household enthusiasts - food, family, parenting, religion, school, sports-fandom
clust1$center[4,] #Current affairs enthusiasts - News, Politics
clust1$center[5,] #Fitness Enthusiasts - Outdoors, Health&Nutrition, Personal fitness
clust1$center[6,] #Social Media people - Photo-sharing, Fashion, Beauty, Cooking
clust1$center[7,] #Inquisitive - Travel, Politics, News, Computers
clust1$center[8,] #No info
clust1$center[9,] #Young Sports enthusiasts - College uni, Sports playing, Online gaming
clust1$center[10,] #Social Butterflies - Chatter, Photo-sharing, Shopping


#Step 3: PCA clustering
#How many PCAs to run 
pr_NH20 = prcomp(NH20_num_data_scaled)
summary(pr_NH20)
scores= pr_NH20$rotation

# Clustering Dendrogram
D_NH20 = dist(scores[,1:10])
hclust_NH20 = hclust(D_NH20, method='complete')
plot(hclust_NH20)

#Fin
