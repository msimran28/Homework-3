library(ggplot2)
library(foreach)
library(LICORS)
library(caret)
library(RColorBrewer)

#We need to import the data on wine chemical properties and two other variables.
wine_df = read.csv("C:/Users/Nyarlathotep/Documents/Econ - Data Mining/Exercise 3/wine.csv")

#The purpose of this exercise is to determine whether unsupervised learning on
#a set of chemical properties can be used to categorize data in such a way
#that it aligns with privileged outcomes omitted from the data set. These
#privileged outcomes are the color of wine and the overall quality. We will
#also comment on which dimension reduction technique is best for this task.

#We will subset the data so that only the chemical properties are considered.
wine_chem = wine_df[1:11]

#normalize the data to adjust for scale. Otherwise, measuring distance between points
#will not be meaningful.
wine_chem_nm = scale(wine_chem, center = TRUE, scale = TRUE)

#First, we will examine how clustering can be used to determine color.

#run k-means with 2 clusters and 25 starts.
clust_k2 = kmeanspp(wine_chem_nm, 2, nstart = 25)


#Before visualizing the clusters, we need to narrow down the variables that we
#want to compare. Let's look at correlation between each chemical property
#and a given wine color.

cor(wine_chem, as.numeric(wine_df$color))

#Note that here higher positive correlation means that the property is more closely
#associated with white wine. We will choose the covariates with the greatest
#correlation (positive or negative) with wine color. In this case, it would be
#volatile acidity, chlorides, total sulfur dioxide, and sulphates.
#We will compare the selected variables by color and them by cluster.

#Comparing select variables by wine color.
pairs(wine_df[, c(2,5,7,10)], col = c("red", "grey")[wine_df$color],
      main = "Chemical Properties of wine: Red vs White",
      oma = c(5,5,5,15))
par(xpd = TRUE)
legend("right", 
       fill = c("red","grey"), 
       legend = c( levels(wine_df$color)),
       title = "Color")


#Now, let's look at cluster.
pairs(wine_df[, c(2,5,7,10)], col = c("red", "grey")[clust_k2$cluster],
      main = "Chemical Properties of wine: Cluster 1 vs Cluster 2",
      oma = c(5,5,5,15))
par(xpd = TRUE)
legend("right", 
       fill = c("red", "grey"), 
       legend = c( levels(as.factor(clust_k2$cluster))),
       title = "cluster")


#At least on sight, the clustering algorithm seems to separate the red
#and white wines effectively.

#We assign red wine to cluster 1 and white wine to cluster 2, and show the 
#confusion matrix.
y_hat1 = clust_k2$cluster
y_hat1 = ifelse(y_hat1 ==1, "red", "white")

confusionMatrix(data = as.factor(y_hat1), reference = wine_df$color)

#To be thorough, we will run the null model.

confusionMatrix(data = as.factor(rep("white", length(clust_k2$cluster))), reference = wine_df$color)

#As we see, from the confusion matrix, clustering seems to be a fairly good
#method of sorting red and white wines. 

#However, clustering does not seem to be the most appropriate method for sorting
#if we are interested in determining what chemical balance is associated with a given
#wine color. Let's consider this while we look at another dimension reduction 
#technique, principle components analysis.

#PCA allows for mixed membership of covariates to construct principle components.

pc_chem = prcomp(wine_chem_nm)

pc_chem$rotation[,1]

#Now, we want to do some visualization.

qplot(pc_chem$x[,1], pc_chem$x[,2], 
      color=wine_df$color, xlab='Component 1', 
      ylab='Component 2',
      geom = c("point", "abline"),
      intercept = 2.5, slope = 2,
      main = "Principle Components Comparison: Wine Colors") + 
  labs(color = 'color')

#We see that red and white wine split roughly along the line component2 = component1 - 2.5.
#We can take this information and see if a point lands on either side of this line to 
#determine whether the wine is red or white.

y_hat2 = pc_chem$x[,2] - 2*pc_chem$x[,1] - 2.5
y_hat2 = ifelse(y_hat2 > 0, "red", "white")

confusionMatrix(data = as.factor(y_hat2), reference = wine_df$color)

#This method is fairly accurate. We could
#fine tune the line to get a better prediction, but we are not sure how possible
#this would by just looking at unsupervised information. Even the way we constructed
#line above required us to peek at that shape of the data.CPA may be preferable
#if we are trying gleam something about the chemical composition of either red
#or white wine, but if we are just looking at classification, clustering seems
#to be the better method.

#Now, let's move on to classifying wine by quality instead of color.

#Using k-means ++, and setting k = 10 may work since wines are rated on a scale of 1 
#to 10. Using k-means ++ and setting k = 10 may work since wines are rated on a scale of 
#1 to 10. But, we will let k = 7. We will see whay later as to why we want to choose 7 over 10.
k_grid = seq(2,20, by = 1) #vector containing various values of k

#Let's cluster the data, find the covariates  of greatest interest, and visualize the clusters.
clust_k7 = kmeanspp(wine_chem_nm, 7, nstart = 25)

cor(wine_chem, as.numeric(wine_df$quality))

#Note, there are not that many chemical properties with a particular strong 
#correlation with quality.


#Like previously, here is a set of pairwise graph of select chemical properties,
#fill is by quality score. Note, that while the wines were score on a scale of 1 to
#10, only scores of 3 through 9 were given.

pairs(wine_df[, c(2,5,8,11)], 
      col = brewer.pal(n = 7, name = "RdBu")[wine_df$quality],
      main = "Chemical Properties of Wine: Quality Scores",
      oma=c(5,5,5,15))
par(xpd = TRUE)
legend("right", 
       fill = brewer.pal(n = 7, name = "RdBu"), 
       legend = c( levels(as.factor(wine_df$quality))),
       title = "Score")

pairs(wine_df[, c(2,5,8,11)], 
      col = brewer.pal(n = 10, name = "RdBu")[as.factor(clust_k10$cluster)],
      main = "Chemical Properties of Wine: Clusters",
      oma=c(5,5,5,15))
par(xpd = TRUE)
legend("right", 
       fill = brewer.pal(n = 10, name = "RdBu"), 
       legend = c( levels(as.factor(clust_k10$cluster))),
       title = "Score")


#We cannot see any "clean" clusters from the plots. What if we were less ambitious 
#and only looked at "high quality" and "low "quality" wines? Here, "high quality" 
#corresponds to scores 7 and higher, while "low quality" corresponds to scores 6 or 
#lower.

wine_qual = wine_df
wine_qual$quality = ifelse(wine_df$quality < 7, "low", "high")

pairs(wine_qual[, c(2,5,8,11)], 
      col = c("red","grey")[as.factor(wine_qual$quality)],
      main = "Chemical properties of Wine: High vs Low Quality",
      oma = c(5,5,5,15))
legend("right", 
       fill = c("red","grey"), 
       legend = c( levels(as.factor(wine_qual$quality))),
       title = "Quality")

pairs(wine_qual[, c(2,5,8,11)], 
      col = c("grey","red")[clust_k2$cluster],
      main = "Chemical properties of Wine: Cluster 1 vs Cluster 2",
      oma = c(5,5,5,15))
legend("right", 
       fill = c("red","grey"), 
       legend = c( levels(as.factor(clust_k2$cluster))),
       title = "Cluster")

#When just considering a binary choice between high and low quality wines versus
#k = 10 many clusters, the clustering algorithm preforms better, but not necessarily
#well. Let's run a confusion matrix, just to check.

y_hat3 = clust_k2$cluster
y_hat3 = ifelse(clust_k2$cluster == 1, "low", "high")

confusionMatrix(data = as.factor(y_hat3), reference = as.factor(wine_qual$quality))

#Our accuracy is 61%, which is worse than our null model.
#PCA may be more appropriate for this classification problem. We've already run 
#our PCA algorithm, so let's go straight to the plots.

qplot(pc_chem$x[,1], pc_chem$x[,2], 
      color=wine_df$quality, 
      main = "Principle Components Comparison: Quality Score",
      xlab='Component 1', 
      ylab='Component 2') + 
  labs(color = 'Quality')

#Let's look at a few more pairwise comparison.

pairs(pc_chem$x[, c(1,2,3,4)],
      col = brewer.pal(n = 7, name = "RdBu")[wine_df$quality],
      oma = c(5,5,5, 15),
      main = "Principle Components Comparision: Quality Score")
legend("right", 
       fill = brewer.pal(n = 7, name = "RdBu"), 
       legend = c( levels(as.factor(wine_df$quality))),
       title = "Score")


#Again, there is no clear delineation what properties correspond to quality
#rankings. Again, let's look at sorting on just "high" and "low" quality wines.

qplot(pc_chem$x[,1], pc_chem$x[,2], 
      color= factor(wine_qual$quality), 
      main = "Principle Components Comparison: High vs Low",
      xlab='Component 1', 
      ylab='Component 2',
      geom = c("point", "abline")) +
  labs(color = "Quality")

#There is not clean divide among the types of wine. To verify, let's look at
#pairwise plots for the first four principle components.

pairs(pc_chem$x[, c(1,2,3,4)],
      col = c("red", "blue")[as.factor(wine_qual$quality)],
      main = "Principle Components Comparision: High vs Low",
      oma = c(5,5,5,15))
legend("right", 
       fill = c("red", "blue"), 
       legend = c( levels(as.factor(wine_qual$quality))),
       title = "Quality")

#It doesn't seem that PCA is capable of sorting high vs. low quality wine.


