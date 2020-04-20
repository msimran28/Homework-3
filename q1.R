# import data and examine it
library(gamlr) 
greenbuildings <- read.csv("C:/Users/swate/Google Drive/MA Econ/spring 2020 classes/data mining/hw 3/greenbuildings.csv")
View(greenbuildings)
ok <- complete.cases(greenbuildings)
greenbuildings <- greenbuildings[ok,]

# chech the correlation between rent and cluster rent
cor(greenbuildings$Rent, greenbuildings$cluster_rent)

# note that rent and cluster_rent are hugely skewed
# probably want a log transformation here
hist(greenbuildings$Rent)
summary(greenbuildings$Rent)

hist(greenbuildings$cluster_rent)

# much nicer :-)
hist(log(greenbuildings$Rent))
hist(log(greenbuildings$cluster_rent))

# i create a matrix of all my independent varaibles except for - CS_PropertyID - LEED -Energystar -cluster_rent -cluster  to make it easily readable for gamlr commands.
x = sparse.model.matrix( log(Rent) ~  . - CS_PropertyID - LEED -Energystar -cluster_rent -cluster , data=greenbuildings)[,-1] # do -1 to drop intercep

# I create a variable y that normalizes Log(rent) over log(cluster_rent)
y = (log(greenbuildings$Rent)- log(greenbuildings$cluster_rent))/sd(log(greenbuildings$cluster_rent))

# Here I fit my lasso regression to the data and do my cross validation of k=10 n folds
# the cv.gamlr command does both things at once.
#(verb just prints progress)
cvl = cv.gamlr(x, y, nfold=10, verb=TRUE)

# plot the out-of-sample deviance as a function of log lambda
plot(cvl, bty="n")

## CV min deviance selection
b.min = coef(cvl, select="min")
log(cvl$lambda.min) # this gives the value of lamda
sum(b.min!=0) # this gives the coefficent not 0

coef(cvl)


