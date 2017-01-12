library(foreign)
rm(list=ls())
dataset = read.spss("C:\\Users\\pfakhari\\Google Drive\\Courses\\S690\\Assignment 4\\Data.sav", , to.data.frame=TRUE)

levels(dataset$T056301)
levels(dataset$DisabilityLevel)

# change yes no to 0 and 1
names=colnames(dataset)
numcol = ncol(dataset)
numrow = nrow(dataset)
dataset3 = matrix(0, numrow, numcol-5)

# convert the first 54 columns to binary 0 and 1
for (i in 1:(numcol-5)){
  dataset3[,i] =as.numeric(dataset[,i])
}


library(psych)
library(polycor)
library(bindata)
library(psy)

x <- as.data.frame(dataset3)
x1<-na.omit(x) # remove NA
x2=t(x1)
ind = matrix(1, nrow=nrow(x2))

# find columns with all 1 or all 2
VecSum = colSums (x1, na.rm = FALSE, dims = 1)
ind[VecSum==nrow(x1)]=0;ind[VecSum==2*nrow(x1)]=0;
x3=t(subset(x2,as.logical(ind)))

R <- tetrachoric(x3)$rho

fit = factanal(covmat = R, factors = 2, rotation = "varimax")
print(fit, digits = 2, cutoff = .2, sort = TRUE) 

A <- factanal(x = x3, covmat = R, factors = 2, rotation = "varimax")$loadings[,1:2]

as.matrix(x3)%*%solve(R)%*%A

scale(as.matrix(x3))%*%solve(R)%*%A

# Estimating factor scores:

scores <- factanal(x3,factors=2, rotation="varimax",scores="regression")$scores


# EFA separately on PD and TS:
xPD = x1[, 5:11]
xPD = cbind(xPD, x1[,26:37, 54])

xTS = x1[, 12:25]
xTS = cbind(xTS, x1[,38:53])

RPD <- tetrachoric(xPD)$rho
RTS <- tetrachoric(xTS)$rho

fitPD = factanal(covmat = RPD, factors = 3, rotation = "varimax")

fitTS = factanal(covmat = RTS, factors = 2, rotation = "varimax")

print(fitPD, digits = 2, cutoff = .2, sort = TRUE) 
print(fitTS, digits = 2, cutoff = .2, sort = TRUE) 


APD <- factanal(x = xPD, covmat = RPD, factors = 3, rotation = "varimax")$loadings[,1:2]
ATS <- factanal(x = xTS, covmat = RTS, factors = 2, rotation = "varimax")$loadings[,1:2]


as.matrix(x3)%*%solve(R)%*%A

scale(as.matrix(x3))%*%solve(R)%*%A


