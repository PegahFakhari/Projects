library(foreign)
rm(list=ls())
dataset1a = read.spss("C:\\Users\\pfakhari\\Google Drive\\Courses\\S690\\Assignment 3\\Research Paper\\DS1a_Self data Restructured.sav")
dataset1a = data.frame(dataset1a)

dataset1b = read.spss("C:\\Users\\pfakhari\\Google Drive\\Courses\\S690\\Assignment 3\\Research Paper\\DS1b_Other data Restructured.sav")
dataset1b = data.frame(dataset1b)

dataset2 = read.spss("C:\\Users\\pfakhari\\Google Drive\\Courses\\S690\\Assignment 3\\Research Paper\\DS2 Restructured2.sav")
dataset2 = data.frame(dataset2)

library (nlme)
library (lme4)

m1DS2 = lme(att ~ 1, data=dataset2, random=~1|Subject,method="ML")
m2DS2 = lme(att ~ 1 + SongType, data=dataset2, random=~1|Subject,method="ML")


m1DS1a = lme(att ~ 1, data=dataset1a, random=~1|Subject,method="ML")
m2DS1a = lme(att ~ 1 + song_type, data=dataset1a, random=~1|Subject,method="ML")
m3DS1a = lme(att ~ 1 + song_type + lrc_type, data=dataset1a, random=~1|Subject,method="ML")


m1DS1b = lme(att ~ 1, data=dataset1b, random=~1|Subject,method="ML")
m2DS1b = lme(att ~ 1 + song_type, data=dataset1b, random=~1|Subject,method="ML")
m3DS1b = lme(att ~ 1 + song_type + lrc_type, data=dataset1b, random=~1|Subject,method="ML")


summary(m1DS2)
summary(m2DS2)

summary(m1DS1a)
summary(m2DS1a)
summary(m3DS1a)

summary(m1DS1b)
summary(m2DS1b)
summary(m3DS1b)

var(m2DS2$fitted[, "fixed"]-m3DS1a$fitted[, "fixed"])
var(m2DS2$fitted[, "fixed"]-m3DS1b$fitted[, "fixed"])
VarCorr(m1DS2)[, 1]
VarCorr(m2DS2)[, 1]
VarCorr(m1DS1a)[, 1]
VarCorr(m2DS1a)[, 1]
VarCorr(m3DS1a)[, 1]
VarCorr(m1DS1b)[, 1]
VarCorr(m2DS1b)[, 1]
VarCorr(m3DS1b)[, 1]


