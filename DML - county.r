library(randomForest)
library(glmnet)
library(hdm)
library(ggplot2)
library(tidyverse)
library(stringr)
library(dummies)
library(sandwich)

# Simple DML2 Code for Partrially Linear Model
DML2.for.PLM <- function(z, w=NULL, x, y, dreg, yreg, nfold=2) {
  # this implements DML2 algorithm, where there moments are estimated via DML, before constructing
  # the pooled estimate of theta randomly split data into folds
  nobs <- nrow(z)
  foldid <- rep.int(1:nfold,times = ceiling(nobs/nfold))[sample.int(nobs)]
  I <- split(1:nobs, foldid)
  # create residualized objects to fill
  ytil <- xtil <- rep(NA, nobs)
  # obtain cross-fitted residuals
  if(is.null(w)){
    zw = z
  }
  else{
    zw = cbind(z, w)
  }
  cat("fold: ")
  for(b in 1:length(I)){
    xfit <- dreg(zw[-I[[b]],], x[-I[[b]]])  #take a fold out
    yfit <- yreg(zw[-I[[b]],], y[-I[[b]]])  # take a folot out
    xhat <- predict(xfit, zw[I[[b]],], type="response")  #predict the fold out
    yhat <- predict(yfit, zw[I[[b]],], type="response")  #predict the fold out
    xtil[I[[b]]] <- (x[I[[b]]] - xhat) #record residual
    ytil[I[[b]]] <- (y[I[[b]]] - yhat) #record residial
    cat(b," ")
  }
  rfit <- lm(ytil ~ xtil)               #estimate the main parameter by regressing one residual on the other
  coef.est <- coef(rfit)[2]             #extract coefficient 
  se <- sqrt(vcovHC(rfit)[2,2])         #record standard error
  cat(sprintf("\ncoef (se) = %g (%g)\n", coef.est , se))
  return( list(coef.est =coef.est , se=se, xtil=xtil, ytil=ytil) )
}
data <- na.omit(read.table("~/Desktop/Stats 305/305C HW2/county_health.csv", sep=",", header = TRUE))
Z = data %>% select(2, 20:22, 30:32, 52, 58:60) %>% dummy.data.frame(sep=".")
W = data %>% select(8:10, 13:15, 18:19, 24:25, 33, 37:43, 46:51) %>% dummy.data.frame(sep=".")
# OLS - No control
model_OLS_noControl = lm(data$age_adjusted_death_rate~data$percent_vaccinated)
summary(model_OLS_noControl)
# OLS - With control
model_OLS_Control = lm(data$age_adjusted_death_rate~data$percent_vaccinated + ., data=cbind(Z, W))
summary(model_OLS_Control)
# DML - Random forest
Xreg <- function(Z,X){ randomForest(Z, X, maxnodes = 15) }
Yreg <- function(Z,Y){ randomForest(Z, Y, maxnodes = 15) }
model_DML_RF = DML2.for.PLM(Z, W, data$percent_vaccinated, data$age_adjusted_death_rate, Xreg, Yreg, nfold=2)
# DML - LASSO
Xreg <- function(Z,X){ rlasso(Z, X) } 
Yreg <- function(Z,Y){ rlasso(Z, Y) }
model_DML_LASSO = DML2.for.PLM(Z, W, data$percent_vaccinated, data$age_adjusted_death_rate, Xreg, Yreg, nfold=2)


