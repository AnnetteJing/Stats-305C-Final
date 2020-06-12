library(clusterGeneration)
library(mvtnorm)
library(randomForest)
library(glmnet)
library(hdm)
library(sparsepca)
library(superpc)
library(ggplot2)
require (reshape)
M = 1000 # Number of simulations
N = 200 # Number of observations
p_Z = 80 # Number of variables in Z
p_W = 80 # Number of variables in W
theta = 1 # True value of the linear effect of X on Y
g_Z = 1/(1:p_Z)^2
g_W = 1/(1:p_Z)^2
g = c(g_Z, g_W)
m_Z = 1/(1:p_Z)
m_W = 1/(p_W:1)
m = c(m_Z, m_W)
K = 5 # Number of PCs selected
# Generate PD matrices as the covariance matrices of Z and W & means of Z and W
set.seed(123)
Sigma_Z = genPositiveDefMat(p_Z,"unifcorrmat")$Sigma 
Sigma_Z = cov2cor(Sigma_Z)
mu_Z = floor(runif(p_Z, min=-5, max=5))
set.seed(456)
Sigma_W = genPositiveDefMat(p_W,"unifcorrmat")$Sigma 
Sigma_W = cov2cor(Sigma_W)
mu_W = floor(runif(p_W, min=-5, max=5))

thetaHat = matrix(NA, M, 7)
colnames(thetaHat) = c("OLS (Control)", "Random Forest", "LASSO", 
                       "PCA - w/o intercept", "PCA - w/ intercept", 
                       "Sparse PCA - w/o intercept", "Sparse PCA - w/ intercept")

DML_iteration <- function(I, IC, gHat1, gHat2, mHat1, mHat2){
  yTilda1 = Y[I] - gHat1
  yTilda2 = Y[IC] - gHat2
  vHat1 = X[I] - mHat1
  vHat2 = X[IC] - mHat2
  thetaHat1 = mean(vHat1*yTilda1)/mean(vHat1*X[I])
  thetaHat2 = mean(vHat2*yTilda2)/mean(vHat2*X[IC])
  return(mean(c(thetaHat1, thetaHat2)))
}
for(i in 1:M)
{
  Z = rmvnorm(N, mean=mu_Z, sigma = Sigma_Z)
  W = rmvnorm(N, mean=mu_W, sigma = Sigma_W)
  ZW = cbind(Z, W)
  V = rnorm(N)
  U = rnorm(N)
  ZWxm = ZW %*% m
  ZWxg = ZW %*% g
  # Nonlinear model
  m_nl = apply(ZWxm, MARGIN=2, function(v) sin(v) + cos(v))
  g_nl = apply(ZWxg, MARGIN=2, function(v) log(abs(v)))
  X = m_nl + V
  Y = X + g_nl + U
  
  ## 1. OLS
  thetaHat[i, 1] = coef(lm(Y~X))[2]
  ## DML
  I = sort(sample(1:N, N/2))
  IC = setdiff(1:N, I)
  # 2. Random Forest
  model_Y1 = randomForest(ZW[IC,], Y[IC], maxnodes = 15)
  model_Y2 = randomForest(ZW[I,], Y[I], maxnodes = 15)
  gHat1 = predict(model_Y1, ZW[I,])
  gHat2 = predict(model_Y2, ZW[IC,])
  model_X1 = randomForest(ZW[IC,], X[IC], maxnodes = 15)
  model_X2 = randomForest(ZW[I,], X[I], maxnodes = 15)
  mHat1 = predict(model_X1, ZW[I,])
  mHat2 = predict(model_X2, ZW[IC,])
  thetaHat[i,2] = DML_iteration(I, IC, gHat1, gHat2, mHat1, mHat2)
  # 3. LASSO
  model_Y1 = rlasso(ZW[IC,], Y[IC])
  model_Y2 = rlasso(ZW[I,], Y[I])
  gHat1 = predict(model_Y1, ZW[I,])
  gHat2 = predict(model_Y2, ZW[IC,])
  model_X1 = rlasso(ZW[IC,], X[IC])
  model_X2 = rlasso(ZW[I,], X[I])
  mHat1 = predict(model_X1, ZW[I,])
  mHat2 = predict(model_X2, ZW[IC,])
  thetaHat[i,3] = DML_iteration(I, IC, gHat1, gHat2, mHat1, mHat2)
  # 4. PCA - w/o intercept
  PCA = prcomp(ZW, retx=TRUE, center=TRUE, scale=TRUE)
  regressors1 = PCA$x[I, 1:K]
  regressors2 = PCA$x[IC, 1:K]
  model_Y1 = lm(Y[IC]~regressors2 - 1)
  model_Y2 = lm(Y[I]~regressors1 - 1)
  model_X1 = lm(X[IC]~regressors2 - 1)
  model_X2 = lm(X[I]~regressors1 - 1)
  gHat1 = regressors1 %*% model_Y1$coefficients
  gHat2 = regressors2 %*% model_Y2$coefficients
  mHat1 = regressors1 %*% model_X1$coefficients
  mHat2 = regressors2 %*% model_X2$coefficients
  thetaHat[i,4] = DML_iteration(I, IC, gHat1, gHat2, mHat1, mHat2)
  # 5. PCA - w/ intercept
  regressors1_intercept = cbind(1, regressors1)
  regressors2_intercept = cbind(1, regressors2)
  model_Y1 = lm(Y[IC]~regressors2)
  model_Y2 = lm(Y[I]~regressors1)
  model_X1 = lm(X[IC]~regressors2)
  model_X2 = lm(X[I]~regressors1)
  gHat1 = regressors1_intercept %*% model_Y1$coefficients
  gHat2 = regressors2_intercept %*% model_Y2$coefficients
  mHat1 = regressors1_intercept %*% model_X1$coefficients
  mHat2 = regressors2_intercept %*% model_X2$coefficients
  thetaHat[i,5] = DML_iteration(I, IC, gHat1, gHat2, mHat1, mHat2)
  # 6. Sparse PCA - w/o intercept
  sPCA = rspca(ZW, k=K, alpha=1e-3, beta=1e-3, center=TRUE, scale=TRUE, verbose=FALSE)
  regressors1 = sPCA$scores[I, ]
  regressors2 = sPCA$scores[IC, ]
  model_Y1 = lm(Y[IC]~regressors2 - 1)
  model_Y2 = lm(Y[I]~regressors1 - 1)
  model_X1 = lm(X[IC]~regressors2 - 1)
  model_X2 = lm(X[I]~regressors1 - 1)
  gHat1 = regressors1 %*% model_Y1$coefficients
  gHat2 = regressors2 %*% model_Y2$coefficients
  mHat1 = regressors1 %*% model_X1$coefficients
  mHat2 = regressors2 %*% model_X2$coefficients
  thetaHat[i,6] = DML_iteration(I, IC, gHat1, gHat2, mHat1, mHat2)
  # 7. Sparse PCA - w/ intercept
  regressors1_intercept = cbind(1, regressors1)
  regressors2_intercept = cbind(1, regressors2)
  model_Y1 = lm(Y[IC]~regressors2)
  model_Y2 = lm(Y[I]~regressors1)
  model_X1 = lm(X[IC]~regressors2)
  model_X2 = lm(X[I]~regressors1)
  gHat1 = regressors1_intercept %*% model_Y1$coefficients
  gHat2 = regressors2_intercept %*% model_Y2$coefficients
  mHat1 = regressors1_intercept %*% model_X1$coefficients
  mHat2 = regressors2_intercept %*% model_X2$coefficients
  thetaHat[i,7] = DML_iteration(I, IC, gHat1, gHat2, mHat1, mHat2)
}
colMeans(thetaHat)
colnames(thetaHat) = c("OLS", "Random_Forest", "LASSO", 
                       "PCA_without_intercept", "PCA_with_intercept", 
                       "sPCA_without_intercept", "sPCA_with_intercept")
thetaHat_df = melt(data.frame(thetaHat))
ggplot(aes(x=value, color=variable), data=thetaHat_df) + geom_density(size=1.2) + geom_hline(yintercept=0, colour="black", size=1.2) + geom_vline(xintercept=theta, color="black", size=0.5, linetype="dashed") + scale_x_continuous(breaks = seq(0.4, 1.8, by = 0.2)) + theme(legend.position = c(0.75, 0.78), legend.text=element_text(size=30), legend.title=element_blank())
write.csv(data.frame(thetaHat), "/Users/sleeper/Desktop/Stats 305/305C Final Project/thetaHat_nonlinear.csv", row.names=FALSE)

