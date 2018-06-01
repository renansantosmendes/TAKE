setwd("/home/renansantos/Área de Trabalho/Machine Learning/TAKE")
dados <- read.csv("data.csv",sep = ",", header = TRUE)

cor.test(dados$times_used,dados$hour,method = "pearson")
cor.test(dados$times_used,dados$hour,method = "spearman")
cor.test(dados$times_used,dados$hour,method = "kendall")

