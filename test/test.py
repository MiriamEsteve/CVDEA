import numpy as np
import random
import cvdea


N = 50
sd = random.randint(0, 1000) #int(time.time())
print(sd)

################################# Mono input ###################################
matrix = cvdea.Data(sd, N, 1).data
matrixTheoric = matrix.copy()
matrix = matrix.iloc[:, :-1]

y = [matrix.columns[-1]]
x = list(matrix.drop(y, axis=1).columns)


# Pilling DEA
D = 10 #Depth of configurations
P = 10 #The number of different combinations of supporting hyperplanes

# Create model
model = cvdea.CVDEA(matrix, x, y, D, P)
# Fit model
model.fit_CVDEA()

#Prediction
matrix_pred = model.prediction(matrix)

# DEA
dataDEA = model.DEA()

#Error theoric mono
errPillingDEA = model.MSE_theoric(matrixTheoric, matrix_pred, "pred_y")
biasPillingDEA = abs(np.sqrt(errPillingDEA))
errDEA = model.MSE_theoric(matrixTheoric, dataDEA, "DEA")
biasDEA = abs(np.sqrt(errDEA))


# Graphic frontier
yTheoric = matrixTheoric["yD"]
matrix_pred = matrix_pred.rename(columns={"pred_y": "ypillingDEA"})
ycvDEA = matrix_pred["ypillingDEA"]

yDEA = dataDEA["yDEA"]

name_graphic = "cvdea"
model.print_Frontiers(matrix, yTheoric, ycvDEA, yDEA, name_graphic)

################################# Multi output ###################################
'''
sd = 0
N = 50
frontier = 1
noise = 0
datamodel = data.Data2(sd, N, frontier, noise)
matrix = datamodel.data.copy()


y = [matrix.columns[-2], matrix.columns[-1]]
x = list(matrix.drop(y, axis=1).columns)

# Theoric frontier
datamodel.fit_Theoric(x, y)
matrixTheoric = datamodel.data.copy()

# Pilling DEA
D = 10
P = 10

# Create model
model = cvdea.CVDEA(matrix, x, y, D, P)
# Fit model
model.fit_pillingDEA()

#Prediction
matrix_pred = model.prediction(matrix)

# DEA
dataDEA = model.DEA()

#Error theoric multi
msecvDEA = model.MSE_theoric_multi(matrixTheoric, matrix_pred, "pred_")
mseDEA = model.MSE_theoric_multi(matrixTheoric, dataDEA, "yDEA")
'''