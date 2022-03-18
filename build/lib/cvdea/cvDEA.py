import pandas as pd
import numpy as np
from docplex.mp.model import Model
import math
INF = math.inf
import matplotlib.pyplot as plt

class CVDEA:
    def __init__(self, matrix, x, y, D, P): # D = "profundidades" del pilling; P = P caminos
        self._check_enter_parameters(matrix, x, y)

        self.xCol = x
        self.yCol = y

        # Matrix original
        self.matrix = matrix.loc[:, x + y]  # Order variables
        self.N = len(self.matrix)

        self.x = matrix.columns.get_indexer(x).tolist()  # Index var.ind in matrix
        self.y = matrix.columns.get_indexer(y).tolist()  # Index var. obj in matrix
        self.m = len(x) # Num. inputs
        self.s = len(y) # Num. outputs

        # Train and test
        self.train = []
        self.test = []

        # Hiperparámetros a tunear
        self.D = D # "Profundidades" del pilling
        self.P = P # P caminos

        # Matrix model
        self.c_p_alpha_matrix = pd.DataFrame() # Matrix with c*, p*, alpha* and alpha** --> Lista de hiperplanos (h1, ..., h_nTrain)

        # List of hyperplanes to remove
        self.P_path = []
        self.H_path = []

        # Result
        self.P_path_result = []
        self.H_path_result = []
        self.c_p_alpha_matrix_result = []
        self.P_result = 0
        self.D_result = 0

    'Destructor'
    def __del__(self):
        try:
            '''
            del self.N
            del self.matrix
            del self.m
            del self.s
            '''
            del self.H_path
            del self.H_path
            del self.c_p_alpha_matrix

        except Exception:
            pass

    def _check_enter_parameters(self, matrix, x, y):
        #var. x and var. y have been procesed
        if type(x[0]) == int or type(y[0]) == int:
            self.matrix = matrix
            if any(self.matrix.dtypes == 'int64'):
                self.matrix = self.matrix.astype('float64')
            return
        else:
            self.matrix = matrix.loc[:, x + y]  # Order variables
            if any(self.matrix.dtypes == 'int64'):
                self.matrix = self.matrix.astype('float64')

        if len(matrix) == 0:
            raise EXIT("ERROR. The dataset must contain data")
        elif len(x) == 0:
            raise EXIT("ERROR. The inputs of dataset must contain data")
        elif len(y) == 0:
            raise EXIT("ERROR. The outputs of dataset must contain data")
        else:
            cols = x + y
            for col in cols:
                if col not in matrix.columns.tolist():
                    raise EXIT("ERROR. The names of the inputs or outputs are not in the dataset")

            for col in x:
                if col in y:
                    raise EXIT("ERROR. The names of the inputs and the outputs are overlapping")

    def _trainingTest(self):
        # 70%-30%
        training = self.matrix.sample(frac=0.7)
        test = self.matrix.drop(list(training.index)).reset_index(drop=True)
        training = training.reset_index(drop=True)

        return training, test

    def fit_pillingDEA(self):
        err_min = INF

        ######################################### Dividir matrix en train y test #######################################
        self.train, self.test = self._trainingTest()
        self.N_train = len(self.train)
        self.N_test = len(self.test)

        ################################## Crear lista de hiperplanos (h1,...,h_nTrain) ################################
        # Para cada (x_k, y_k) € train calcular modelo de optimización
        self.c_p_alpha_matrix = pd.DataFrame() # Matrix with c*, p*, alpha* and alpha**

        for i in range(self.N_train):
            c, p, alpha_1 = self.c_p_alpha_model(self.train.iloc[i, self.x].tolist(), self.train.iloc[i, self.y].tolist())
            # Add result in c_p_alpha_matrix
            self.c_p_alpha_matrix = self.c_p_alpha_matrix.append({"c*": c, "p*": p, "alpha*": alpha_1, "alpha**": None}, ignore_index=True)

        ######################### Probar con varias "profundidades" de pilling (d <= d_nTrain) ##########################
        if self.D > self.N_train: raise EXIT("ERROR. The 'D' hyperparameter are higher to nTrain")
        for d in range(self.D):
            for p in range(self.P):
                # Select random hyperplanes to remove
                self.P_path = self.c_p_alpha_matrix.sample(n=d+1).index.tolist()
                print("P = ", self.P_path)
                self.H_path = self.c_p_alpha_matrix.drop(self.P_path).index.tolist()
                #Calculate alpha**
                alpha_2 = self.alpha_model()
                #Add alpha** in c_p_alpha_matrix in hyperplanes no remove
                for l in range(len(alpha_2)):
                    self.c_p_alpha_matrix.loc[l, "alpha**"] = alpha_2[l]

                # Calculate prediction error for each (xi, yi) € Stest
                self.prediction_error()

                # Error asociado a P_path
                err = 0
                for i in range(self.N_test):
                    for j in range(self.s):
                        err += (self.test.iloc[i, self.y[j]] - self.test.iloc[i, self.y[j]] * self.test.iloc[i, -1])**2
                err = err/self.N_test

                if err < err_min:
                    err_min = err
                    self.P_path_result = self.P_path.copy()
                    self.H_path_result = self.H_path.copy()
                    self.c_p_alpha_matrix_result = self.c_p_alpha_matrix.copy()
                    self.P_result = p+1
                    self.D_result = d+1
                    print("p = ", p, " - d = ", d, " - err = ", err, " - P_path = ", self.P_path)

        self.__del__()



    # Modelo (1)
    def c_p_alpha_model(self, x_k, y_k):
        # create one model instance, with a name
        m = Model(name='pillingDEA_1')

        # Constrain c, p, alpha
        c = {i: m.continuous_var(name="c_{0}".format(i)) for i in range(self.m)}  # >= 0 para cada input
        p = {i: m.continuous_var(name="p_{0}".format(i)) for i in range(self.s)}  # >= 0 para cada output
        alpha = {0: m.continuous_var(name="alpha", lb=-INF, ub=INF)}  # free

        # Constrain 1.1
        for i in range(self.N_train):
            x_i = self.train.iloc[i, self.x].tolist()
            y_i = self.train.iloc[i, self.y].tolist()

            m.add_constraint(m.sum(p[r] * y_i[r] for r in range(self.s)) \
                             - m.sum(c[j] * x_i[j] for j in range(self.m)) \
                             - alpha[0] <= 0)
        # Constrain 1.2
        m.add_constraint(m.sum(p[r] * y_k[r] for r in range(self.s)) == 1)

        # Objetive
        m.minimize(m.sum(c[j] * x_k[j] for j in range(self.m)) + alpha[0])

        # Model Information
        #m.export()

        m.solve(agent='local')

        # Solution
        if m.solution is not None:
            sol_c = []
            sol_p = []
            for i in range(self.m):
                sol_c.append(round(m.solution.get_value("c_" + str(i)), 6))
            for i in range(self.s):
                sol_p.append(round(m.solution.get_value("p_" + str(i)), 6))

            sol_alpha = round(m.solution.get_value("alpha"), 6)
        else:
            sol_c = sol_p = [0 for i in range(self.s)]
            sol_alpha = 0

        return sol_c, sol_p, sol_alpha

    # Modelo (2)
    def alpha_model(self):
        N_H_path = len(self.H_path)
        # create one model instance, with a name
        m = Model(name='pillingDEA_2')

        # Constrain alpha
        alpha = {i: m.continuous_var(name="alpha_{0}".format(i), lb=-INF, ub=INF) for i in self.H_path}  # free

        # Constrain 1.1
        for l in self.H_path:
            p = self.c_p_alpha_matrix.iloc[l]["p*"]
            c = self.c_p_alpha_matrix.iloc[l]["c*"]
            alpha_1 = self.c_p_alpha_matrix.iloc[l]["alpha*"]

            # Constrain 1.1
            for i in range(self.N_test):
                x_i = self.test.iloc[i, self.x].tolist()
                y_i = self.test.iloc[i, self.y].tolist()

                m.add_constraint(m.sum(p[r] * y_i[r] for r in range(self.s)) \
                                 - m.sum(c[j] * x_i[j] for j in range(self.m)) \
                                 - alpha[l] <= 0)
            # Constrain 1.2
            m.add_constraint(alpha[l] >= alpha_1)

        # Objetive
        m.minimize(m.sum(alpha[l] for l in self.H_path))

        # Model Information
        # m.export()

        m.solve(agent='local')

        # Solution
        if m.solution is not None:
            sol_alpha = [None] * (len(self.H_path) + len(self.P_path))
            for i in self.H_path:
                sol_alpha[i] = round(m.solution.get_value("alpha_" + str(i)), 6)
        else:
            sol_alpha = [0]

        return sol_alpha

    # Prediction error
    def prediction_error(self):
        err_min = INF
        self.test = self.test.copy()

        for i in range(self.N_test):
            xi = self.test.iloc[i, self.x]
            yi = self.test.iloc[i, self.y]

            for l in self.H_path:
                nom = sum(np.array(self.c_p_alpha_matrix.iloc[l]["c*"]) * np.array(xi)) + self.c_p_alpha_matrix.iloc[l]["alpha**"]
                denom = sum(np.array(self.c_p_alpha_matrix.iloc[l]["p*"]) * np.array(yi))
                err = nom/denom
                if err < err_min:
                    err_min = err
            self.test.loc[i, "err_pred"] = err_min

    # Prediction
    def prediction(self, matrix):
        N = len(matrix)

        for i in range(N):
            err_min = INF

            xi = matrix.iloc[i, self.x].tolist()
            yi = matrix.iloc[i, self.y].tolist()

            for l in self.H_path_result:
                nom = sum(np.array(self.c_p_alpha_matrix_result.iloc[l]["c*"]) * np.array(xi)) + \
                      self.c_p_alpha_matrix_result.iloc[l]["alpha**"]
                denom = sum(np.array(self.c_p_alpha_matrix_result.iloc[l]["p*"]) * np.array(yi))
                err = nom / denom

                if err < err_min:
                    err_min = err

            #Score
            matrix.loc[i, "pDEA"] = err_min

            # Prediction
            for j in range(len(yi)):
                matrix.loc[i, "pred_" + self.yCol[j]] = err_min * yi[j]
        return matrix

    ######################################################## DEA #######################################################
    def _scoreDEA_BCC_output(self, x, y):
        # Prepare matrix
        self.xmatrix = self.matrix.iloc[:, self.x].T  # xmatrix
        self.ymatrix = self.matrix.iloc[:, self.y].T  # ymatrix

        # create one model instance, with a name
        m = Model(name='beta_DEA')

        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        beta = {0: m.continuous_var(name="beta")}

        # Constrain 2.4
        name_lambda = {i: m.continuous_var(name="l_{0}".format(i)) for i in range(self.N)}

        # Constrain 2.3
        m.add_constraint(m.sum(name_lambda[n] for n in range(self.N)) == 1)  # sum(lambda) = 1

        # Constrain 2.1 y 2.2
        for i in range(self.m):
            # Constrain 2.1
            m.add_constraint(m.sum(self.xmatrix.iloc[i, j] * name_lambda[j] for j in range(self.N)) <= x[i])

        for i in range(self.s):
            # Constrain 2.2
            m.add_constraint(
                m.sum(self.ymatrix.iloc[i, j] * name_lambda[j] for j in range(self.N)) >= beta[0] * y[i])

        # objetive
        m.maximize(beta[0])

        # Model Information
        # m.print_information()

        m.solve()

        # Solución
        if m.solution == None:
            sol = 0
        else:
            sol = m.solution.objective_value
        return sol

    def DEA(self):
        nameCol = "DEA"
        matrix = self.matrix.copy()
        matrix.loc[:, nameCol] = 0

        for i in range(len(matrix)):
            matrix.loc[i, nameCol] = self._scoreDEA_BCC_output(self.matrix.iloc[i, self.x].to_list(),
                                                                    self.matrix.iloc[i, self.y].to_list())
        # Caso mono-output. Predicción
        if self.s == 1:
            # Predicción
            matrix.loc[:, "yDEA"] = matrix.loc[:, "DEA"] * matrix.loc[:, "y"]
        elif self.s > 1:
            # Create data frame
            for j in range(self.s):
                matrix.loc[:, "yDEA{0}".format(self.yCol[j])] = 0
            # Calculate predictions
            for j in range(self.s):
                matrix.loc[:, "yDEA{0}".format(self.yCol[j])] = matrix.loc[:, "DEA"] * matrix.loc[:, self.yCol[j]]

        return matrix

    ################################### ERROR with Theorical Frontier ##################################################
    def MSE_theoric(self, matrixTheoric, matrix, dea_or_pillingdea):
        err = 0
        for i in range(len(matrix)):
            err += (matrixTheoric.loc[i, "yD"] - matrix.loc[i, dea_or_pillingdea]) ** 2
        return err / len(matrix)

    def MSE_theoric_multi(self, matrixTheoric, matrix, dea_or_pillingdea):
        err = 0
        for i in range(len(matrix)):
            for j in range(self.s):
                err += (matrixTheoric.iloc[i, self.y[j]] - matrix.loc[i, dea_or_pillingdea + str(self.yCol[j])]) ** 2
        return err / len(matrix)

    ################################### Print with Theorical Frontier ##################################################
    def print_Frontiers(self, matrix, yTheoric, ypillingDEA, yDEA, name_exp):
        #Unir datos
        matrix["yD"] = yTheoric
        matrix["yDEA"] = yDEA
        matrix["ypillingDEA"] = ypillingDEA

        # Ordenar "X" para que el gráfico no se confunda al dibujar
        datos = matrix.sort_values(by=["x1"])

        plt.figure()

        # ------------  Graphic Data ---------------------
        my_label = 'Data'
        plt.plot(datos['x1'], datos['y'], 'bo', color="b", markersize=5, label=my_label)

        # ------------  Graphic frontera Dios ---------------------
        my_label = 'True frontier'
        plt.plot(datos['x1'], datos['yD'], 'r--', label=my_label)  # Experimentos Monte Carlo

        # --------------- Graphic DEA ----------------------------
        my_label = "DEA"
        plt.plot(datos['x1'], datos["yDEA"], 'r', color="g", label=my_label)

        # --------------- Graphic uEAT ----------------------------
        my_label = "Cross-validated DEA"
        plt.plot(datos['x1'], datos["ypillingDEA"], 'r', color="c", label=my_label)

        # --------------- Graphic  ----------------------------
        # plt.title("Deep EAT")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc='upper left')
        #plt.show()
        plt.savefig(name_exp + '.png')




class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

class EXIT(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return style.YELLOW + "\n\n" + self.message + style.RESET
