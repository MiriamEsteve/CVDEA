import numpy as np
import pandas as pd
from math import e
from scipy.stats import truncnorm


#sd = semilla, N = tam.muestra, nX = num. X, nY = num. Y, border = % frontera, noise = ruido {0/1}
class Data2:
    def __init__(self, sd, N, frontier, noise):
        self.sd = sd
        self.N = N
        self.nX = 2
        self.nY = 2
        self.frontier = frontier
        self.noise = noise

        # Seed random
        np.random.seed(self.sd)

        # DataFrame vacio
        self.data = pd.DataFrame()

        # P1 (Generar de forma aleatoria x1, x2 y z)
        self._generate_X_Z()


    def _generate_X_Z(self):
        # Generar nX
        for x in range(self.nX):
            # Generar X's
            self.data["x" + str(x + 1)] = np.random.uniform(5, 50, self.N)

        # Generar z
        z = np.random.uniform(-1.5, 1.5, self.N)

        # Generar cabeceras nY
        for y in range(self.nY):
            self.data["y" + str(y + 1)] = None

        # Ln de x1 y x2
        ln_x1 = np.log(self.data["x1"])
        ln_x2 = np.log(self.data["x2"])

        # Operaciones para ln_y1_ast
        op1 = -1 + 0.5 * z + 0.25 * (z ** 2) - 1.5 * ln_x1
        op2 = -0.6 * ln_x2 + 0.2 * (ln_x1 ** 2) + 0.05 * (ln_x2 ** 2) - 0.1 * ln_x1 * ln_x2
        op3 = 0.05 * ln_x1 * z - 0.05 * ln_x2 * z
        ln_y1_ast = -(op1 + op2 + op3)

        # Y de ese valor determinamos y1*=exp(ln(y1*))
        self.data["y1"] = np.exp(ln_y1_ast)

        # P3(Calculamos ln(y2*) como z + ln(y1*). Del ln(y2*), sacamos y2* = exp(ln(y2*))
        self.data["y2"] = np.exp(ln_y1_ast + z)

        # Jugaremos con generar un cierto porcentaje de DMUs que estarán en la frontera, ES DECIR, nos quedaremos en el paso 3 y no seguiremos para generar esos datos. Bastará con y1* y y2*,
        # Si %frontera == 0 no hacer nada
        if self.frontier > 0:
            index = self.data.sample(frac=self.frontier, random_state=self.sd).index
            # Tam. data_sample
            half_normal = np.exp(abs(np.random.normal(0, (0.3 ** (1 / 2)), len(index))))  # Half-normal

            # P5(Calculamos y1(2) y y2(2))
            if self.noise:
                normal1 = np.exp(np.random.normal(0, (0.01 ** (1 / 2)), len(index)))
                normal2 = np.exp(np.random.normal(0, (0.01 ** (1 / 2)), len(index)))

                self.data.loc[index, "y1"] /= (half_normal * normal1)
                self.data.loc[index, "y2"] /= (half_normal * normal2)

            # P4(Calculamos y1(1) y y2(1). Son outputs sin ruido aleatorio (sólo con ineficiencias))
            else:
                self.data.loc[index, "y1"] /= half_normal
                self.data.loc[index, "y2"] /= half_normal

    def _fi_Theoric(self, x, y):
        # ---------------------- z = ln(y2, y1) ------------------------------------
        z = np.log(y[1] / y[0])

        # -------------- Pasos 2 y 3 para obtener y1*, y2* -------------------------
        # Ln de x1 y x2
        ln_x1 = np.log(x[0])
        ln_x2 = np.log(x[1])
        # Operaciones para ln_y1_ast
        op1 = -1 + 0.5 * z + 0.25 * (z ** 2) - 1.5 * ln_x1
        op2 = -0.6 * ln_x2 + 0.2 * (ln_x1 ** 2) + 0.05 * (ln_x2 ** 2) - 0.1 * ln_x1 * ln_x2
        op3 = 0.05 * ln_x1 * z - 0.05 * ln_x2 * z
        ln_y1_ast = -(op1 + op2 + op3)

        # Y de ese valor determinamos y1*=exp(ln(y1*))
        y1_ast = np.exp(ln_y1_ast)
        # P3(Calculamos ln(y2*) como z + ln(y1*). Del ln(y2*), sacamos y2* = exp(ln(y2*))
        # y2_ast = np.exp(ln_y1_ast + z)

        # ------------------ Obtener fi --------------------------------------------
        fi_y1 = y1_ast / y[0]
        # fi_y2 = y2_ast / y[1]

        return fi_y1

    def fit_Theoric(self, x, y):
        nameCol = "yD"
        self.data.loc[:, nameCol] = 0

        for i in range(self.N):
            self.data.loc[i, nameCol] = self._fi_Theoric(self.data.loc[i, x].to_list(), self.data.loc[i, y].to_list())



class Data:
    def __init__(self, sd, N, nX):
        self.sd = sd
        # Seed random
        np.random.seed(self.sd)

        self.N = N
        self.nX = nX

        # DataFrame vacio
        self.data = pd.DataFrame()

        # Generate nX
        for x in range(self.nX):
            self.data["x" + str(x + 1)] = np.random.uniform(0, 1, self.N)

        self.u = np.random.exponential(1 / 3, self.N)

        if self.nX == 1:
            self.generate_1()
        elif self.nX == 2:
            self.generate_2()
        elif self.nX == 3:
            self.generate_3()
        elif self.nX == 4:
            self.generate_4()
        elif self.nX == 5:
            self.generate_5()
        elif self.nX == 6:
            self.generate_6()
        elif self.nX == 9:
            self.generate_9()
        elif self.nX == 12:
            self.generate_12()
        elif self.nX == 15:
            self.generate_15()
        else:
            print("Error. Input size")

    def generate_1(self):
        y = (self.data["x1"] ** 0.6)
        self.data["y"] = y * e ** -self.u
        self.data["yD"] = y

    def generate_2(self):
        y = (self.data["x1"] ** 0.4) * (self.data["x2"] ** 0.1)
        self.data["y"] = y * e ** -self.u
        self.data["yD"] = y

    def generate_3(self):
        y = (self.data["x1"] ** 0.3) * (self.data["x2"] ** 0.1) * (self.data["x3"] ** 0.1)
        self.data["y"] = y * e ** -self.u
        self.data["yD"] = y

    def generate_4(self):
        y = (self.data["x1"] ** 0.3) * (self.data["x2"] ** 0.1) * (self.data["x3"] ** 0.08) * (self.data["x4"] ** 0.02)
        self.data["y"] = y * e ** -self.u
        self.data["yD"] = y

    def generate_5(self):
        y = (self.data["x1"] ** 0.3) * (self.data["x2"] ** 0.1) * (self.data["x3"] ** 0.08) * (self.data["x4"] ** 0.1) \
            * (self.data["x5"] ** 0.1)
        self.data["y"] = y * e ** -self.u
        self.data["yD"] = y

    def generate_6(self):
        y = (self.data["x1"] ** 0.3) * (self.data["x2"] ** 0.1) * (self.data["x3"] ** 0.08) \
            * (self.data["x4"] ** 0.01) * (self.data["x5"] ** 0.006) * (self.data["x6"] ** 0.004)
        self.data["y"] = y * e ** -self.u
        self.data["yD"] = y

    def generate_9(self):
        y = (self.data["x1"] ** 0.3) * (self.data["x2"] ** 0.1) * (self.data["x3"] ** 0.08) \
            * (self.data["x4"] ** 0.005) * (self.data["x5"] ** 0.004) * (self.data["x6"] ** 0.001) \
            * (self.data["x7"] ** 0.005) * (self.data["x8"] ** 0.004) * (self.data["x9"] ** 0.001)
        self.data["y"] = y * e ** -self.u
        self.data["yD"] = y

    def generate_12(self):
        y = (self.data["x1"] ** 0.2) * (self.data["x2"] ** 0.075) * (self.data["x3"] ** 0.025) \
            * (self.data["x4"] ** 0.05) * (self.data["x5"] ** 0.05) * (self.data["x6"] ** 0.08) \
            * (self.data["x7"] ** 0.005) * (self.data["x8"] ** 0.004) * (self.data["x9"] ** 0.001) \
            * (self.data["x10"] ** 0.005) * (self.data["x11"] ** 0.004) * (self.data["x12"] ** 0.001)
        self.data["y"] = y * e ** -self.u
        self.data["yD"] = y

    def generate_15(self):
        y = (self.data["x1"] ** 0.15) * (self.data["x2"] ** 0.025) * (self.data["x3"] ** 0.025) \
            * (self.data["x4"] ** 0.05) * (self.data["x5"] ** 0.025) * (self.data["x6"] ** 0.025) \
            * (self.data["x7"] ** 0.05) * (self.data["x8"] ** 0.05) * (self.data["x9"] ** 0.08) \
            * (self.data["x10"] ** 0.005) * (self.data["x11"] ** 0.004) * (self.data["x12"] ** 0.001) \
            * (self.data["x13"] ** 0.005) * (self.data["x14"] ** 0.004) * (self.data["x15"] ** 0.001)
        self.data["y"] = y * e ** -self.u
        self.data["yD"] = y

"""
Parameters
    ----------
    seed : int
        Semilla para generar los datos
    n_inp : int
        Número de inputs
    rts : string
        Valor que indica e tipo de retorno a escala:
            IRS: sumatorio de las betas > 1 -> 1.2
            CRS: sumatorio de las betas = 1 -> 1
            DRS: sumatorio de las betas < 1 -> 0.8
    p : float
        Valor que indica el tipo de ruido -> (0, 0.5, 1, 2)
    ine : int
        Valor que indica el porcentaje de ineficiencia:
            90% -> N^+(0, 0.136^2)
            80% -> N^+(0, 0.299^2)
            70% -> N^+(0, 0.488^2)
    size : int
        Tamaño del dataframe
    Returns
    -------
    data_simulation : DataFrame
        Devuelve un conjunto de datos para simular
        
    data_dios : DataFrame
    """
class Data3:
    def __init__(self, seed, n_inp, rts, p, ine, size):
        self.seed = seed
        # Seed random
        np.random.seed(self.seed)

        self.n_inp = n_inp
        self.rts = rts
        self.p = p
        self.ine = ine
        self.size = size

        # DataFrame vacio
        self.data = pd.DataFrame()

        # Generar inputs
        for x in range(self.n_inp):
            # Generar X's
            self.data["x" + str(x + 1)] = np.random.uniform(5, 15, self.size)

        # Generar outputs
        for y in range(self.n_inp):
            # Generar Y's
            self.data["y" + str(y + 1)] = 0

        # Generar betas
        if (self.rts == "IRS"):
            beta = 1.2 / self.n_inp
        elif (self.rts == "CRS"):
            beta = 1 / self.n_inp
        elif (self.rts == "DRS"):
            beta = 0.8 / self.n_inp
        else:
            print("ERROR TIPO RTS")

        y_j_ast = list()
        # Generamos ||(y_j)^*||_2 = e^0*(x_1)^b0 * (x_2)^b1 * ...
        for j in range(self.size):
            y = 1
            for i in range(self.n_inp):
                y *= self.data.iloc[j, i] ** beta
            y_j_ast.append(y)

        # Generamos alpha_i_j -> i = observacion j = input
        alpha = [[0 for i in range(self.n_inp)] for j in range(self.size)]
        for i in range(self.size):
            for j in range(self.n_inp):
                alpha[i][j] = truncnorm.rvs(1 / self.n_inp ** 2, 1 / self.n_inp, size=1)[0]

        # Generamos a e y_j en la frontera
        a = [[0 for i in range(self.n_inp)] for j in range(self.size)]
        for j in range(self.size):
            sumat = sum(alpha[j])
            for i in range(self.n_inp):
                a[j][i] = alpha[j][i] / sumat
                self.data.iloc[j, i + self.n_inp] = a[j][i] ** (1 / 2) * y_j_ast[j]
        self.data_dios = self.data.copy()

        if (p != 0):
            # Generar ruido
            v_j = np.random.uniform(0, (p * 0.136) ** 2, self.size)

        else:
            v_j = np.random.uniform(0, 0, self.size)
        # Generar ineficiencia
        if (self.ine == 90):
            desv = 0.136 ** 2
        elif (self.ine == 80):
            desv = 0.299 ** 2
        elif (self.ine == 70):
            desv = 0.488 ** 2
        else:
            desv = 0
        mu_j = abs(np.random.uniform(0, desv, self.size))

        # Añadir el ruido y la ineficiencia
        # Calculamos la norma 2 de y_j
        y_j = list()
        for j in range(self.size):
            y = y_j_ast[j] * np.exp(-mu_j[j]) * np.exp(v_j[j])
            y_j.append(y)

        # Generamos y_j con ruido e ineficiencias
        for j in range(self.size):
            for i in range(self.n_inp):
                self.data.iloc[j, i + self.n_inp] = a[j][i] ** (1 / 2) * y_j[j]

