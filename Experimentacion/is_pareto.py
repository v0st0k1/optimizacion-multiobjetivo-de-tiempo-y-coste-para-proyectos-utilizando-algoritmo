import numpy as np
import inspyred
from pandas import *


def is_pareto(sols):
    """Devuelve una mascara con el frente de Pareto a partir de las soluciones
    en un array de tuplas
    """

    pareto = np.arange(sols.shape[0])
    n_sols = sols.shape[0]
    siguiente = 0

    while siguiente < len(sols):

        sols_m = sols < sols[siguiente]
        no_dominadas = np.any(sols_m, axis=1)

        no_dominadas[siguiente] = True
        pareto = pareto[no_dominadas]
        sols = sols[no_dominadas]
        siguiente = np.sum(no_dominadas[:siguiente]) + 1


    pareto_mask = np.zeros(n_sols, dtype = bool)
    pareto_mask[pareto] = True

    return pareto_mask


def devuelve_pareto_from_lists(tiempo, coste):
    """" Devuelve el conjunto de soluciones no-dominadas a partir de las listas
         tiempo y coste
    """

    sols = [(tiempo[i],coste[i]) for i in range(len(tiempo))]

    mask = is_pareto(np.array(sols))

    pareto_set = []
    i = 0
    for punto_pareto in mask.tolist():
        if punto_pareto:
            pareto_set.append([sols[i][0], sols[i][1]])
        i += 1

    return pareto_set



def devuelve_pareto(candidatos):
    """" Devuelve el conjunto de soluciones no-dominadas a partir de una lista
         de cromosomas
    """

    #Creamos lista de tuplas con los puntos y obtenemos una mascara de los que
    #son frente de pareto
    tupla_set = [(indiv.fitness[0], indiv.fitness[1]) for indiv in candidatos]
    mask = is_pareto(np.array(tupla_set))

    pareto_set = []

    i = 0
    for punto_pareto in mask.tolist():
        if punto_pareto:
            pareto_set.append(candidatos[i])
        i += 1

    return pareto_set


def devuelve_pareto_from_file(f_name_x, f_name_y):
    """" Devuelve el conjunto de soluciones no-dominadas a partir de dos
         archivos con soluciones
    """

    f_x = open(f_name_x, 'r')
    x_total = f_x.readline()

    x_total = x_total.replace('[','').replace(']','').replace(' ','')
    x_tiempo = x_total.split(',')

    for i, x in enumerate(x_tiempo):
        x_tiempo[i] = float(x)

    f_y = open(f_name_y, 'r')
    y_total = f_y.readline()

    y_total = y_total.replace('[','').replace(']','').replace(' ','')
    y_coste = y_total.split(',')

    for i, y in enumerate(y_coste):
        y_coste[i] = float(y)


    tupla_set = list(zip(x_tiempo, y_coste))

    mask = is_pareto(np.array(tupla_set))

    pareto_set = []

    i = 0
    for punto_pareto in mask.tolist():
        if punto_pareto:
            pareto_set.append(tupla_set[i])
        i += 1

    return pareto_set


def devuelve_pareto_from_csv(f_name):
    """" Devuelve el conjunto de soluciones no-dominadas a partir de un
         archivo en formato .csv con soluciones
    """

    csv = read_csv(f_name)

    x_tiempo = csv['tiempo'].tolist()
    y_coste = csv['coste'].tolist()

    for i, x in enumerate(x_tiempo):
        x_tiempo[i] = float(x)

    for i, y in enumerate(y_coste):
        y_coste[i] = float(y)

    tupla_set = list(zip(x_tiempo, y_coste))

    mask = is_pareto(np.array(tupla_set))

    pareto_set = []

    i = 0
    for punto_pareto in mask.tolist():
        if punto_pareto:
            pareto_set.append(tupla_set[i])
        i += 1

    return pareto_set

def devuelve_pareto_from_list_csv(list_fs):
    """" Devuelve el conjunto de soluciones no-dominadas a partir de una
         lista de archivos en formato .csv con soluciones
    """

    paretos = []
    for f_csv in list_fs:
        paretos = paretos + devuelve_pareto_from_csv(f_csv)

    mask = is_pareto(np.array(paretos))

    pareto_set = []

    i = 0
    for punto_pareto in mask.tolist():
        if punto_pareto:
            pareto_set.append(paretos[i])
        i += 1

    return pareto_set
