import numpy as np
import inspyred


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



def devuelve_pareto(candidatos):
    """ Recibe el conjunto de cromosomas y usando la funcion is_pareto
    devuelve el conjunto de los cromosomas que forma el frente de Pareto
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
