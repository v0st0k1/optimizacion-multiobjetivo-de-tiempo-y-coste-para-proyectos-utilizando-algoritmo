from pandas import *
import matplotlib.pyplot as plt

from jmetal.core.quality_indicator import HyperVolume, InvertedGenerationalDistance, EpsilonIndicator

from is_pareto import devuelve_pareto_from_lists

#Lista de cadenas con los nombres de los archivos
f_algoritmos = []

#Cruce: uniforme
#Seleccion: Torneo binario
#Reemplazamiento: Crowding
#Prob. mutacion: 0.1
f_default01 = "nsgadefault_mr01.csv"
f_algoritmos.append(f_default01)
#Cruce: uniforme
#Seleccion: Torneo binario
#Reemplazamiento: Crowding
#Prob. mutacion: 0.2
f_default02 = "nsgadefault_mr02.csv"
f_algoritmos.append(f_default02)
#Cruce: uniforme
#Seleccion: Rank selection
#Reemplazamiento: crowding replacement
#Prob. mutacion: 0.1
f_rankselection = "rankselection.csv"
f_algoritmos.append(f_rankselection)
#Cruce: uniforme
#Seleccion: Torneo binario
#Reemplazamiento: NSGA replacement
#Prob. mutacion: 0.1
f_nsga_rep = "nsga_rep.csv"
f_algoritmos.append(f_nsga_rep)
#Cruce: uniforme
#Seleccion: Rank selection
#Reemplazamiento: NSGA replacement
#Prob. mutacion: 0.1
f_nsga_rep_rank = "nsga_rep_rank.csv"
f_algoritmos.append(f_nsga_rep_rank)
#Cruce: 1 punto
#Seleccion: Torneo binario
#Reemplazamiento: Crowding
#Prob. mutacion: 0.1
f_crossover_1_point = "crossover_1_point.csv"
f_algoritmos.append(f_crossover_1_point)
#Cruce: 2 punto
#Seleccion: Torneo binario
#Reemplazamiento: Crowding
#Prob. mutacion: 0.1
f_crossover_2_point = "crossover_2_point.csv"
f_algoritmos.append(f_crossover_2_point)
#Cruce: 1 punto
#Seleccion: Rank selection
#Reemplazamiento: Crowding
#Prob. mutacion: 0.1
f_crossover_1_point_rank_selection = "crossover_1_point_rank_selection.csv"
f_algoritmos.append(f_crossover_1_point_rank_selection)
#Cruce: 1 punto
#Seleccion: Torneo binario
#Reemplazamiento: NSGA replacement
#Prob. mutacion: 0.1
f_crossover_1_point_nsgarep = "crossover_1_point_nsgarep.csv"
f_algoritmos.append(f_crossover_1_point_nsgarep)
#Cruce: 1 punto
#Seleccion: Rank selection
#Reemplazamiento: NSGA replacement
#Prob. mutacion: 0.1
f_crossover_1_point_nsgarep_rank_selection = "crossover_1_point_nsgarep_rank_selection.csv"
f_algoritmos.append(f_crossover_1_point_nsgarep_rank_selection)


#Directorio problemas
problema_1 = "j3057_9"
problema_2 = "j3064_10"
problema_3 = "m561_10"
problema_4 = "n356_1"
problema_5 = "r557_10"
problemas = []
problemas.append(problema_1)
problemas.append(problema_2)
problemas.append(problema_3)
problemas.append(problema_4)
problemas.append(problema_5)

#Directorio por tipo de operador de mutacion
mut_actv = "mutacion actividades"
mut_modo = "mutacion modo"
op_mutacion = []
op_mutacion.append(mut_actv)
op_mutacion.append(mut_modo)

for problema in problemas:

    #Tomamos el pto. de refenrecia para HV y el pareto proximo al optimo de refenrecia para IGD y epsilon
    pto_referencia = read_csv(problema+'/'+'punto_referencia_hv_'+problema+'.csv').values.squeeze().tolist()
    pareto_referencia = read_csv(problema+'/pareto_ref_'+problema+'.csv').values.tolist()

    #Inicializamos las instancia para calcular las metricas
    hv = HyperVolume([pto_referencia[0]*(1+1/(1000-1)), pto_referencia[1]*(1+1/(1000-1))])
    hv.is_minimization = True

    igd = InvertedGenerationalDistance(pareto_referencia)

    ei = EpsilonIndicator(pareto_referencia)

    #Listas donde guardaremos las metricas
    hvbp = []
    igdbp = []
    epsilonbp = []

    for op in op_mutacion:
        for algoritmo in f_algoritmos:

            data = read_csv(problema+'/'+op+'/'+algoritmo)


            hvs = []
            igds = []
            epsilons = []
            for i in range(10):
                tiempo = data["tiempo"][i*100:i*100+99].tolist()
                coste = data["coste"][i*100:i*100+99].tolist()

                frente = devuelve_pareto_from_lists(tiempo, coste)

                hvs.append(hv.compute(frente))
                hv1 = hv.compute(frente)
                hv2 = hv.compute([[tiempo[i], coste[i]] for i in range(len(coste))])
                if hv1 != hv2:
                    print("Te cagas")

                igds.append(igd.compute(frente))
                epsilons.append(ei.compute(frente))

            hvbp.append(hvs)
            igdbp.append(igds)
            epsilonbp.append(epsilons)

    #Creamos los boxplots para este problema
    plt.boxplot(hvbp)
    plt.ylabel("Hipervolumen")
    plt.xlabel("Algoritmos")

    figure = plt.gcf()

    figure.set_size_inches(19.2, 9.77)
    plt.savefig("imgs_boxplots/"+problema+"_hipervolumen.png", dpi=100)
    plt.close()


    plt.boxplot(igdbp)
    plt.ylabel("IGD")
    plt.xlabel("Algoritmos")

    figure = plt.gcf()

    figure.set_size_inches(19.2, 9.77)

    plt.savefig("imgs_boxplots/"+problema+"_IGD.png", dpi=100)
    plt.close()


    plt.boxplot(epsilonbp)
    plt.ylabel("Indicador épsilon aditivo")
    plt.xlabel("Algoritmos")

    figure = plt.gcf()

    figure.set_size_inches(19.2, 9.77)

    plt.savefig("imgs_boxplots/"+problema+"_epsilon.png", dpi=100)
    plt.close()
