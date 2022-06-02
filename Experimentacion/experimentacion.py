from pandas import *
from is_pareto import devuelve_pareto_from_csv, is_pareto_efficient, devuelve_pareto_from_list_csv
import numpy as np
import csv

from jmetal.core.quality_indicator import HyperVolume, InvertedGenerationalDistance, EpsilonIndicator

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

    tiempo_M = -1
    coste_M = -1

    list_files = []
    for op in op_mutacion:

        for algoritmo in f_algoritmos:

            list_files.append(problema+"/"+op+"/"+algoritmo)

            #Obtencion de punto de referencia para HV
            csv_rr = read_csv(problema+"/"+op+"/"+algoritmo)
            x_tiempo = csv_rr['tiempo'].tolist()
            y_coste = csv_rr['coste'].tolist()

            x_tiempo_M = max(x_tiempo)
            y_coste_M = max(y_coste)

            if x_tiempo_M > tiempo_M:
                tiempo_M = x_tiempo_M
            if y_coste_M > coste_M:
                coste_M = y_coste_M


    #Escribimos pareto optimo aproximado de referencia
    pareto_ref = devuelve_pareto_from_list_csv(list_files)

    with open(problema+"/"+"pareto_ref_"+problema+".csv", mode='w') as f_csv:
        csv_writer = csv.writer(f_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer.writerow(["tiempo", "coste"])

        for pto in pareto_ref:
            csv_writer.writerow([pto[0], pto[1]])

    #Escribimos el punto de refenrecia para el HV
    with open(problema+"/"+"punto_referencia_hv_"+problema+".csv", mode='w') as f_csv:
        csv_writer = csv.writer(f_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer.writerow(["tiempo", "coste"])

        csv_writer.writerow([tiempo_M, coste_M])

    #Numero de soluciones (cardinalidad) que forman el conjunto de pareto optimo aproximado
    with open(problema+"/"+"cardinalidad_pareto_aprox_"+problema+".csv", mode='w') as f_csv:
        csv_writer = csv.writer(f_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer.writerow(["Cardinalidad del conjunto de pareto optimo aproximado"])

        csv_writer.writerow([len(pareto_ref)])

#Calculo de los indicadores de calidad
for problema in problemas:
    csv_pr = read_csv(problema+"/"+"pareto_ref_"+problema+".csv")
    csv_rr = read_csv(problema+"/"+"punto_referencia_hv_"+problema+".csv")

    x_pr = csv_pr["tiempo"]
    y_pr = csv_pr["coste"]

    pareto_referencia = [[x_pr[i], y_pr[i]] for i in range(len(x_pr))]
    pto_referencia = [float(csv_rr["tiempo"]), float(csv_rr["coste"])]

    #Declaramos los indicadores
    hv = HyperVolume([pto_referencia[0]*(1+1/(1000-1)), pto_referencia[1]*(1+1/(1000-1))])
    hv.is_minimization = True

    igd = InvertedGenerationalDistance(pareto_referencia)

    ei = EpsilonIndicator(pareto_referencia)

    with open(problema+"/"+"metricas_calidad_"+problema+".csv", mode='w') as f_csv:
        csv_writer = csv.writer(f_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Algoritmo (operador de mutacion)", "Hipervolumen", "IGD", "Epsilon", "Cardinalidad del cto. de soluciones no-dominadas"])

        for op in op_mutacion:
            for algoritmo in f_algoritmos:

                frente_raw = devuelve_pareto_from_csv(problema+"/"+op+"/"+algoritmo)
                frente = [[pfr[0], pfr[1]] for pfr in frente_raw]

                #Hipervolumen
                hv_calculado = hv.compute(frente)
                #IGD
                igd_calculado = igd.compute(frente)
                #epsilon aditivo
                ei_calculado = ei.compute(frente)

                #Redondeado hv en 2 decimales e IGD y epsilon en 4
                csv_writer.writerow([problema+" ("+op+")", round(hv_calculado,2), round(igd_calculado,4), round(ei_calculado,4), len(frente)])
