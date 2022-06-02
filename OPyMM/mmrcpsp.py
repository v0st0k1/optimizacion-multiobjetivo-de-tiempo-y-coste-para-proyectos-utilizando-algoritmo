from random import Random
import time
import inspyred
import numpy as np
import matplotlib.pyplot as plt

from pymoo.factory import get_performance_indicator

from is_pareto import devuelve_pareto

#Pruebas
import sys, os

rand = Random()


class CromIterator:

    def __init__(self, cromosoma):
        self.cromosoma = cromosoma
        self.indice = 0

    def __next__(self):
        if self.indice < len(self.cromosoma.I):
            resultado = Cromosoma(self.cromosoma.I[self.indice], self.cromosoma.m[self.indice])
            self.indice += 1
            return resultado
        raise StopIteration

class Cromosoma:
    """ Modela la estructura que compete al cromosoma de nuestro AG,
        compuesto y sirviendo tambien como definicion del metodo inicializador.
            - Lista de claves aleatorias: usado para ver que actividad realizar a partir de las posibles, estadistica
            - Lista de modos: emparejado posicionalmente con la clave aleatoria, sirve para especificar el  modo de una actividad
    """

    def __init__(self, random_keys, modes):
        self.I = random_keys
        self.m = modes

    def __iter__(self):
        return CromIterator(self)

    def __getitem__(self, indice):
        return Cromosoma(self.I[indice], self.m[indice])

    def __setitem__(self, indice, valor):
        self.I[indice] = valor.I
        self.m[indice] = valor.m

    def __len__(self):
        return len(self.I)


def busca_posibles(realizadas, ejecutandose, lim_recursos, predecesores):
    """ Busca las posibles, no contempla las ya realizadas y en_ejecucion
        Devuelve lista con las actividades que se pueden realizar
        Solo contempla las restinciones temporales
    """

    posibles = []

    for i in range(len(predecesores)):
        sin_preced_completos = np.setdiff1d(predecesores[i], realizadas)
        if sin_preced_completos.size == 0 and not i in realizadas and not i in ejecutandose:
            posibles.append(i)

    return posibles

def selecciona_actividad(posibles, I):
    """ Posible te dice los indices (actv) que puedes coger, e I para ordenar
    """

    I_posibles = np.array(I)[posibles]

    #Guardamos el indice donde es mayor para las posibles, el indice real lo dara posibles
    indx_max = np.argmax(I_posibles)
    return posibles[indx_max]

def es_factible(actividad, modo, recursos_en_uso, recursos_modo, lim_recursos):
    """ Comprueba si insertando actividad,modo en conjunto solucion da una solucion posible
    """

    factible = False

    #Si no se esta ejecutando nada, se puede meter porque no existe modos inejecutables, si no, se comprueba si cabe
    if recursos_en_uso == []:
        factible = True
    else:
        #Se crea matriz con recursos en columna, se suma las filas y se comprueba que ninguno exceda
        peticion = recursos_en_uso + [recursos_modo[actividad][modo]]
        suma = np.sum(peticion, axis = 0)
        factible = not np.any(suma > lim_recursos)


    return factible


def inserta_en_ejecucion(en_ejecucion, actividad, modo, recursos_modo, dias_modo):
    """ Inserta una actividad-modo en la estructura para su ejecucion
    """

    #Se añade actividad a la lista
    en_ejecucion["actividades"] = en_ejecucion["actividades"] + [actividad]

    en_ejecucion["recursos_en_uso"] = en_ejecucion["recursos_en_uso"] + [recursos_modo[actividad][modo]]

    en_ejecucion["dias_restantes"] = en_ejecucion["dias_restantes"] + [dias_modo[actividad][modo]]

    return en_ejecucion


def actualiza_dia(realizadas, en_ejecucion):
    """Actualiza un dia, o ud. de tiempo, en la estructura de ejecucion
    """

    end = len(en_ejecucion["dias_restantes"])
    i = 0
    while i < end:
        en_ejecucion["dias_restantes"][i] -= 1

        if en_ejecucion["dias_restantes"][i] == 0:
            realizadas = realizadas + [en_ejecucion["actividades"][i]]
            del en_ejecucion["dias_restantes"][i]
            del en_ejecucion["actividades"][i]
            del en_ejecucion["recursos_en_uso"][i]
            end -= 1 #Como hemos eliminado uno habra que iterar hasta una posicion menos
        else:
            i += 1
    return [realizadas, en_ejecucion]



def decodifica(cromosoma, predecesores, lim_recursos, recursos_modo, dias_modo):
    """Funcion que nos decodifica un cromosoma para obtener el fenotipo
    """

    #Inicializacion
    en_ejecucion = {"actividades" : [], "recursos_en_uso" : [] , "dias_restantes" : []}
    realizadas = []
    dia_actual = 0
    solucion = [] #Lista de tuplas formato <actividad,dia_comienzo>

    #Mientras no se haya terminado todas
    while len(realizadas) < len(cromosoma.I):
        #Se busca las posibles, luego,  se recorren todas las posibles
        posibles = busca_posibles(realizadas, en_ejecucion["actividades"], lim_recursos, predecesores)

        while not len(posibles) == 0:
            actividad = selecciona_actividad(posibles, cromosoma.I)
            posibles.remove(actividad)

            #Se elige actv. con mas probabilidad dentro de las posibles
            if es_factible(actividad, cromosoma.m[actividad], en_ejecucion["recursos_en_uso"], recursos_modo, lim_recursos):
                #Si es factible, se agrega a solucion con el dia de comienzo y se inserta en tabla en_ejecucion
                solucion.append((actividad, dia_actual))
                en_ejecucion = inserta_en_ejecucion(en_ejecucion, actividad, cromosoma.m[actividad], recursos_modo, dias_modo)


        dia_actual += 1
        [realizadas, en_ejecucion] = actualiza_dia(realizadas, en_ejecucion)

    return solucion


class Problema(inspyred.benchmarks.Benchmark):
""" Clase que modela nuestro problema MRCPSP

    Atributos:

        recursos_modo - gasto de recursos renovables por cada actividad-modo

        dias_modo - tiempo necesario de ejecucion de cada actividad-modo

        lim_recursos - limite de los recursos renovables

        Mn - cantidad de modos disponibles para cada actividad

        predecesores - lista de predecesores de cada actividad

    Atributos derivados:

        coste_modo - coste de cada actividad-modo

        modos_ban - lista de los modos prohibidos
"""

    def __init__(self, recursos_modo, dias_modo, lim_recursos, Mn, predecesores, coste_fijo, coste_rnr, rnr_modo, objetivos = 2):

        inspyred.benchmarks.Benchmark.__init__(self, len(Mn), objetivos)

        self.recursos_modo = recursos_modo
        self.dias_modo = dias_modo
        self.lim_recursos = lim_recursos
        self.Mn = Mn
        self.predecesores = predecesores


        #Calculamos el coste de cada modo según su gasto de recursos no renovables y su coste fijo
        coste_modo = np.array([np.array(i, dtype=object)*coste_rnr for i in np.array(rnr_modo, dtype=object)], dtype=object)
        coste_modo = [np.sum(i, axis=1).tolist() for i in coste_modo]

        for i in range(len(coste_modo)):
            coste_modo[i] = (np.array(coste_modo[i], dtype=object) + np.array(coste_fijo[i])).tolist()

        self.coste_modo = coste_modo

        #Modos prohibidos para obtener las planificaciones sin equivocaciones
        self.modos_ban = [[] for _ in range(len(Mn))]

        self.preprocesamiento()

        self.maximize = False



    def preprocesamiento(self):
        #Mientras se haga un cambio, o la primera vez, se hace
        Flag_changes = True

        while(Flag_changes):

            #Segundo resources-redudants
            f_err, self.recursos_modo, self.lim_recursos  = self.elimina_recursos_redundantes()

            #Tercero modos inefficient
            f_ei, self.recursos_modo, self.dias_modo, self.coste_modo = self.elimina_ineficientes()

            if(not f_err or not f_ei):
                Flag_changes = False


        #Corregimos la forma de las listas en caso de que el problema se haya convertido en monomodo
        try:
            len(self.dias_modo[0])
        except:
            self.dias_modo = [[dm] for dm in self.dias_modo]

        try:
            len(self.recursos_modo[0])
        except:
            self.recursos_modo = [[rm] for rm in self.recursos_modo]

        try:
            len(self.coste_modo[0])
        except:
            self.coste_modo = [[cm] for cm in self.coste_modo]
        #--------------------------------------------------------------------------------------------

        self.Mn = [len(self.dias_modo[i]) for i in range(len(self.dias_modo))]

        return 0


    def elimina_recursos_redundantes(self):
        """Sean recursos redundantes aquellos para los que ni realizando todas las actividades con su modo mas costoso para
           determinado recurso se vean agotados
        """

        #Tomamos el mayor gasto de cada modo para todos los recursos
        max_sum_resources = np.sum([np.max(np.array(self.recursos_modo[i]), 0) for i in range(len(self.recursos_modo))], 0)

        #Vemos aquellos recursos que no sean redundantes
        index_ok = max_sum_resources > np.array(self.lim_recursos)

        if not index_ok.all(): #Si ninguno es redundante no se cambia nada

            nuevo_recursos_modo = []

            for i in range(len(self.recursos_modo)):

                nuevo_recursos_modo.append([])

                for j in range(len(self.recursos_modo[i])):

                    nuevo_recursos_modo[i].append((np.array(self.recursos_modo[i][j])[index_ok]).tolist())

            nuevo_lim_recursos = np.array(self.lim_recursos)[index_ok].tolist()
            cambios = True

        else:
            nuevo_lim_recursos = self.lim_recursos
            nuevo_recursos_modo = self.recursos_modo
            cambios = False


        return cambios, nuevo_recursos_modo, nuevo_lim_recursos


    def elimina_ineficientes(self):
        """ Elimina los modos ineficientes, siendo estos aquellos que necesitan mayor cantidad de recursos,
            para todos los tipos de recurso, y además tienen mayor duración que otro modo para la misma
            actividad.
        """
        #Estructuras donde reconstruir
        new_recursos_modo = []
        new_dias_modo = []
        new_coste_modo = []

        cambios = False

        #Iteramos todas las actividades, y si hay mas de un modo de realizarla se saca el modo de mayor duracion
        #y se comprueba que no consuma mas recursos que el resto de modos para esa actividad
        for actividad, dias in enumerate(self.dias_modo):
            new_recursos_modo.append([])
            new_dias_modo.append([])
            new_coste_modo.append([])

            if len(dias) > 1: #Si hay mas de un modo por dia
                #Se coge el dia con mas duracion y se ve los recursos que gasta para compararlo con los demas
                modo_mas_largo = np.argmax(dias)

                recursos_mas_largo = self.recursos_modo[actividad][modo_mas_largo]

                ineficiente = True

                #Aqui se guarda el resto sin contar el modo de mas duracion, se hace uso de estrucutras array para comprobarlo
                #facilmente
                resto = np.delete(self.recursos_modo[actividad], modo_mas_largo, axis=0)
                ineficiente = not (resto > recursos_mas_largo).any()

                if ineficiente:
                    cambios = True

                    new_recursos_modo[actividad].append(resto.tolist())
                    new_dias_modo[actividad].append(np.delete(self.dias_modo[actividad], modo_mas_largo).tolist())
                    new_coste_modo[actividad].append(np.delete(self.coste_modo[actividad], modo_mas_largo).tolist())

                    #Introducimos el actividad-modo baneado en la lista

                    #Primero obtenemos el modo real que tratamos
                    m_real = modo_mas_largo


                    for mb in self.modos_ban[actividad]:
                        if m_real >= mb:
                            m_real += 1

                    #Y lo introducimos ordenadamente
                    mbj = self.modos_ban[actividad][:]
                    mbj.append(m_real)
                    mbj = sorted(mbj)

                    self.modos_ban[actividad] = mbj[:]

                else:
                    new_recursos_modo[actividad].append(self.recursos_modo[actividad])
                    new_dias_modo[actividad].append(self.dias_modo[actividad])
                    new_coste_modo[actividad].append(self.coste_modo[actividad])
            else:

                new_recursos_modo[actividad].append(self.recursos_modo[actividad])
                new_dias_modo[actividad].append(self.dias_modo[actividad])
                new_coste_modo[actividad].append(self.coste_modo[actividad])


        return cambios, np.squeeze(new_recursos_modo).tolist(), np.squeeze(new_dias_modo).tolist(), np.squeeze(new_coste_modo).tolist()


    def genera_candidato(self, random, args):
        """ Genera aleatoriamente cromosoma, compuestos por su random-key + modos
        """

        num_acts = len(self.recursos_modo)
        rk_acts = [random.random() for _ in range(num_acts)]
        rk_modos = [random.randint(0, self.Mn[i]-1) for i in range(num_acts)]

        return Cromosoma(rk_acts, rk_modos)


    def mutacion_modos(self, random, candidates, args):
        """ Operador de mutacion con cambio de modo
        """

        prob_mutacion = args.setdefault('mutation_rate', 0.1)

        #Para cada individuo
        for it, cromosoma in enumerate(candidates):

            #Se elige actividad aleatoria
            actv_i = random.randint(0, len(cromosoma.I) - 1)

            if random.random() < prob_mutacion and self.Mn[actv_i] > 1:

                #Lista con modos disponibles, excepto el actual, se escoge uno nuevo aleatorio
                modos_disponibles = list(range(self.Mn[actv_i]))

                modos_disponibles.remove(cromosoma.m[actv_i])

                new_modo = random.choice(modos_disponibles)
                new_m = cromosoma.m
                new_m[actv_i] = new_modo

                #Se crea el nuevo cromosoma con el nuevo modo
                new_cromosoma = Cromosoma(cromosoma.I, new_m)

                candidates[it] = new_cromosoma

        return candidates


    def mutacion_actividades(self, random, candidates, args):
        """ Operador de mutacion con permutaciones en las actividades
        """

        prob_mutacion = args.setdefault('mutation_rate', 0.1) #Con esto tenemos que probar!

        #Para cada individuo
        for it, cromosoma in enumerate(candidates):

            #Se eligen dos actividad aleatoria
            actv_i = random.randint(0, len(cromosoma.I) - 1)
            #otra aleatoria
            actv_j = random.randint(0, len(cromosoma.I) - 1)

            #que sean distintas
            while (actv_i == actv_j and len(cromosoma.I)>2):#por precacucion
                actv_j = random.randint(0, len(cromosoma.I) - 1)


            if random.random() < prob_mutacion:

                i_aux = cromosoma.I[actv_i]
                cromosoma.I[actv_i] = cromosoma.I[actv_j]
                cromosoma.I[actv_j] = i_aux

        return candidates


    def makespan(self, cromosoma):
        """Calcula el tiempo necesario para completar una planificación
        """
        sol = decodifica(cromosoma, self.predecesores, self.lim_recursos, self.recursos_modo, self.dias_modo) #Aki fallo

        dia_finalizacion = []

        for i, actv_dia in enumerate(sol):
            dia_finalizacion.append(actv_dia[1]+self.dias_modo[actv_dia[0]][cromosoma.m[actv_dia[0]]])

        makespan = max(dia_finalizacion)

        return makespan


    def calcula_coste(self, cromosoma):
        """Calcula el coste de una planificacion dada
        """
        coste = 0
        for actv, modo in enumerate (cromosoma.m):
            coste += self.coste_modo[actv][modo]
        return coste


    def evaluador(self, candidates, args):
        """Funcion de evaluacion multiobjetivo
        """
        fitness = []
        for cromosoma in candidates:

            obj1 = self.makespan(cromosoma)

            obj2 = self.calcula_coste(cromosoma)

            fitness.append(inspyred.ec.emo.Pareto([obj1, obj2]))

        return fitness




def resuelve(Mn, lim_recursos, recursos_modo, coste_rnr, rnr_modo, coste_fijo, tiempo_modo, predecesores, n_experimentos, pdefecto):
    """Función útil para poner en marcha los algoritmos pertinente en función del modo escogido.
       Devuelve el conjunto Pareto solucion al problema.
    """

    problem = Problema(recursos_modo, tiempo_modo, lim_recursos, Mn, predecesores, coste_fijo, coste_rnr, rnr_modo, objetivos = 2)


    """Estructura con algoritmos del 1 al 20
    # 0.- Op. de seleccion
    # 1.- Op. de cruce
    # 2.- Op. de mutacion
    # 3.- Prob. de mutacion
    # 4.- Op. de reemplazamiento
    # 5.- num. selected
    # 6.- num. crossover point"""
    algoritmos = [
        [inspyred.ec.selectors.tournament_selection,
        inspyred.ec.variators.uniform_crossover,
        problem.mutacion_actividades,
        0.1,
        inspyred.ec.replacers.crowding_replacement,
        1,
        0],

        [inspyred.ec.selectors.tournament_selection,
        inspyred.ec.variators.uniform_crossover,
        problem.mutacion_actividades,
        0.2,
        inspyred.ec.replacers.crowding_replacement,
        1,
        0],

        [inspyred.ec.selectors.rank_selection,
        inspyred.ec.variators.uniform_crossover,
        problem.mutacion_actividades,
        0.1,
        inspyred.ec.replacers.crowding_replacement,
        100,
        0],

        [inspyred.ec.selectors.tournament_selection,
        inspyred.ec.variators.uniform_crossover,
        problem.mutacion_actividades,
        0.1,
        inspyred.ec.replacers.nsga_replacement,
        1,
        0],

        [inspyred.ec.selectors.rank_selection,
        inspyred.ec.variators.uniform_crossover,
        problem.mutacion_actividades,
        0.1,
        inspyred.ec.replacers.nsga_replacement,
        100,
        0],

        [inspyred.ec.selectors.tournament_selection,
        inspyred.ec.variators.n_point_crossover,
        problem.mutacion_actividades,
        0.1,
        inspyred.ec.replacers.crowding_replacement,
        1,
        1],

        [inspyred.ec.selectors.tournament_selection,
        inspyred.ec.variators.n_point_crossover,
        problem.mutacion_actividades,
        0.1,
        inspyred.ec.replacers.crowding_replacement,
        1,
        2],

        [inspyred.ec.selectors.rank_selection,
        inspyred.ec.variators.n_point_crossover,
        problem.mutacion_actividades,
        0.1,
        inspyred.ec.replacers.crowding_replacement,
        100,
        1],

        [inspyred.ec.selectors.tournament_selection,
        inspyred.ec.variators.n_point_crossover,
        problem.mutacion_actividades,
        0.1,
        inspyred.ec.replacers.nsga_replacement,
        1,
        1],

        [inspyred.ec.selectors.rank_selection,
        inspyred.ec.variators.n_point_crossover,
        problem.mutacion_actividades,
        0.1,
        inspyred.ec.replacers.nsga_replacement,
        100,
        1],

        [inspyred.ec.selectors.tournament_selection,
        inspyred.ec.variators.uniform_crossover,
        problem.mutacion_modos,
        0.1,
        inspyred.ec.replacers.crowding_replacement,
        1,
        0],

        [inspyred.ec.selectors.tournament_selection,
        inspyred.ec.variators.uniform_crossover,
        problem.mutacion_modos,
        0.2,
        inspyred.ec.replacers.crowding_replacement,
        1,
        0],

        [inspyred.ec.selectors.rank_selection,
        inspyred.ec.variators.uniform_crossover,
        problem.mutacion_modos,
        0.1,
        inspyred.ec.replacers.crowding_replacement,
        100,
        0],

        [inspyred.ec.selectors.tournament_selection,
        inspyred.ec.variators.uniform_crossover,
        problem.mutacion_modos,
        0.1,
        inspyred.ec.replacers.nsga_replacement,
        1,
        0],

        [inspyred.ec.selectors.rank_selection,
        inspyred.ec.variators.uniform_crossover,
        problem.mutacion_modos,
        0.1,
        inspyred.ec.replacers.nsga_replacement,
        100,
        0],

        [inspyred.ec.selectors.tournament_selection,
        inspyred.ec.variators.n_point_crossover,
        problem.mutacion_modos,
        0.1,
        inspyred.ec.replacers.crowding_replacement,
        1,
        1],

        [inspyred.ec.selectors.tournament_selection,
        inspyred.ec.variators.n_point_crossover,
        problem.mutacion_modos,
        0.1,
        inspyred.ec.replacers.crowding_replacement,
        1,
        2],

        [inspyred.ec.selectors.rank_selection,
        inspyred.ec.variators.n_point_crossover,
        problem.mutacion_modos,
        0.1,
        inspyred.ec.replacers.crowding_replacement,
        100,
        1],

        [inspyred.ec.selectors.tournament_selection,
        inspyred.ec.variators.n_point_crossover,
        problem.mutacion_modos,
        0.1,
        inspyred.ec.replacers.nsga_replacement,
        1,
        1],

        [inspyred.ec.selectors.rank_selection,
        inspyred.ec.variators.n_point_crossover,
        problem.mutacion_modos,
        0.1,
        inspyred.ec.replacers.nsga_replacement,
        100,
        1]
    ]

    try:
        soluciones = []

        #Ejecucion por defecto: algoritmos 11, 12, 13, 19, 2 y 3
        if pdefecto:
            algs_pdf = [1,2,10,11,12,18]

            for _, alg in enumerate(algs_pdf):

                for i in range(int(n_experimentos)):

                    print("Algoritmo : ", alg+1)
                    print("Experimento : ", i)
                    ag = inspyred.ec.emo.NSGA2(rand)
                    ag.terminator = inspyred.ec.terminators.generation_termination
                    ag.selector = algoritmos[alg][0]
                    ag.variator = [algoritmos[alg][1], algoritmos[alg][2]]
                    ag.replacement = algoritmos[alg][4]

                    final_pop = ag.evolve(generator = problem.genera_candidato,
                                                evaluator = problem.evaluador,
                                                pop_size = 100,
                                                maximize = False,
                                                num_selected = algoritmos[alg][5],
                                                mutation_rate = algoritmos[alg][3],
                                                max_evaluations = 100,
                                                num_crossover_points = algoritmos[alg][6],
                                                max_generations = 100
                                                )

                    soluciones = soluciones + final_pop
                    print("Experimento ",i," concluido")

        else:

            for alg in range(20):

                for i in range(int(n_experimentos)):

                    print("Algoritmo : ", alg+1)
                    print("Experimento : ", i)
                    ag = inspyred.ec.emo.NSGA2(rand)
                    ag.terminator = inspyred.ec.terminators.generation_termination
                    ag.selector = algoritmos[alg][0]
                    ag.variator = [algoritmos[alg][1], algoritmos[alg][2]]
                    ag.replacement = algoritmos[alg][4]

                    final_pop = ag.evolve(generator = problem.genera_candidato,
                                                evaluator = problem.evaluador,
                                                pop_size = 100,
                                                maximize = False,
                                                num_selected = algoritmos[alg][5],
                                                mutation_rate = algoritmos[alg][3],
                                                max_evaluations = 100,
                                                num_crossover_points = algoritmos[alg][6],
                                                max_generations = 100
                                                )

                    soluciones = soluciones + final_pop
                    print("Experimento ",i," concluido")



        pareto_set = devuelve_pareto(soluciones)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    return pareto_set, problem.modos_ban
