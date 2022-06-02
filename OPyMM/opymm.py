import PySimpleGUI as sg
import re
from PIL import Image, ImageTk, ImageSequence

from multiprocessing.pool import ThreadPool
from multiprocessing import Queue
import time

from mmrcpsp import Cromosoma, resuelve, decodifica

import matplotlib.pyplot as plt

from win32api import GetSystemMetrics

#Tema para la aplicación
sg.theme('DarkBlue')


def primera_pantalla():
    """Primera pantalla donde se introducira el numero de actividades, recursos renovables y no renovables.
       Luego se introducira el numero de modos de cada actividad, el limite de los recursos renovables
       y el coste unitario de los recursos no renovables
    """
    primer_layout =[
        [sg.Text('Creador de planificaciones óptimas', size=(30, 1), justification='center', font=("Helvetica", 25), relief=sg.RELIEF_RIDGE)],
        [sg.Text('Introduzca el número de actividades que conformarán el proyecto')],
        [sg.Input(key='-NUMACTV-', enable_events = True)],
        [sg.Text('Introduzca el número de recursos no renovables')],
        [sg.Input(key='-NUMRNR-', enable_events = True)],
        [sg.Text('Introduzca el número de recursos renovables')],
        [sg.Input(key='-NUMRR-', enable_events = True)],
        [sg.Button('Establecer', key='-CONTINUAR-')],
        [sg.Text('Debe introducir los datos correctamente. Comprobar que no sean 0 y que tampoco haya menos de 2 actividades', key='-VACIOCHECK-', text_color = 'red', visible = False)],
        [sg.Text('Número de actividades debe ser mayor a 1', key='-MONOACTV-', text_color = 'red', visible = False)],
        [sg.Button('Salir')]
    ]


    ancho = GetSystemMetrics(0)
    alto = GetSystemMetrics(1)

    window = sg.Window('Creador de planificaciones óptimas', primer_layout, resizable=True, finalize=True)

    #Flag para comprobar datos en segunda fase de la primera pantalla
    continuacion = False

    #Flags para continuar
    f_rnr = f_rr = f_mn = False
    preparado = False
    preparadisimo = 0

    N_rnr = 0
    N_rr = 0
    N_actv = 0

    while True:
        event, values = window.read()
        #Comprueba que es un numero entero y no nulo, en caso contrario lo borra
        if event == '-NUMACTV-' and values['-NUMACTV-'] and values['-NUMACTV-'][-1] not in ('0123456789'):
            window['-NUMACTV-'].update(values['-NUMACTV-'][:-1])

        #Comprueba que es un numero entero y no nulo, en caso contrario lo borra
        if event == '-NUMRNR-' and values['-NUMRNR-'] and values['-NUMRNR-'][-1] not in ('0123456789'):
            window['-NUMRNR-'].update(values['-NUMRNR-'][:-1])

        #Comprueba que es un numero entero y no nulo, en caso contrario lo borra
        if event == '-NUMRR-' and values['-NUMRR-'] and values['-NUMRR-'][-1] not in ('0123456789'):
            window['-NUMRR-'].update(values['-NUMRR-'][:-1])


        #Si cierra ventana o pulsar salir se cierra la aplicación
        if event in (sg.WIN_CLOSED, 'Salir'):
            break
        #Pulsamos continuar y se actualiza el numero de recursos renovables que introducir coste
        elif '-CONTINUAR-' in event:
            if values['-NUMACTV-'] == '' or values['-NUMRNR-'] == '' or values['-NUMRR-'] == '' or int(values['-NUMACTV-']) < 2 or int(values['-NUMRNR-']) == 0 or int(values['-NUMRR-']) == 0:
                window['-VACIOCHECK-'].update(visible = True)
            else:
                scroll_actv = int(values['-NUMACTV-']) > 20
                scroll_rr = int(values['-NUMRR-']) > 20
                scroll_rnr = int(values['-NUMRNR-']) > 20

                col_rnr = [[sg.Text("                    "),sg.Text("Coste unitario de los recursos no renovables")],*[[sg.Text("Recurso NR {}".format(i+1)),sg.Input(key = "Rnr_{}".format(i)),] for i in range(int(values['-NUMRNR-']))]]
                col_rr = [[sg.Text("                  "),sg.Text("Límite por unidad de tiempo de los recursos renovables")],*[[sg.Text("Recurso R {}".format(i+1)),sg.Input(key = "Rr_{}".format(i)),] for i in range(int(values['-NUMRR-']))]]
                col_actv = [[sg.Text("                 "),sg.Text("Número de modos de cada actividad")],*[[sg.Text("Actividad {}".format(i+1)), sg.Input(key = "Actv_{}".format(i)),] for i in range(int(values['-NUMACTV-']))]]

                primer_layout =[
                    [sg.Text('Creador de planificaciones óptimas', size=(30, 1), justification='center', font=("Helvetica", 25), relief=sg.RELIEF_RIDGE)],
                    [sg.Text('Introduzca el número de actividades que conformarán el proyecto')],
                    [sg.Input(key='-NUMACTV-', enable_events = True)],
                    [sg.Text('Introduzca el número de recursos no renovables')],
                    [sg.Input(key='-NUMRNR-', enable_events = True)],
                    [sg.Text('Introduzca el número de recursos renovables')],
                    [sg.Input(key='-NUMRR-', enable_events = True)],
                    [sg.Button('Establecer', key='-CONTINUAR-')],
                    [sg.Text('Debe introducir los datos o que estos no sean 0!', key='-VACIOCHECK-', text_color = 'red', visible = False)],
                    [sg.Column(col_rnr, scrollable = scroll_rnr, size = (ancho / 5, alto / 2)), sg.Column(col_rr, scrollable = scroll_rr, size = (ancho / 4, alto / 2)),
                    sg.Column(col_actv, scrollable = scroll_actv, size = (ancho / 5, alto / 2))],
                    [sg.Text('Error: comprueba que los datos están bien introducidos antes de continuar', key='-DATACHECKING-', text_color = 'red', visible = False)],
                    [sg.Text('Una vez pulsado el botón no podrá cambiar estos datos. Compruébalos y vuelva a pulsar.', key='-REVISE-', text_color = 'white', visible = False)],
                    [sg.Button('Siguiente', key = '-SIG-'), sg.Button('Salir')]
                ]

                N_rnr = int(values['-NUMRNR-'])
                N_rr = int(values['-NUMRR-'])
                N_actv = int(values['-NUMACTV-'])

                window1 = sg.Window('Creador de planificaciones óptimas', primer_layout, resizable=True, finalize=True)
                window.Close()
                window = window1

                continuacion = True



        elif continuacion and '-SIG-' in event:

            #Comprobacion de RNR
            coste_rnr = []
            for i in range(N_rnr):
                try:
                    if float(values["Rnr_{}".format(i)]) <= 0:
                        raise ValueError("El coste de los recursos no renovables no puede ser 0 o menor")
                    else:
                        coste_rnr.append(float(values["Rnr_{}".format(i)]))
                except:
                    window['-DATACHECKING-'].update(visible = True)
                    coste_rnr = []
                    break

            if coste_rnr != []:
                f_rnr = True
            else:
                f_rnr = False

            #Comrpobacion de RR
            lim_rr = []
            for i in range(N_rr):
                try:
                    if int(values["Rr_{}".format(i)]) <= 0:
                        raise ValueError("El limite de los recursos renovables no puede ser 0 o menor")
                    else:
                        lim_rr.append(int(values["Rr_{}".format(i)]))
                except:
                    window['-DATACHECKING-'].update(visible = True)
                    lim_rr = []
                    break

            if lim_rr != [] :
                f_rr = True
            else:
                f_rr = False

            #Comprobacion de modos actividades
            Mn = []
            for i in range(N_actv):
                try:
                    if int(values["Actv_{}".format(i)]) <= 0:
                        raise ValueError("El número de modos de una actividad no puede ser 0 o menor")
                    elif int(values["Actv_{}".format(i)]) > 9:
                        raise ValueError("El número de modos de una actividad debe ser menor de 10")
                    else:
                        Mn.append(int(values["Actv_{}".format(i)]))
                except:
                    window['-DATACHECKING-'].update(visible = True)
                    Mn = []
                    break

            if Mn != [] :
                f_mn = True
            else:
                f_mn = False

            if f_rnr and f_rr and f_mn :
                window['-REVISE-'].update(visible = True)
                window['-DATACHECKING-'].update(visible = False)
                preparado = True


        if f_rnr and f_rr and f_mn and preparado and '-SIG-' in event:

            preparadisimo += 1

        if f_rnr and f_rr and f_mn and preparado and preparadisimo == 2 and '-SIG-' in event:
            window.Close()
            return N_actv, N_rnr, N_rr, coste_rnr, lim_rr, Mn


    window.Close()


def segunda_pantalla(N_actv, N_rnr, N_rr, coste_rnr, lim_rr, Mn, actividad_actual):
    """Segunda pantalla donde se introducira de forma iterativa la informacion pertinente a cada actividad-modo
       Esta informacion es: gasto de recursos, tiempo necesario y coste fijo
    """

    ancho = GetSystemMetrics(0)
    alto = GetSystemMetrics(1)

    segundo_layout =[
        [sg.Text('Creador de planificaciones óptimas', size=(30, 1), justification='center', font=("Helvetica", 25), relief=sg.RELIEF_RIDGE)],
        [sg.Text("Actividad {}".format(actividad_actual+1), size=(30, 1), justification='center', font=("Helvetica", 20), relief=sg.RELIEF_RIDGE)]
    ]

    for m in range(Mn[actividad_actual]):

        col_rnr = [[sg.Text("                    "),sg.Text("Gasto de recursos no renovables")],*[[sg.Text("Recurso NR {}".format(i+1)),sg.Input(key = "Uso_Rnr_{}{}".format(m,i)),] for i in range(N_rnr)]]
        col_rr = [[sg.Text("                 "),sg.Text("Gasto de recursos renovables")],*[[sg.Text("Recurso R {}".format(i+1)),sg.Input(key = "Uso_Rr_{}{}".format(m,i)),] for i in range(N_rr)]]


        col_tiempo = [[sg.Text("Tiempo necesario: ")],[sg.Input(key = "t_{}".format(m))]]
        col_coste_fijo = [[sg.Text("Coste fijo: ")],[sg.Input(key = "cf_{}".format(m))]]

        segundo_layout.append([sg.Text("Modo {}".format(m+1))])
        segundo_layout.append([sg.Column(col_rnr, scrollable = N_rnr > 3, size = (ancho / 6, alto / 10)),
        sg.Column(col_rr, scrollable = N_rr > 3, size = (ancho / 6, alto / 10)),
        sg.Column(col_tiempo, size = (ancho / 6, alto / 10)), sg.Column(col_coste_fijo, size = (ancho / 6, alto / 10))])

    segundo_layout.append([sg.Text('Comprueba que los datos están bien introducidos antes de continuar', key='-DATACHECKING2-', text_color = 'red', visible = False)])
    botones = [sg.Button('Siguiente', key='-SIG2-')]
    if actividad_actual != 0:
        botones.append(sg.Button('Atrás', key = '-ATRAS-'))
    botones.append(sg.Button('Salir'))

    segundo_layout.append(botones)

    window = sg.Window('Creador de planificaciones óptimas', segundo_layout, resizable=True, finalize=True)

    #variables de retorno
    rnr_j = []
    rr_j = []
    t_j = []
    cf_j = []
    preparadisimo = False

    while True:
        event, values = window.read()


        RNR = []
        RR = []

        #Si cierra ventana o pulsar salir se cierra la aplicación
        if event in (sg.WIN_CLOSED, 'Salir'):
            break


        elif '-SIG2-' in event:
            #Comprobar cada modo
            for m in range(Mn[actividad_actual]):
                #Comprobar cada RNR_mi
                RNR_mi = []
                for i in range(N_rnr):
                    try:
                        if int(values["Uso_Rnr_{}{}".format(m,i)]) < 0:
                            raise Exception()
                        RNR_mi.append(int(values["Uso_Rnr_{}{}".format(m,i)]))

                    except Exception as e:

                        window['-DATACHECKING2-'].update(visible = True)
                        preparadisimo = False
                        RNR_mi = []
                        break

                if RNR_mi == []:
                    break
                else:
                    RNR.append(RNR_mi)

                RR_mi = []
                for i in range(N_rr):
                    try:
                        #Si es modo no ejecutable
                        if int(values["Uso_Rr_{}{}".format(m,i)]) > lim_rr[i] or int(values["Uso_Rr_{}{}".format(m,i)]) < 0:
                            raise Exception()
                        RR_mi.append(int(values["Uso_Rr_{}{}".format(m,i)]))

                    except Exception as e:

                        window['-DATACHECKING2-'].update(visible = True)
                        preparadisimo = False
                        RR_mi = []
                        break

                if RR_mi == []:
                    break
                else:
                    RR.append(RR_mi)

                try:
                    if int(values["t_{}".format(m)]) <= 0:
                        raise Exception()
                    t_j.append(int(values["t_{}".format(m)]))

                except Exception as e:

                    window['-DATACHECKING2-'].update(visible = True)
                    preparadisimo = False
                    t_j = []
                    break

                try:
                    if int(values["cf_{}".format(m)]) <= 0:
                        raise Exception()
                    cf_j.append(float(values["cf_{}".format(m)]))

                except Exception as e:

                    window['-DATACHECKING2-'].update(visible = True)
                    preparadisimo = False
                    cf_j = []
                    break

                #Si se introdujo bien los datos
                if RNR_mi != [] and RR_mi != [] and t_j != [] and cf_j != []:
                    window['-DATACHECKING2-'].update(visible = False)
                    preparadisimo = True
                    rnr_j.append(RNR_mi)
                    rr_j.append(RR_mi)
                else:
                    rnr_j = rr_j = t_j = cf_j = []


        if preparadisimo and '-SIG2-' in event:

            window.Close()
            return rnr_j, rr_j, t_j, cf_j, True

        if '-ATRAS-' in event:
            window.Close()
            return [], [], [], [], False


    window.Close()



def tercera_pantalla(N_actv):
    """Tercera pantalla donde se introducira la informacion necesaria para
       establecer las relaciones de precedencia entre actividades
    """

    ancho = GetSystemMetrics(0)
    alto = GetSystemMetrics(1)

    col_actv = [[sg.Text("  "),sg.Text("Número de modos de cada actividad")],*[[sg.Text("Actividad {}".format(i)), sg.Input(key = "Actv_{}".format(i)),] for i in range(N_actv)]]

    tercer_layout =[
        [sg.Text('Creador de planificaciones óptimas', size=(30, 1), justification='center', font=("Helvetica", 25), relief=sg.RELIEF_RIDGE)],
        [sg.Text('Establezca las restricciones temporales a partir del orden de precedencia entre las actividades, \n para ello escriba las actividades que preceden a cada una:')]
    ]

    col_predj = [[sg.Text("               "), sg.Text("Predecesores de las actividades")], *[[sg.Text("Actividad {}".format(j+1)), sg.Input(key = "Pred_{}".format(j)),] for j in range(N_actv)]]
    tercer_layout.append([sg.Column(col_predj, scrollable = N_actv > 18, size = (ancho / 3 , alto / 2))])

    #Warning para que no se introduzcan actividades que no existen
    #un segundo warning para que se cumpla el orden topologico
    #tercero para que se introduzca datos validos y no caracteres prohibidos
    tercer_layout.append([
        [sg.Text('Compruebe los datos, las precedencias deben cumplir el orden topológico', key='-TOPOLOGICO-', text_color = 'red', visible = False)],
        [sg.Text('Compruebe los datos, se ha introducido una actividad inexistente', key='-NOEXISTE-', text_color = 'red', visible = False)],
        [sg.Text('Compruebe los datos introducidos, deben ser numeros espaciados o separados por comas', key='-DATACHECKING3-', text_color = 'red', visible = False)],
        [sg.Text('Una vez pulsado el botón no podrá cambiar estos datos. Compruébalos y vuelva a pulsar', key='-REVISE3-', text_color = 'white', visible = False)],
        [sg.Button('Siguiente', key = '-SIG3-'), sg.Button('Salir')]])

    window = sg.Window('Creador de planificaciones óptimas', tercer_layout, resizable=True, finalize=True)

    pred_j = []
    #Flags para continuar y que el usuario compruebe datos
    preparado = False
    preparadisimo = 0

    while True:
        event, values = window.read()


        if event in (sg.WIN_CLOSED, 'Salir'):
            #Para detener la ejecucion de la aplicacin
            raise Exception("Salimos")
            break

        elif '-SIG3-' in event:
            window['-DATACHECKING3-'].update(visible = False)
            window['-TOPOLOGICO-'].update(visible = False)
            window['-NOEXISTE-'].update(visible = False)
            pred_j = []


            for j in range(N_actv):
                try:
                    pred_list = values["Pred_{}".format(j)]
                    if len(pred_list) > 0:

                        #Formateamos, eliminando caracteres y multiples espacios
                        special_cars = "-,_\/|."
                        for sc in special_cars:
                            pred_list = pred_list.replace(sc,' ')

                        pred_list = re.sub(' +', ' ', pred_list)

                        #Convertimos a lista de enteros, ordenados y eliminando repetidos
                        #predecesores = sorted([int(i) for i in list(pred_list.split(" "))], reverse = False)
                        predecesores = list(set([int(i) - 1 for i in list(pred_list.split(" "))]))
                        pred_j.append(predecesores)
                    else:
                        pred_j.append([])
                except Exception as e:

                    window['-DATACHECKING3-'].update(visible = True)
                    pred_j = []
                    break

            if len(pred_j) > 0:

                #Comrpobar si hay ciclos! Con comprobar lo de orden TOPOLOGICO es suficiente
                #Para todo i en pred(j) i < j
                orden_correcto = True
                actividades_posibles = True
                for i, pj in enumerate(pred_j):
                    if len(pj) > 0 :
                        for j in pj:
                            if i <= j:
                                orden_correcto = False
                            if j >= N_actv:
                                actividades_posibles = False

                if not orden_correcto:
                    window['-TOPOLOGICO-'].update(visible = True)
                    pred_j = []

                if not actividades_posibles:
                    window['-NOEXISTE-'].update(visible = True)
                    pred_j = []

            if len(pred_j) > 0:
                preparado = True
                window['-DATACHECKING3-'].update(visible = False)
                window['-TOPOLOGICO-'].update(visible = False)
                window['-NOEXISTE-'].update(visible = False)
                window['-REVISE3-'].update(visible = True)


        if len(pred_j) > 0 and preparado and '-SIG3-' in event:

            preparadisimo += 1

        if len(pred_j) > 0 and preparado and preparadisimo == 2 and '-SIG3-' in event:

            window.Close()
            return pred_j

    window.Close()



def cuarta_pantalla():
    """Cuarta pantalla donde se elegira el modo de ejecucion y la cantidad de experimentos que se realizara
    """
    cuarto_layout =[
        [sg.Text('Creador de planificaciones óptimas', size=(30, 1), justification='center', font=("Helvetica", 25), relief=sg.RELIEF_RIDGE)],
        [sg.Text('Elegir el algoritmo por defecto llevará un tiempo mucho menor \nElegir todos los algoritmos llevará un tiempo considerable pero se asegura encontrar todas las planificaciones óptimas')],
        [sg.Radio('Algoritmo genético por defecto', "ELECCIONAG", key = '-PORDEFECTO-', default=True), sg.Radio('Probar todos los algoritmos genéticos', "ELECCIONAG", key = '-TOTAL-')],
        [sg.Text('Ajuste el número de experimentos (por defecto 10), cuanto mayor sea este número mejores resultados se garantiza a costa de un mayor tiempo de ejecución.')],
        [sg.Slider(range = (1, 100), default_value = 10, orientation = 'h', key= '-NEXP-')],
        [sg.Button('Ejecutar'), sg.Button('Salir')]
    ]

    window = sg.Window('Creador de planificaciones óptimas', cuarto_layout, resizable=True, finalize=True)

    while True:

        event, values = window.read()


        if event in (sg.WIN_CLOSED, 'Salir'):
            break

        elif event in 'Ejecutar':


            window.Close()
            return values['-PORDEFECTO-'], values['-NEXP-']


    window.Close()


def ejecutar_algoritmo(Mn, lim_recursos, recursos_modo, coste_rnr, rnr_modo, coste_fijo, tiempo_modo, predecesores, n_experimentos, pdefecto, queue):
    """Funcion util para ejecutar los algoritmos en un thread y comunicar cuando han terminado su ejecucion para parar la animacion y proceder
    """

    #--------Ejecutamos el algoritmo
    try:
        pareto_set, modos_ban = resuelve(Mn, lim_recursos, recursos_modo, coste_rnr, rnr_modo, coste_fijo, tiempo_modo, predecesores, n_experimentos, pdefecto)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    #--------Cuando termine mandamos cualquier valor a la cola para informar
    queue.put(1)

    #Devolvemos los resultados obtenidos de la ejecucion
    return pareto_set, modos_ban



def pantalla_carga(Mn, lim_recursos, recursos_modo, coste_rnr, rnr_modo, coste_fijo, tiempo_modo, predecesores, n_experimentos, pdefecto):
    """Pantalla de carga con animacion en formato .GIF que avisara cuando los algoritmos hayan acabado su ejecucion
    """

    layout_carga = [
        [sg.Text('Creador de planificaciones óptimas', size=(30, 1), justification='center', font=("Helvetica", 25), relief=sg.RELIEF_RIDGE)],
        [sg.Text('Todo listo! Pulse el botón de ejecución para que comience la optimización', key = '-LISTOTEXTO-', visible = True)],
        [sg.Button('Ejecutar', key = '-EJECUTAR-')],
        [sg.Text('', key = '-START-', visible = True)],
        [sg.Image(filename=r'./loading.gif', enable_events=True, key='-LOADING-', visible = False)],
        [sg.Text('', key = '-SEACABO-')],
        [sg.Button('Ver soluciones', key = '-VSOLS-', visible = False), sg.Button('Salir')]
    ]

    window = sg.Window('Creador de planificaciones óptimas', layout_carga, resizable=True, finalize=True)

    while True:

        event, values = window.Read(timeout=100)
        completed_work = False
        comienza_animacion = False

        if event is None or event in 'Salir':
            break

        elif event in '-EJECUTAR-':
            window.Element('-START-').Update('Comienza la ejecución!', font = ("Helvetica", 16) )
            window.Element('-LOADING-').Update(visible = True)
            window.Element('-EJECUTAR-').Update(visible = False)
            window.Element('-LISTOTEXTO-').Update(visible = False)


            #Creamos pool de thread con un unico proceso
            pool = ThreadPool(processes=1)

            #Objeto para la comunicacion entre los procesos
            queue = Queue()

            #Mandamos el metodo ejecutar_algoritmo con los parametros propios y queue para comunicarlo
            async_result = pool.apply_async(ejecutar_algoritmo, ( Mn, lim_recursos, recursos_modo, coste_rnr, rnr_modo, coste_fijo, tiempo_modo, predecesores, n_experimentos, pdefecto, queue))


            #Mientras no exista comunicacion, queue este vacio
            while queue.empty():
                #Se completa un bucle de todos los frames de la animacion
                for frame in ImageSequence.Iterator(Image.open(r'./loading.gif')):
                    event, values = window.read(timeout=100)
                    window['-LOADING-'].update(data=ImageTk.PhotoImage(frame))

                    if 'Salir' in event:
                        window.Close()
                        return

            #Obtenemos los resultados de ejecutar el algoritmo
            pareto_set, modos_ban = async_result.get()

            window.Element('-SEACABO-').Update('El algoritmo terminó su ejecución!')
            window.Element('-START-').Update(visible = False)
            window.Element('-VSOLS-').Update(visible = True)
            window.Element('-LOADING-').Update(visible = False)

        if '-VSOLS-' in event:
            window.Close()
            return pareto_set, modos_ban

    window.Close()



from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def dibuja_soluciones(canvas, fig):
    """Funcion util que nos sirve para dibujar las distintas soluciones en un grafico
    """
    fig_canvas_agg = FigureCanvasTkAgg(fig, canvas)
    fig_canvas_agg.draw()
    fig_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return fig_canvas_agg

def pantalla_soluciones(resultados, predecesores, lim_recursos, recursos_modo, tiempo_modo, modos_ban):
    """Pantalla destinada al muestreo de las soluciones e identificacion de las mismas para que el usuario pueda
       desplegar la informacion de aquella planificacion que desee
    """

    layout_soluciones = [
        [sg.Text('Creador de planificaciones óptimas', size=(30, 1), justification='center', font=("Helvetica", 25), relief=sg.RELIEF_RIDGE)],
        [sg.Canvas(key = '-CANVAS-')]
    ]

    #---- Crear botones para ver planificaciones en grid de 10 columnas

    N_res = len(resultados)
    cont_res = 0

    while cont_res < N_res:
        botones = []
        cont_diez = 0
        while cont_diez < 10 and cont_res < N_res:
            botones.append(sg.Button("Planificación {}".format(cont_res), key = "Sol_{}".format(cont_res)))
            cont_res += 1
            cont_diez += 1
        layout_soluciones.append(botones)

    #------------------------------------------------------

    layout_soluciones.append([sg.Button('Salir')])

    window = sg.Window('Creador de planificaciones óptimas', layout_soluciones, resizable=True, finalize=True)

    #---- Ordenamos las soluciones según su tiempo para que sea mas legible e intuitivo

    tiempos = [indiv.fitness[0] for indiv in resultados]
    zipped = zip(tiempos, resultados)
    ordenados = sorted(zipped)
    unzipped = zip(*ordenados)
    resultados_ordenados = list(list(unzipped)[1])

    #-------------------------------------------------------


    fig = plt.figure()


    #Preparamos el plot de la solucion con tag numerico en cada punto
    x_pareto = [indiv.fitness[0] for indiv in resultados_ordenados]
    y_pareto = [indiv.fitness[1] for indiv in resultados_ordenados]
    fig.add_subplot().plot(x_pareto, y_pareto, 'ro')

    #----Anotar a cada punto con un numero entero = 0, ... , N
    i = 0

    for x,y in zip(x_pareto, y_pareto):

        label = "{}".format(i)

        plt.annotate(label,
                     (x,y),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center')
        i += 1

    #----------------------------------------------------------

    plt.xlabel("Tiempo")
    plt.ylabel("Coste")

    fig_canvas_agg = dibuja_soluciones(window['-CANVAS-'].TKCanvas, fig)

    while True:

        event, values = window.read()

        if event in (sg.WIN_CLOSED, 'Salir'):
            break


        #--- Mostrar ventana con info de la planificacion que se elija

        for boton in range(len(resultados_ordenados)):
            if event == "Sol_{}".format(boton):
                actividad_dia = decodifica(resultados_ordenados[boton].candidate, predecesores, lim_recursos, recursos_modo, tiempo_modo)
                muestra_planificacion(resultados_ordenados[boton], boton, actividad_dia, modos_ban)


    window.Close()



def muestra_planificacion(planificacion, plan_id, actividad_dia, modos_ban):
    """Pantalla estilo pop-up que muestra la informacion pertinente a la planificacion que el usuario haya seleccinado
    """
    layout = [
        [sg.Text("Planificación número {}".format(plan_id), font = ('Helvetica', 20))],
        [sg.T(" ")]
    ]

    actividades = range(len(actividad_dia))
    contador = 0
    num_actividades = len(actividad_dia)

    #Ordenamos tuplas actividad_dia por actividad
    actividad_dia = sorted(actividad_dia)

    actividad_modos = planificacion.candidate.m

    #Para mostrar la informacion de la planificacion creamos una tabla
    #para que se vea bien usamos un max de 10 columnas
    #en cada columna se especifica la actividad, el dia que comienza
    #y el modo de ejecucion
    while num_actividades > 0 :
        #Si todavia hay mas de 10 actividades por entablar
        if num_actividades - 10 >= 0:
            headings = ["Actividad {}".format(i) for i in actividades[contador:contador+10]]
            header =  [[sg.Text('       ')] + [sg.Text(h, size=(9,1),background_color = 'black', pad=(1,1)) for h in headings]]

            layout.append(header)

            contenido =  ["{}".format(i[1]) for i in actividad_dia[contador:contador+10]]
            dias = [[sg.Text('Día   ')] + [sg.Text(d, size=(9,1),background_color = 'gray', text_color = 'black', pad=(1,1)) for d in contenido]]

            layout.append(dias)

            contenido_modo = ["{}".format(i) for i in actividad_modos[contador:contador+10]]
            modos = [[sg.Text('Modo')] + [sg.Text(m, size=(9,1),background_color = 'gray', text_color = 'black', pad=(1,1)) for m in contenido_modo]]

            layout.append(modos)

            contador += 10
            num_actividades -= 10

        #Entablar el resto
        else:

            headings = ["Actividad {}".format(i) for i in actividades[contador:contador+num_actividades%10]]
            header =  [[sg.Text('       ')] + [sg.Text(h, size=(9,1),background_color = 'black', pad=(1,1)) for h in headings]]

            layout.append(header)

            contenido =  ["{}".format(i[1]) for i in actividad_dia[contador:contador+num_actividades%10]]
            dias = [[sg.Text('Día   ')] + [sg.Text(d, size=(9,1),background_color = 'gray', text_color = 'black', pad=(1,1)) for d in contenido]]

            layout.append(dias)

            contenido_modo = ["{}".format(i) for i in actividad_modos[contador:contador+num_actividades%10]]
            modos = [[sg.Text('Modo')] + [sg.Text(m, size=(9,1),background_color = 'gray', text_color = 'black', pad=(1,1)) for m in contenido_modo]]

            layout.append(modos)

            contador += 10
            num_actividades -= 10


    layout.append([[sg.T(" ")],[sg.T("Tiempo total: {}".format(planificacion.fitness[0]) ,font = ('Arial', 15))]])
    layout.append([sg.T("Coste total: {}".format(planificacion.fitness[1]), font = ('Arial', 15))])
    layout.append([[sg.T(" ")],[sg.Button('Cerrar')]])

    window = sg.Window("Planificación {}".format(plan_id), layout, resizable=True, finalize=True)

    while True:

        event, _ = window.read()

        if event in (sg.WIN_CLOSED, 'Cerrar'):
            break

    window.Close()



if __name__ == "__main__":

    #Flag para saber si se ha cerrado el programa
    continuar = True

    #-----Primera pantalla informacion general
    try:
        N_actv, N_rnr, N_rr, coste_rnr, lim_rr, Mn = primera_pantalla()
    except Exception as e:
        continuar = False


    #---------------------------------------------------------------------------

    #-----Segunda pantalla informacion de cada actividad, de forma iterativa
    #     se puede avanzar o retroceder en las actividades
    if continuar:
        recursos_modo = []
        rnr_modo = []
        dias_modo = []
        coste_fijo = []


        actividad_actual = 0
        while actividad_actual < N_actv and continuar:
            try:
                rnr_j, rr_j, t_j, cf_j, siguiente = segunda_pantalla(N_actv, N_rnr, N_rr, coste_rnr, lim_rr, Mn, actividad_actual)

                if siguiente:
                    actividad_actual += 1
                    recursos_modo.append(rr_j)
                    rnr_modo.append(rnr_j)
                    dias_modo.append(t_j)
                    coste_fijo.append(cf_j)


                else:
                    actividad_actual -= 1
                    recursos_modo.pop()
                    rnr_modo.pop()
                    dias_modo.pop()
                    coste_fijo.pop()

            except:
                continuar = False
    #---------------------------------------------------------------------------

    #-----Tercera pantalla conocer las relaciones de precedencia
    if continuar:
        try:
            predecesores = tercera_pantalla(N_actv)
        except:
            continuar = False
    #---------------------------------------------------------------------------

    #-----Cuarta pantalla: tipo de ejecucion y num. de experimentos
    if continuar:
        try:
            pdefecto, n_experimentos = cuarta_pantalla()

        except:
            continuar = False


    #-----Quinta pantalla: pantalla de carga y modos prohibidos
    if continuar:
        try:
            resultados, modos_ban = pantalla_carga(Mn, lim_rr, recursos_modo, coste_rnr, rnr_modo, coste_fijo, dias_modo, predecesores, n_experimentos, pdefecto)

        except:
            continuar = False
    #---------------------------------------------------------------------------

    #-----Ultima pantalla: muestra grafica con planificaciones segun su
    #     tiempo y coste y permite obtener la informacion de cada planificacion
    if continuar:
        #Tenemos que cambiar los modos de las actividades en funcion de los modos baneados que haya
        #para que se muestre todo bien
        for r in resultados:
            actividad_modos = r.candidate.m

            #Conseguimos los modos reales con los modos_ban--------
            new_modos = actividad_modos[:]

            for i, m in enumerate(actividad_modos):
                for mb in modos_ban[i]:
                    if new_modos[i] >= mb:
                        new_modos[i] += 1

            r.candidate.m = new_modos[:]

        pantalla_soluciones(resultados, predecesores, lim_rr, recursos_modo, dias_modo, modos_ban)
    #---------------------------------------------------------------------------
