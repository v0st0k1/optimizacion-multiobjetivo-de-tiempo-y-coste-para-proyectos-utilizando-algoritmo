# Abstract

A lo largo de los años se han realizado numerosos trabajos en torno a los problemas de optimización en el desarrollo de planificaciones para proyectos. La técnica más usada para el desarrollo de planificaciones es el método de la ruta crítica (CPM de sus siglas en inglés). Esta técnica, aunque pueda resultar útil en la creación de planificaciones en algunos aspectos, presenta ciertas deficiencias, ya que solo atañe al tiempo de desarrollo del proyecto. Este método nos resulta insuficiente si buscamos una solución al problema más allá de la planificación temporal de las actividades que componen el proyecto; por ejemplo, si tenemos en cuenta que existen múltiples modos de llevar a cabo una misma actividad. De esta forma, dependiendo del modo elegido para cada actividad podría variar el tiempo que necesite para terminarse así como su necesidad de recursos y, por lo tanto, también el coste que implicaría para el desarrollo del proyecto. Para resolver este problema de optimización, llamado MRCPSP por sus siglas en inglés Multi-mode Resource Constrained Project Scheduling Problem, el problema reside en la relación entre las dos variables a optimizar, minimizándolas en este caso, estas son el tiempo y el coste. Esta relación la podemos llamar compensación tiempo-coste, o tradeoff de su anglicismo, o TCT de sus siglas en inglés (Time-Cost Trade off). Este problema como la mayoría de los problemas de optimización multiobjetivo se encuentra dentro de los problemas NP-Hard, de dificultad no-polinómica, por tanto el uso de técnicas  exactas para encontrar una solución en tiempo razonable es totalmente inviable. Por ello,  y basándonos en la literatura que existe al respecto de este tipo de problema, vamos a  presentar una solución desde un enfoque evolutivo haciendo uso de algoritmos genéticos.

# Manual de uso

## Resumen

Nuestra aplicación, OPyMM, le permite encontrar la mejor planificación
posible a su proyecto de obra. Simplemente, deberá iniciar la aplicación
e introducir los datos pertinentes al proyecto tal y como se le irá
indicando.

Los datos necesarios para construir las planificaciones serán:

  - Número de actividades

  - Número de recursos renovables

  - Número de recursos no renovables

  - Relaciones de precedencia entre actividades

  - Límite de los recursos renovables

  - Precio por unidad de los recursos no renovables

  - Número de modos de ejecución posibles de cada actividad

  - Gasto de recursos para cada actividad-modo

  - Tiempo necesario para cada actividad-modo

  - Coste fijo para cada actividad-modo

## Inicialización

La aplicación no necesita una instalación previa. Para iniciar la
aplicación simplemente necesita descomprimir el archivo *OPyMM.rar* y
lanzar el archivo ejecutable *OPyMM.exe* dentro del directorio *dist*.

## Guía de uso

En la primera pantalla se debe introducir los datos más básicos que
conforman una planificación: el número de actividades y el número de
recursos, renovables y no renovables, de los que se hará uso. Estos
datos tienen que ser números enteros y positivos.

Una vez introducidos estos datos, se pulsa el botón ’Establecer’ y se
actualizará la pantalla. Si desea cambiar los datos anteriores en
cualquier momento durante esta pantalla simplemente vuelva a pulsar el
botón ’Establecer’.

En las nuevas celdas se debe introducir, en orden de izquierda a
derecha, el coste por unidad de los recursos no renovables, el límite
por unidad de tiempo de los recursos renovables y el número de modos que
posee cada actividad. Los primeros dos conjunto de datos deberán ser
número reales positivos. El número de modos de cada actividad deberá
ser entero y positivo. Una vez introducido estos datos, para continuar
pulse el botón ’Siguiente’.

En las siguientes pantalla deberá aportar la información necesaria para
cada modo de cada actividad. Podrá saber en que actividad se encuentra
observando la cabecera en la parte superior de la pantalla. En estas
pantallas podrá introducir el gasto de recursos no renovable y
renovable, el tiempo necesario y el coste fijo de cada modo para la
actividad en cuestión.

Puede introducir los datos para cada actividad en el orden que prefiera
viajando entre las distintas actividades pulsando los botones
’Siguiente’ y ’Atrás’.

Acabado este proceso, en la última pantalla correspondiente a la
actividad con mayor número el botón ’Continuar’ pasará a la siguiente
pantalla. En esta pantalla deberá introducir las relaciones de
precedencia para cada actividad. Cada actividad vendrá acompañada de una
celda donde deberá introducir el número de las actividades que deben
estar completadas para que dé inicio esa actividad en cuestión.

En la siguiente pantalla tendrá que elegir entre uno de los siguientes
modo de ejecución.

  - **Ejecución por defecto.** Esta ejecución es más rápida pero es
    probable que no se llegue a encantar todas las mejores
    planificaciones.

  - **Ejecución completa.** Esta ejecución utilizará todos los tipos de
    algoritmos que se dispone. De esta forma, es probable que se
    encuentre alguna solución que con el anterior modo de ejecución no
    se hallase. Este tipo de ejecución tarda mucho más que la ejecución
    por defecto.

El siguiente dato a introducir será el número de experimentos que hará
cada algoritmo para buscar las soluciones óptimas. Se recomienda que si
no se comprende su función se deje en valor por defecto en 10. Explicado
resumidamente, cuanto menor sea el número de experimentos peores serán
las soluciones obtenidas y cuanto mayor sea mejores serán, hasta cierto
límite, donde ya no se hallarán mejores. Cuando tenga elegido estos dos
aspectos pulse ’Siguiente’ para dar paso a la ventana de ejecución.

En la ventana de ejecución simplemente tendrá que pulsar el botón
’Ejecutar’ para que la herramienta empiece a desarrollar y optimizar
las planificaciones. Mientras ocurre esta ejecución aparecerá una
animación con la palabra *Loading*.

Una vez concluida la ejecución de los algoritmos y encontradas las
mejores planificaciones, se le mostrará una pantalla donde aparecerán
las planificaciones dispuestas sobre un eje de coordenadas para
representar lo buenas que son en función del tiempo y coste. Estas
planificaciones vendrán identificadas con un número. Debajo de esta
gráfica aparecerán tantos botones como planificaciones halla. Para
desplegar la información sobre alguna de las planificaciones de la
gráfica simplemente tendrá que pulsar el botón con el mismo
identificador que dicha planificación. Podrá desplegar y cerrar la
información de cuantas planificaciones optimizadas desee.

Una vez tomada la información de las mejores planificaciones, podrá
cerrar la aplicación y disfrutar de sus planificaciones optimizadas.

# Comportamiento del algoritmo

![image](https://user-images.githubusercontent.com/10656513/171680601-92accfbd-4556-4260-802b-edf9d9c10002.png)


# Experimentación

Para nuestra fase de experimentación trabajaremos con distintos
algoritmos genéticos formados por diferentes combinaciones de operadores
de cruce, selección, mutación y reemplazo. Por lo tanto, es necesario
definir las métricas de calidad que usaremos para comparar los distintos
conjuntos de soluciones no-dominadas que obtendremos con estos
algoritmos.

Una métrica o un indicador de calidad nos sirve para comparar algoritmos
en términos de efectividad y eficiencia. Numerosos investigadores han
trabajado en este campo para diseñar mecanismos que nos permitan
comparar los resultados dados por distintos algoritmos y determinar cuál
de estos presentó mejores resultados. Por ejemplo, en el estudio de
Zitzler (2000)  se nos propone los siguientes tres objetivos a tener en
cuenta para comparar los conjuntos de soluciones:

  - Minimizar la diferencia entre el conjunto de soluciones no-dominadas
    dado por un algoritmo y el frente de Pareto óptimo (convergencia).

  - Lograr una distribución uniforme de las soluciones no-dominadas en
    su conjunto (diversidad).

  - Buscar el mayor grado de cardinalidad del conjunto de soluciones
    (cantidad).

En el estudio de Riquelme et al. (2015) podemos ver una revisión de 54
métricas de optimización multiobjetivo que se han utilizado en la
literatura de este ámbito. En este estudio se indica que las métricas
más utilizadas son: hipervolumen, distancia generacional invertida,
indicador de épsilon aditivo y distancia generacional .

Estas métricas pueden ser también unarias o binarias. Se dice que una
métrica es unaria cuando para calcularla solo se necesita como
parámetro un conjunto de soluciones formando el frente de Pareto, y se
dice que son binarias cuando para su cálculo se necesitan dos conjuntos.

Para nuestro trabajo decidimos que las mejores métricas a utilizar serán
el hipervolumen, la distancia generacional invertida y el indicador de
épsilon aditivo.

### Hipervolumen

Es una métrica unaria usada para medir los aspectos de convergencia,
diversidad y cardinalidad de un frente de Pareto dado para problemas
donde todos los objetivos deben ser minimizados, siendo la única métrica
unaria con esta capacidad. Esta métrica mide el tamaño del espacio
objetivo logrado por los miembros de un conjunto de soluciones
no-dominadas  . Y es como bien indica el trabajo de Riquelme et al.
(2015) la más utilizada de todas las métricas por parte de la comunidad.

Para conseguir medir este espacio es necesario aportar un punto de
referencia en el espacio (punto **r**), como podemos ver en la siguiente
figura.

![image](https://user-images.githubusercontent.com/10656513/171679170-2f03ebb1-d1cf-4a36-abef-3fb6964f8313.png)

Para este ejemplo de un problema de minimización con dos objetivos, dado
el punto de referencia \(r\) y el frente de soluciones no-dominadas
formado por los puntos \(P1\), \(P2\), \(P3\) y \(P4\).
Podemos ver gráficamente como el cálculo del hipervolumen para este
ejemplo se puede obtener de una forma visual y simple, calculando el
área total que forman los distintos rectángulos cuyas esquinas están
formadas por los distintos puntos del frente y el punto de referencia.

En la figura anterior podemos ver como las áreas coloreadas tienen
marcado el cálculo del área de dichas zonas, por lo que sumándolas las
cuatro podemos calcular el hipervolumen de este conjunto de soluciones
que sería de \(20\).

Por tanto, podemos ver claramente como al tratarse de problemas de
minimización los algoritmos que alcanzan mayores valores para esta
medida son objetivamente mejores.

### Distancia generacional invertida (IGD)

La distancia generacional invertida es una métrica binaria en la que,
aparte de dar como parámetro el frente al cual se le quiere calcular la
métrica (conjunto de aproximación), es necesario aportar el frente de
Pareto óptimo para el problema en cuestión (conjunto de referencia); en
caso de no poder contar con el frente óptimo para el problema, como es
el caso del problema descrito en este trabajo, es necesario hallar un
frente de Pareto que no sea el óptimo pero que al menos sea lo más
aproximado a este. Para lograr este frente de Pareto aproximado lo que
se suele hacer es: unir todos los frentes de Pareto obtenidos por todos
los experimentos de todos los algoritmos a probar, descartar aquellas
soluciones que sean dominadas y entonces ya tendríamos nuestro frente de
Pareto aproximado al óptimo.

La distancia generacional invertida nos da información sobre la
diversidad y la convergencia del conjunto de aproximación . En resumen,
nos proporciona la distancia promedio entre cualquier punto desde el
conjunto de referencia y su punto más cercano desde el conjunto de
aproximación. Podemos formalizar la IGD de un frente A de la siguiente
forma:

![image](https://user-images.githubusercontent.com/10656513/171678893-468836b7-45e7-4767-8d31-cc7755cfd6f4.png)

Donde n es el número de soluciones en el frente óptimo de Pareto y
di es la distancia euclidiana entre cada punto de dicho frente y
la solución más cercana.

En este indicador un conjunto de soluciones no-dominadas A es mejor que
otro conjunto B si IGD(A) < IGD(B)

### Indicador de épsilon aditivo

Épsilon aditivo es un indicador binario al que se le aporta tanto el
frente de Pareto obtenido, al cual le aplicaremos la medida, y el frente
de Pareto óptimo o aproximado a este.

Dado un conjunto no-dominado \(A\) como solución del problema, el
indicador épsilon nos da la medida de la menor distancia que se
necesitaría para trasladar cada solución de \(A\) de forma que domine al
frente de Pareto óptimo o aproximado de dicho problema.

Entonces, podemos entender que cuanto menor sea el valor del indicador
épsilon para un conjunto frente de Pareto \(A\) más cerca estará de su
frente óptimo \(P\) y por tanto, mejor sería. Si el valor proporcionado
por el indicador fuese \(0\) significaría que \(R\;\subseteq\;A\)


## Hallar frente de Pareto aproximado al óptimo

![image](https://user-images.githubusercontent.com/10656513/171679737-25af1f0c-3345-4a26-8ff6-8eeddaa9d51e.png)

## Ejemplo de resultado de la experimentación 

![image](https://user-images.githubusercontent.com/10656513/171679890-d071bc54-b6b6-4d80-8ee5-1637c8f7ad94.png)


![image](https://user-images.githubusercontent.com/10656513/171679941-b851c8ba-e0f3-470b-954b-4c418aabe7dc.png)

