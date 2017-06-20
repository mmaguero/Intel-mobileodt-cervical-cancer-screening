<!--Portada-->

<div class="portada">


# Práctica 2
# Clasificación de Imágenes
*****

<img src="http://secretariageneral.ugr.es/pages/ivc/descarga/_img/vertical/ugrmarca01color_2/!" alt="Logo UGR" style="width: 200px; height: auto;"/>

<div class="portada-middle">

## Nombre del equipo: **AythaE_MMAguero**
## Ranking: **242** Puntuación: **0.84173**
</br>

### Sistemas Inteligentes para la Gestión en la Empresa
### Máster en Ingeniería Informática
### Curso 2016/17
### Universidad de Granada

</div>

> Nombre: Aythami Estévez Olivas
> Email: <aythae@correo.ugr.es>
> Nombre: Marvin M. Agüero Torales
> Email: <maguero@correo.ugr.es>

</div>

<!-- Salto de página -->
<div style="page-break-before: always;"></div>

## Índice

<!--
Ejemplo de Indice final eliminando el enlace y añadiendo el número de página
- Apartado 1 <span style='float:right'>2</span>
-->

<!-- toc -->

- [1. Exploración de datos](#1-exploracion-de-datos)
  * [1.1. Las mujeres y los niños primero](#11-las-mujeres-y-los-ninos-primero)
  * [1.2. Clase social](#12-clase-social)
  * [1.3. Uniendo ambos criterios](#13-uniendo-ambos-criterios)
  * [1.4. Otras variables](#14-otras-variables)
- [2. Preprocesamiento de datos](#2-preprocesamiento-de-datos)
  * [2.1. Integración y detección de conflictos e inconsistencias en los datos](#21-integracion-y-deteccion-de-conflictos-e-inconsistencias-en-los-datos)
  * [2.2. Transformaciones](#22-transformaciones)
  * [2.3. Reducción de datos](#23-reduccion-de-datos)
- [3. Técnicas de clasificación y discusión de resultados](#3-tecnicas-de-clasificacion-y-discusion-de-resultados)
  * [3.1. Árbol de decisión simple](#31-arbol-de-decision-simple)
  * [3.2. Random Forest](#32-random-forest)
  * [3.3. CForest](#33-cforest)
- [4. Conclusiones y trabajos futuros](#4-conclusiones-y-trabajos-futuros)
- [5. Listado de soluciones](#5-listado-de-soluciones)
- [Bibliografía](#bibliografia)

<!-- tocstop -->

<!-- Salto de página -->
<div style="page-break-before: always;"></div>

## 1. Exploración de datos

Utilizaremos CNNs (Convolutional Neural Networks) y técnicas de clasificación para la predicción del conjunto de imágenes. Antes que nada descargamos las imágenes, hay un conjunto de train y otro de test, también un conjunto extra de train, alrededor de 35 GB en imágenes (2000*3000px aproximadamente, en formato .jpg), por lo que debemos aplicar técnicas para poder reducirlas. A simple vista, se ven imágenes de fondo verde, otras a color, enfocadas de distintos puntos, por lo que, además de reducirlas, se podría tratar de hacerlas más uniformes, en blanco y negro por ejemplo, utilizar data augmentation o aumento de datos, respetar la relación de aspecto, etc.

Para comprobar si el conjunto de datos es balanceado o no-balanceado, necesitamos saber cuantos registros hay de cada variable de clase. Hasta 1.5 se considera clases balanceadas, poseemos tres tipos de clases:
- Tipo 1, con 250
- Tipo 2, con 781
- Tipo 3, con 450

Ahora si tenemos en cuenta el conjunto total de train, incluyendo extras, tenemos para cada uno:
- Tipo 1, con 1441
- Tipo 2, con 4348
- Tipo 3, con 2426

Si consideramos sólo las de train originales, vemos que el conjunto no es balanceado (3.12%), si tomamos además las extras (3.07%), tampoco, aunque se trata de un leve no-balanceo.

En deep learning las redes neuronales necesitan ser entrenadas con un gran número de imágenes para lograr un rendimiento satisfactorio, y sí el conjunto de datos de imagen original es limitada, es mejor hacer el aumento de datos para aumentar el rendimiento. Hay muchas maneras de hacer el aumento de datos, como horizontally flipping, random crops and color jitterin, o intentar combinaciones de múltiples procesamientos diferentes, por ejemplo, realizar la rotación y escalar al azar al mismo tiempo. Además, se puede tratar de aumentar la saturación y el valor (componentes S y V del espacio de color HSV), incluso propuestas como Fancy PCA, introducida por Alex-Net en 2012: "la Fancy PCA podría capturar aproximadamente una propiedad importante de las imágenes naturales, es decir, que la identidad del objeto es invariable a los cambios en la intensidad y el color de la iluminación" (Xiu-Shen, 2015).

En (Schmidt, 2017) se hace un análisis exploratorio sobre el conjunto original de imágenes, vamos a adaptarlo para tomar también el conjunto total de train.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread, imshow
import cv2

%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from subprocess import check_output
print(check_output(["ls", "../input/train"]).decode("utf8"))
```
Type_1 <br/>
Type_2 <br/>
Type_3

```python
from glob import glob
basepath = '../input/train/'

all_cervix_images = []

for path in sorted(glob(basepath + "*")):
    cervix_type = path.split("/")[-1]
    cervix_images = sorted(glob(basepath + cervix_type + "/*"))
    all_cervix_images = all_cervix_images + cervix_images

all_cervix_images = pd.DataFrame({'imagepath': all_cervix_images})
all_cervix_images['filetype'] = all_cervix_images.apply(lambda row: row.imagepath.split(".")[-1], axis=1)
all_cervix_images['type'] = all_cervix_images.apply(lambda row: row.imagepath.split("/")[-2], axis=1)
all_cervix_images.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>imagepath</th>
      <th>filetype</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>../input/train/Type_1/0.jpg</td>
      <td>jpg</td>
      <td>Type_1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>../input/train/Type_1/10.jpg</td>
      <td>jpg</td>
      <td>Type_1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>../input/train/Type_1/1013.jpg</td>
      <td>jpg</td>
      <td>Type_1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>../input/train/Type_1/1014.jpg</td>
      <td>jpg</td>
      <td>Type_1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>../input/train/Type_1/1019.jpg</td>
      <td>jpg</td>
      <td>Type_1</td>
    </tr>
  </tbody>
</table>

Con el anterior bloque, creamos un práctico *dataframe* para hacer algunas agregaciones en los datos.

En este conjunto de imágenes (1481) los archivos están en formato .JPG, el Tipo 2 es el más común, cuenta con un poco más del 50% en los datos de entrenamiento en total, y el Tipo 1 por otro lado, tiene un poco menos del 20%.

```python
print('We have a total of {} images in the whole dataset'.format(all_cervix_images.shape[0]))
type_aggregation = all_cervix_images.groupby(['type', 'filetype']).agg('count')
type_aggregation_p = type_aggregation.apply(lambda row: 1.0*row['imagepath']/all_cervix_images.shape[0], axis=1)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

type_aggregation.plot.barh(ax=axes[0])
axes[0].set_xlabel("image count")
type_aggregation_p.plot.barh(ax=axes[1])
axes[1].set_xlabel("training size fraction")
```
<img src="https://www.kaggle.io/svf/1263512/8dd58488255be0d7f6798385495dd5cb/__results___files/__results___5_2.png" alt="Cantidad de imágenes / Fracción de entrenamiento" style="width: 450px; height: auto;"/>

Lo mismo sucede sobre el conjunto de entrenamiento adicional (6734).

<img src="https://www.kaggle.io/svf/1263512/8dd58488255be0d7f6798385495dd5cb/__results___files/__results___11_2.png" alt="Cantidad de imágenes / Fracción de entrenamiento" style="width: 450px; height: auto;"/>

Como era de esperarse al utilizar el conjunto total (8215), la tendencia es la misma.

<img src="https://www.kaggle.io/svf/1263512/8dd58488255be0d7f6798385495dd5cb/__results___files/__results___15_2.png" alt="Cantidad de imágenes / Fracción de entrenamiento" style="width: 450px; height: auto;"/>

Como se ven en lás imágenes, el conjunto de datos no es balanceado, tiene una cantidad considerable de imágenes y de gran tamaño. En el siguiente apartado hablaremos de los métodos de Preprocesamiento aplicados a este conjunto.

## 2. Preprocesamiento de datos

Descripción y discusión de las técnicas de preprocesamiento
utilizadas y análisis crı́tico de su utilidad en el problema.
.
- [ ] Integración y detección de conflictos e inconsistencias en los datos: valores perdidos, valores fuera de rango, ruido, etc.
- [ ] Transformaciones: normalización con OpenCV, agregación, generación de caracterı́sticas adicionales, etc.
- [ ] Reducción de datos: técnicas utilizadas para selección de caracterı́sticas, selección de ejemplos, discretización, agrupación de valores, etc.
- [ ] Aumento de datos: técnicas utilizadas para incrementar la cantidad de datos disponibles.

## 3. Técnicas de clasificación y discusión de resultados

Discusión de las técnicas y herramientas de clasificación empleadas, justificación de su elección.

Descripción y discusión de las soluciones obtenidas, incidiendo en la interpretación de los resultados. Análisis comparativo en caso de utilizar diferentes técnicas y/o parámetros de configuración en diferentes aproximaciones.

### Herramientas

Hemos utilizado [Keras](https://keras.io/), una librería de Python para Deep Learning, con [Tensorflow](https://www.tensorflow.org/) como backend, una librería de Python para computación numérica, con un equipo con una GPU (Unidad de Procesamiento Gráfico) antigua (pero para suerte nuestra, aún podía ejecutar esta herramienta), y con otro equipo (con un para de años) con CPU (Unidad de Procesamiento Central) solamente. Los tiempo con CPU estaban sobre el doble (o un poco más) que con GPU, lo que delata que no se trata de una GPU de última generación.

Además nos hemos valido de [Scikit-learn](http://scikit-learn.org), una librería de Python para Machine Learning, para realizar predicciones con algoritmos de clasificación sobre las características extraídas de las CNNs.

La decisión de utilizar de estas herramientas es porqué son las más populares en el ámbito de la competencia, en la misma página de Kaggle hay mucha documentación proporcionada por la comunidad: como discusiones, tutoriales, etc. Además hoy por hoy, Tensorflow se ganado el mercado de Deep Learning, que con Keras se logra abstraerla bastante, pudiendo aprovecharla en tan sólo pocas líneas. Intentamos utilizar además las herramientas sugeridas en la asignatura sin éxito, puede consultarse más abajo el Apartado *Otras herramientas* para más detalles.

#### Otras herramientas

Primeramente hemos intentado utilizar las herramientas propuestas en clase, [Intel Deep Learning SDK](https://software.intel.com/en-us/deep-learning-training-tool) y [MXNet](http://mxnet.io/api/r/index.html) con R. La primera presentó muchos problemas a la hora de la instalación, que una vez subsanados, al ser una herramienta en versión beta, no iba muy bien de rendimiento en local sobre Linux: tiempos de cómputo altos dejando inutilizado el ordenador para otras tareas, incluso a veces la herramienta daba fallos posteriores a la instalación y uso que que la dejaba no funcional. Pero creemos que en un futuro sería una herramienta muy completa, puesto que se pueden utilizar varias técnicas a tan sólo un clic. La segunda, como veníamos familiarizados con ella (al utilizarlas en prácticas), quisimos montarla sobre GPU, pero el resultado no fue bueno, con o sin GPU no se completaban las tareas, ya que R Studio, no podía funcionar del todo bien con MXNet.

Además de las anteriores, intentamos aprovechar el [clúster Colfax](https://colfaxresearch.com/kaggle-2017/) con 256 cores con Keras. Utilizando como backend Theano, nos fue imposible instalar algunos módulos de Python, puesto que utilizan su propia arquitectura y hay algunos paquetes, módulos o versiones faltantes, y el camino para hacerlo funcionar era largo y extenso; con TensorFlow como backend, solo corría en la increíble cantidad de un core solamente, con un tiempo de cómputo de un ordenador stándart mucho menor. Para lo que si nos fue útil, fue para el tratamiento de imágenes, donde si pudimos aprovechar la capacidad de cómputo de este clúster. Además en el preprocesamiento, como se mencionó anteriormente utilizamos EBImage, una librería de R para el tratamiento de imágenes.

Incluso hemos intentando contratar instancias de [Amazon Web Services (AWS)](https://aws.amazon.com/es/ec2/Elastic-GPUs/) con GPU con nuestras cuentas de estudiante pero no era posible utilizar éstas debido a las limitaciones de dichas cuentas.

### Técnicas

#### Modelo propio

#### Learning from scratch vs fine-tuning

#### Uso de CNNs con Machine Learning

#### Post-procesamiento OVO

#### Otros

#### Feature maps
Con VGG16, red entrenada y fine-tuning, Red completa y Última capa

- [ ] ensambles, etc.

### Consideraciones

Capacidad de cómputo
Necesidad de clústeres
Necesidad de GPUs de última generación
Trabajar en CPU es un martirio
EL resultado es directamente proporcional a la capacidad de cómputo
Desigualdad de condiciones
Tiempos de ejecución

## 4. Conclusiones y trabajos futuros

Breve resumen de las técnicas aplicadas y de los resultados obtenidos.

### Trabajo futuro
... ideas para continuar mejorando las soluciones desarrolladas.

Se podría aplicar técnicas como *features extraction* sobre las imágenes, aunque eso requiere conocer a fondo herramientas como [OpenCV](http://opencv.org/) o [SciLab](http://www.scilab.org/), que se alejan del objetivo de la asignatura y requieren su esfuerzo de aprendizaje. En [11] mencionan que es útil para detectar y aislar diversas porciones o características de una imagen, particularmente importante en el área del *reconocimiento óptico de caracteres*. Existen técnicas de bajo nivel como la detección de bordes, o esquinas, entre otros, teniendo en cuenta la curvatura y el movimiento de la imagen; de forma basada, como umbrales, extracción de blob, o "Hough transform"; y otros de métodos flexibles, como formas parametrizables/deformables o contornos activos.

Utilizar ensambles sobre CNN

Aplicar OVO además de OVA, de una manera manual, puesto que existen módulos como scikit-learn de Python, con modelos como SVM que permite aplicar OVO sobre un conjunto de datos.

<!-- Salto de página -->
<div style="page-break-before: always;"></div>

## 5. Listado de soluciones

Las siguientes abreviaturas representan el Preprocesamiento o el Algoritmo/Software utilizado en las soluciones (y utilizadas en el tabla de abajo):
- ...: ...
-

La siguiente tabla recoge las distintas soluciones presentadas en Kaggle.

| Nº Solución | Preprocesamiento | Algoritmo/Software | % de acierto entrenamiento | % de acierto test (Kaggle) | Posición Ranking           |
|-------------|-------------------|--------------------|----------------------------|----------------------------|----------------------------|
| 1           |                   |                    | 0.8554                     | 0.88509                    | 292                        |
| 2           |                   |                    | 0.7675                     | 1.30324                    | 300                        |
| 3           |                   |                    | **0.**                     | **0.841**                  | **stg 1: 242 / stg 2: 75** |
| 4           |                   |                    | 0.                         | 0.845                      | 261                        |
| 5           |                   |                    | 0.                         | 0.98                       | 261


Posición al cierre de la primera etapa: 160

<!-- Salto de página -->
<div style="page-break-before: always;"></div>

## Bibliografía

<p id="1">

[1]: Kaggle (n.d). Intel & MobileODT Cervical Cancer Screening. Recuperado en Junio de 2017, desde <https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening>

</p>

<p id="2">

[2]: Xiu-Shen, W. (2015, octubre). Must Know Tips/Tricks in Deep Neural Networks (by <a href="http://lamda.nju.edu.cn/weixs/">Xiu-Shen Wei</a>). Recuperado en Junio de 2017, a partir de http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html

</p>

<p id="3">

[3]: Marek 3000 (n.d). Test num 001. Recuperado en Junio de 2017, a partir de <https://www.kaggle.com/marek3000/test-num-001>

</p>

<p id="4">

[4]: J. Brownlee (9 de Agosto 2016). How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras. Recuperado en Junio de 2017, desde <http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/>

</p>

<p id="5">

[5]: Keras (n.d). ImageNet: VGGNet, ResNet, Inception, and Xception with Keras. Recuperado en Junio de 2017, desde <https://keras.io/>

</p>

<p id="6">

[6]: A. Rosebrock (20 de marzo 2017). ImageNet: VGGNet, ResNet, Inception, and Xception with Keras. Recuperado en Junio de 2017, desde <http://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/>

</p>

<p id="7">

[7]: D. Gupta (1 de junio 2017). Transfer learning & The art of using Pre-trained Models in Deep Learning. Recuperado en Junio de 2017, desde <https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/>

</p>

<p id="8">

[8]: F. Chollet (5 de junio 2016). Building powerful image classification models using very little data. Recuperado en Junio de 2017, desde <https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html>

</p>

<p id="9">

[9]: scikit-learn (n.d). Scikit-learn, Machine Learning in Python. Recuperado en Junio de 2017, desde <http://scikit-learn.org/stable/index.html>

</p>

<p id="10">

[10]: P. Schmidt (n.d). Cervix EDA & Model selection. Recuperado en Junio de 2017, desde <https://www.kaggle.com/philschmidt/cervix-eda-model-selection>

</p>

<p id="10">
[11]: Feature extraction. (2017, mayo 12). En Wikipedia. Recuperado a partir de https://en.wikipedia.org/w/index.php?title=Feature_extraction&oldid=779974336

</p>

<!-- Salto de página -->
<div style="page-break-before: always;"></div>

## Anexos

Se adjuntan los scripts utilizados de Python, también disponibles en [GitHub](https://github.com/mmaguero/Intel-mobileodt-cervical-cancer-screening).
