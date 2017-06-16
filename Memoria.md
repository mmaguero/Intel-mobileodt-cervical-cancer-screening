<!--Portada-->

<div class="portada">


# Práctica 2
# Clasificación de Imágenes
*****

<img src="imgs/ugr.png" alt="Logo UGR" style="width: 200px; height: auto;"/>

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
> Nombre: Marvin Agüero Torales
> Email: <mmaguero@correo.ugr.es>

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
## 2. Preprocesamiento de datos

## 3. Técnicas de clasificación y discusión de resultados


## 4. Conclusiones y trabajos futuros

<!-- Salto de página -->
<div style="page-break-before: always;"></div>

## 5. Listado de soluciones
La siguiente tabla recoge las distintas soluciones presentadas en Kaggle, tengo que mencionar inicialmente que son 11 filas en lugar de 12 a pesar de ser estos mis intentos en Kaggle. Esto se debe a que he subido la solución 3 dos veces ya que se produjo un error durante la subida y lo volví a subir, por esto no la menciono en la tabla. Respecto a las posiciones del ranking son algo aproximadas ya que seleccionando una entrega como solución final no varia el ranking de Kaggle, por lo que he aproximado a las posiciones ocupadas por puntuaciones idénticas. Como software utilizado para todos los intentos se ha utilizado RStudio y los paquetes y funciones indicadas en la lista de abreviaturas.
La siguiente lista de abreviaturas por orden alfabético recoge los preprocesamientos y algoritmos utilizados para las distintas soluciones:
-


| Nº Solución | Pre-procesamiento | Algoritmo/Software | % de acierto entrenamiento | % de acierto test (Kaggle) | Posición Ranking       |
|-------------|-------------------|--------------------|----------------------------|----------------------------|------------------------|
|             |                   |                    | 0.8554                     | 0.88509                    | 292                    |
|             |                   |                    | 0.7675                     | 1.30324                    | 300                    |
|             |                   |                    | **0.**                         | **0.841**                      | **stg 1: 242 / stg 2: 75** |
|             |                   |                    | 0.                         | 0.845                      | 261                    |
|             |                   |                    | 0.                         | 0.98                       | *                      |



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
