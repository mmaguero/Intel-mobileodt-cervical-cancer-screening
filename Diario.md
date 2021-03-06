# Diario

## Exploración de datos
Descripción y discusión de las técnicas utilizadas para estudiar la
estructura y la semántica de los datos y los hallazgos preliminares, ası́ como discusión y
justificación de decisiones iniciales sobre el proceso que se llevará a cabo.

Utilizaremos CNNs (Convolutional Neural Networks) y técnicas de clasificación para la predicción del conjunto de imágenes. Antes que nada descargamos las imágenes, hay un conjunto de train y otro de test, también un conjunto extra de train, alrededor de 35 GB en imágenes (2000*3000px aproximadamente, en formaro .jpg), por lo que debemos aplicar técnicas para poder reducirlas. A simple vista, se ven imágenes de fondo verde, otras a color, enfocadas de distintos puntos, por lo que, además de reducirlas, se podría tratar de hacerlas más uniformes, en blanco y negro por ejemplo, utilizar data augmentation o aumento de datos, respetar la relación de aspecto, etc.

Para comprobar si el conjunto de datos es balanceado o no-balanceado, necesitamos saber cuantos registros hay de cada variable de clase. Hasta 1.5 se considera clases balanceadas, poseemos tres tipos de clases:
- Tipo 1, con 250
- Tipo 2, con 781
- Tipo 3, con 450

Ahora si tenemos en cuenta el conjunto total de train, incluyendo extras, tenemos para cada:
- Tipo 1, con 1441
- Tipo 2, con 4348
- Tipo 3, con 2426

Si consideramos sólo las de train originales, vemos que el conjunto no es balanceado (3.12%), si tomamos las además las extras (3.07%), tampoco, aunque se trata de un leve no-balaanceo.

En deep learning las redes neuronales necesitan ser entrenadas con un gran número de imágenes para lograr un rendimiento satisfactorio, y sí el conjunto de datos de imagen original es limitada, es mejor hacer el aumento de datos para aumentar el rendimiento. Hay muchas maneras de hacer el aumento de datos, como horizontally flipping, random crops and color jitterin, o intentar combinaciones de múltiples procesamientos diferentes, por ejemplo, realizar la rotación y escalar al azar al mismo tiempo. Además, se puede tratar de aumentar la saturación y el valor (componentes S y V del espacio de color HSV), incluso propuestas como Fancy PCA, introducida por Alex-Net en 2012: "la Fancy PCA podría capturar aproximadamente una propiedad importante de las imágenes naturales, es decir, que la identidad del objeto es invariable a los cambios en la intensidad y el color de la iluminación" (Xiu-Shen, 2015).

En (Schmidt, 2017) se hace un análisis exploratorio sobre el conjunto original de imágenes, vamos a adaptarlo para tomar también el conjunto total de train.

## Pre-procesamiento
Descripción y discusión de las técnicas de preprocesamiento
utilizadas y análisis crı́tico de su utilidad en el problema.
- [ ] Integración y detección de conflictos e inconsistencias en los datos: valores perdidos,
valores fuera de rango, ruido, etc.
- [ ] Transformaciones: normalización, agregación, generación de caracterı́sticas adiciona-
les, etc.
- [ ] Reducción de datos: técnicas utilizadas para selección de caracterı́sticas, selección
de ejemplos, discretización, agrupación de valores, etc.
- [ ] Aumento de datos: técnicas utilizadas para incrementar la cantidad de datos dispo-
nibles.

Resize de imágenes

## Técnicas de clasificación:
Discusión de las técnicas y herramientas de clasificación empleadas, justificación de su elección. Por ejemplo:
- [ ] Learning from scratch vs fine-tuning
- [ ] Uso de CNNs + OVO
- [ ] Post-procesamiento OVO

Otros:
- [ ] feature maps,
- [ ] ensambles, etc.

## Presentación y discusión de resultados:
Descripción y discusión de las soluciones obte-
nidas, incidiendo en la interpretación de los resultados.

### Análisis comparativo... en caso de utilizar diferentes técnicas y/o parámetros de configuración en diferentes aproximaciones.

## Conclusiones y trabajo futuro:
Breve resumen de las técnicas aplicadas y de los resulta-
dos obtenidos, ası́ como ideas de ...

### Trabajo futuro
... para continuar mejorando las soluciones desarrolladas.

## Listado de soluciones:
Tabla de soluciones, incluyendo una fila por cada solución subida
a Kaggle durante la competición. El número de filas deberá coincidir con el número de
intentos reflejado en la web de la competición.

| Nº Solución | Pre-procesamiento | Algoritmo/Software | % de acierto entrenamiento | % de acierto test (Kaggle) | Posición Ranking       |
|-------------|-------------------|--------------------|----------------------------|----------------------------|------------------------|
|             |                   |                    | 0.8554                     | 0.88509                    | 292                    |
|             |                   |                    | 0.7675                     | 1.30324                    | 300                    |
|             |                   |                    | 0.                         | 0.841                      | stg 1: 242 / stg 2: 75 |
|             |                   |                    | 0.                         | 0.845                      | 261                    |
|             |                   |                    | 0.                         | 0.98                       | *                      |


Posición al cierre de la primera etapa: 160
## Bibliografía
Schmidt, P. (2017, abril). Cervix EDA & Model selection. Recuperado 16 de junio de 2017, a partir de https://www.kaggle.com/philschmidt/cervix-eda-model-selection/
Xiu-Shen, W. (2015, octubre). Must Know Tips/Tricks in Deep Neural Networks (by <a href="http://lamda.nju.edu.cn/weixs/">Xiu-Shen Wei</a>). Recuperado 6 de junio de 2017, a partir de http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
