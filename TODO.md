# TODO list
- [x] Implementar función de evaluación logloss según se describe [aquí](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening#evaluation), **@AythaE**: según he estado buscando esta funcion es conocida como _categorical_crossentropy_, cuya implementación está disponble en keras, pero requiere strings, por eso se define la _sparse_categorical_crossentropy_, que es la función utilizada.
- [ ] Lanzar modelo de learning from scratch un numero elevado de epocas para obtener el mejor resultado posible
- [ ] Empezar fine-tuning con los modelos de keras
- [ ] Realizar [EDA](https://www.kaggle.com/philschmidt/cervix-eda-model-selection)
- [ ] Probar eliminando imagenes incorrectas según se comenta en este [kernel](https://www.kaggle.com/deveaup/checking-bounding-boxes-and-additional-dataset/notebook/notebook) y en [este](https://www.kaggle.com/chiszpanski/non-cervix-images)
- [ ] Probar más data augmentatión en Keras usando [ImagePreprocessing](https://keras.io/preprocessing/image/#imagedatagenerator)
- [ ] Comparar arquitectura red buena preentrenada con inicialización aleatoria
