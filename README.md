# 🧪 Proyecto predicción de ventas en Rossman


## ❓ Planteamiento del proyecto
Junto a la gran explotación de creación de empresas, surge una gran problemática. El preguntarse "¿Cuánto venderé el próximo tiempo?", "¿Qué factores me ayudan o perjudican en mis ventas?. La incertidumbre y muchas veces la poca capacidad de saber hacia donde apuntar (y que elementos mejorar) afecta de gran manera a las empresas de todo el mundo, las cuales cada una por si sola tiene comportamientos diferentes y por supuesto, factores que la afectan de diferente manera. Es por ello que nace la necesidad de realizar análisis rigurosos que permitan mejorar el futuro de las empresas. Éste proyecto tiene por objetivo identificar los factores que más afectan en las ventas (Sales) de el dataset de Rossman, en conjunto de modelos de Machine Learning que tengan la capacidad de predecir las ventas de la manera más precisa posible.


## ℹ️ Dataset
Recurso: Kaggle

Registros:
- Stores (1115): Representa registros de datos de cada tienda, señalando el número de la tienda, a que distancia está su competidor y otras características propias de la tienda.

- Train (1017209): Cada registro representa un día en específico en una tienda en específico, junto a su total de ventas, si era feriado, había promo, etc.


## 🎯 Objetivos
- Encontrar features que expliquen el comportamiento de las ventas
- Generar modelos de ML capaces de realizar predicciones


## 🧹 Data cleaning
- Existen tiendas en "Stores" que no tienen CompetitionDistance, las cuales tales tiendas aparecen pocas veces dentro de "Train", con un total de 2642 registros, al ser tan pocos se opta por simplemente eliminarlos.

- CompetitionOpenSinceMonth y CompetitionOpenSinceYear contienen nulos, pero estos serán imputados con mediana futuramente, ya que para hacerlo en este momento se requiere de separar y decidir desde ya cuales datos serán de entrenamiento y pruebas, se dejará para una próxima etapa.

- Nulos en Promo2SinceWeek y Promo2SinceYear se rellenan con 0's ya que estos valores faltantes representan que no hay promo.

- Nulos en PromoInterval se rellenan con el string "NoPromo" señalando que no existe promoción en ese registro.

- Convertir feature "Date" a datetime.

- StateHoliday figura como tipo de dato "object", por lo que nos encontramos que el feature tiene tipos de datos mezclados debido a que existen registros con 0 y otros con "0". Por lo que se opta convertir toda la serie en un str, unificando los 0's como strings simplemente.


## 🔀 Data integration
Combinar ambos dataframes en razón del número de la Store, esto se hace justo después de guardar la data "limpia" de forma separada. Este merge lo hacemos simplemente para realizar análisis de datos.


## 📊 EDA
Los principales y más importantes hallazgos en este dataset son los siguientes:

1. Si bien las ventas a través del tiempo de forma total no tienen una característica de crecimiento evidente, en el gráfico se puede apreciar ciertos patrones que indican un comportamiento usual. Con tiempos de grandes picos de ventas y otros con ventas cercanas al 0.

![Total sales across the time](/outputs/figures/01_total_sales_across_time.png)


2. Para realizar un análisis más profundo del punto anterior, se ha optado por tomar como muestra Junio de los 3 años de los que se tiene información, siendo este un tiempo que indica mitades de año. Aquí se descubre que la tendencia de los picos y bajas del gráfico anterior se explican de la siguiente manera: Las ventas tienden a subir fuertemente todos los días Lunes, además los Martes siguen tal tendencia pero con una ligera baja, aumentando nuevamente los días Viernes y quedando practicamente en 0 los días Domingos. Junto a esto además se descubre que la 1ra y 3ra semana de cada mes las ventas tienden a ser más altas, mientras que la 2da y 4ta son más bajas.

![June Sales 2013](/outputs/figures/02_june_sales_day_names_2013.png)
![June Sales 2014](/outputs/figures/03_june_sales_day_names_2014.png)
![June Sales 2015](/outputs/figures/04_june_sales_day_names_2015.png)


3. En el siguiente gráfico podemos ver porqué es que los Domingos se vende mucho menos. Se puede apreciar que es debido a que durante todos los años los Domingos casi siempre están cerrados, además también podemos encontrar una posible explicación de porqué los Jueves también se tiende a vender un poco menos, se puede ver que las tiendas abren un poco menos que el resto de los días de la semana.

![Open Stores](/outputs/figures/09_count_of_open_stores.png)


4. A pesar de que los Domingos sean los días con menos ventas, el siguiente gráfico demuestra que considerando solo los días de apertura, los Domingos tienen ventas equiparables en promedio con los días Lunes, encontrando una posibilidad de capacidad de decisión en ello.

![Mean Sales Of Week Opens](/outputs/figures/10_sales_mean_x_dayofweek.png)


5. En este punto podemos ver uno de los indicadores más fuertes del aumento de ventas, en promedio los días que hay Promo aumentan claramente los "Sales".

![Promo sales mean](/outputs/figures/11_promo_sales_mean.png)


6. Las ventas se ven fuertemente afectadas durante los StateHoliday's, no así las ventas durante los SchoolHoliday's, e incluso las ventas tienden a ser mayores cuando hay SchoolHoliday

![Sales mean School/State Holidays](/outputs/figures/12_sales_mean_of_state_and_school_holidays.png)


7. A pesar de que las tiendas con Assortment de tipo "b" son las que menos hay, también son las que tienen las ventas más altas, justamente al contrario de lo que ocurre con las tiendas de tipo "a".

![Sales mean by assortment](/outputs/figures/13_sales_mean_by_assortment.png)


*Extra: Otros gráficos pueden encontrarse dentro de la carpeta outputs/figures


## ⚙️ Feature engineering
### Preprocessing
Antes de continuar con el feature engineering es necesario realizar un preprocesamiento de la información, esto se debe a que los datos que ya están limpios aún les falta ser imputados, ser mergeados, etc. Por ello primero procederemos con lo siguiente:

1. Hacer merge de los datos limpieados anteriormente según "Store".

2. Aparecieron nulos que tienen que ver con la información de la Store, esto es debido a que habían stores dentro del train que no estaban registradas en los datos de "Store", por lo que estos registros que han quedado nulo son eliminados.

3. Volver a convertir feature "Date" a datetime, debido a que en el guardado el dataframe pierde su característica de tipo de dato.

4. Anteriormente habíamos dejado los nulos de CompetitionOpenSinceMonth y CompetitionOpenSinceYear. Ahora se procede a decidir donde se hará la separación de los datos para entrenamiento y pruebas. Por el tiempo del dataset se opta por dejar todo 2013 y 2014 para entrenamiento y 2015 para pruebas, ya que éste solo cuenta con los meses hasta Julio. Dentro de los registros existentes en 2013 y 2014 para ambos features con nulos mencionados, se decide obtener su mediana y aplicarla tanto para rellenar los datos de los mismos años como para rellenar los valores faltantes en el año 2015.

5. Se ha decidido eliminar los registros que contaban con ventas en 0, esto pensando en que casi todos los días que tiene Sales en 0, son días que están cerrados, siendo algo lógico en el negocio. Además, tener ventas en 0 podría afectar a la evaluación por porcentajes futuras de los algoritmos de Machine Learning.

### Comienzo feature engineering

1. Primero creamos features de tiempo que son extraidos a partir de el feature "Date", creando así features como lo son el año del registro, el mes, día y la semana dentro del año.

2. Se elimina el feature customers, que si bien es un feature valioso para predecir las ventas, también es un feature del que no podemos tener certeza de cuantos habrá en el futuro.

3. Creamos features de lag's, que indican las ventas de 1, 7, 14 y 30 días antes del registro actual.

4. Features de rolling mean, añadiendo features que representan el promedio de ventas de 7, 14 y 30 días antes.

5. Creación de rolling standard, el cual representa la volatilidad de los "Sales" de 3, 7, 14 y 30 días antes del registro actual.

6. Finalmente se quitan los nulos generados y se guarda el dataframe completo procesado.

* Es necesario que se había intentado generar un feature el cual indica la cantidad de meses exactos que han pasado hasta que se realiza la observación en el registro. Pero este no ha generado mejorar en los modelos, además de afectar negativamente a su velocidad de aprendizaje.


## 🤖 Machine Learning
Para el trabajo con Machine Learning se ha optado por utilizar 3 modelos, los cuales son los siguientes:

- LinearRegression (Baseline)

- RandomForestRegressor

- XGBRegressor

* Primero cabe aclarar que luego de importar la información, separar en train y test, es necesario generar dummies de cada X y alinearlos para no generar problemas por diferencias de features.

* También se opta por medir los modelos segun MAPE (mean absolute percentage error), de esta manera sabremos porcentualmente cuanto es que se equivocan los modelos.

* Además cabe aclarar que los modelos serán evaluados tanto en sus datos de entrenamiento como en sus pruebas para detectar si existe overfitting.

### LinearRegression V1
Modelo completamente sin tuning (ya que no existen muchos hiperparámetros que tocar). ES importante señalar que LinearRegression solo tendrá su versión 1.

Mape LinearRegression V1 train: 16.11%

Mape LinearRegression V1 test: 15.13%


### RandomForestRegressor V1
Modelo casi completamente sin tuning, el único hiperparámetro que se ha movido es n_jobs=-1, con la intención de utilizar más recursos de la máquina desde donde se ejecuta el proyecto.

Mape RandomForestRegressor V1 train: 3.12%

Mape RandomForestRegressor V1 test: 9.80%


### XGBRegressor V1
Modelo sin tuning.

Mape xgboost V1 train: 8.65%

Mape xgboost V1 test: 10.47%


### RandomForestRegressor V2
Modelo con tuning, ajustando hiperparámetros en:
- n_estimators
- max_depth
- min_samples_leaf

Mape RandomForestRegressor V2 train: 7.27%

Mape RandomForestRegressor V2 test: 10.00%


### XGBRegressor V2
Modelo con tuning, ajustando hiperparámetros en:
- learning_rate
- n_estimators
- max_depth
- subsample
- colsample_bytree

Mape xgboost V2 train: 5.08%

Mape xgboost V2 test: 9.94%

## 🏆 Conclusiones y recomendaciones finales
Para empezar con esta etapa final del proyecto podemos determinar características más importantes en el comportamiento de las ventas, las cuales responden a posibles tendencias humanas de compra, las cuales son las siguientes:

1. Los usuarios tienden a comprar más durante los Lunes, Martes y Viernes. Aunque se recomienda evaluar la posibilidad de realizar ventas los Domingos de manera más seguida, esto debido a su alta venta promedio los días que si abren.

2. Las ventas y por ende los clientes se ven altamente influenciados a generar mayores ganancias cuando existe Promo. Esto se podría combinar con la idea de ejecutar las promos dentro de la primera y tercera semana del mes.

3. Los feriados de SchoolHoliday y StateHoliday se ven afectados por las ventas, pero de diferente manera. Cuando existe un StateHoliday se puede esperar una clara baja en las ventas, mientras que no importa mucho si hay un SchoolHoliday o no, de hecho, son incluso beneficiosos para la empresa los SchoolHolidays.

4. Las tiendas con Assortment de tipo "b" son las menos existentes, pero combiene plantear la idea de crear más de estas tiendas ya que son las que más ganancias generar en promedio.

Además, se concluye que los modelos generados de Machine Learning cumplen con la función de lograr realizar predicciones en las ventas, con un margen de error del 10% aproximadamente. El cual para efectos de este ejercicio se considera un buen desempeño.



## Versión recomendada de python para el kernel:
- 3.11

## 👤 Autor
Carlos Rojas Villegas

## Adicional
- Para la ejecución y prueba del proyecto se recomienda la siguiente estructura de carpeta para los datos:


proyecto/

│

├── data/

│   ├── raw/          # Datos originales

│   ├── clean/        # Datos limpios

│   └── processed/    # Datos transformados para ML

│

├── notebooks/        # Jupyter Notebooks

│

│
...
