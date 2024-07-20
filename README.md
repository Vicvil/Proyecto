# RESERVA DE INCOBRABLES

SE TRATABA DE CREAR UN MÉTODO PARA REALIZAR EL CÁLCULO DE LA RESERVA DE INCOBRABLES POR LA METODOLOGÍA ROLLRATES, EN UN INICIO SE ME ASIGNÓ LA CREACIÓN DE UN MODELO PARA ESTIMAR LA PROPENSIÓN DE PAGO EN EL SECTOR SALUD, CUYO OBJETIVO ERA DESARROLLAR UN MODELO DE APRENDIZAJE AUTOMÁTICO QUE PUEDIERA PREDECIR LA PROPENSIÓN DE LOS PACIENTES A PAGAR SUS FACTURAS MÉDICAS EN EL SECTOR DE LA SALUD.
    
REALICÉ UN DIAGRAMA DE FLUJO QUE PERMITIERA SEGUIR EL ORDEN DE LOS PASOS A SEGUIR ACORDE CON DATOS MÁS CERCADOS A MI TRABAJO, EL OBJETIVO ERA LLEGAR A CALCULAR LA MATRIZ DE COSECHA QUE PERMITIERA REALIZAR LOS CÁLCULOS PARA ESTIMAR LA PROBABILIDAD DE INCOBRABILIDAD Y ASU VEZ LA PÉRDIDA ESPERADA (ECL)

[DIAGRAMA DE FLUJO.pdf](https://github.com/user-attachments/files/16322209/DIAGRAMA.DE.FLUJO.pdf)

![MATRIZ DE SALDOS](https://github.com/user-attachments/assets/7330c784-6377-49df-b998-91cd822b7203)

![MATRIZ DE TASAS](https://github.com/user-attachments/assets/00068c3f-0e9c-46cf-b609-33196d42e90f)

![TABLA DE CALCULOS ECL](https://github.com/user-attachments/assets/28aa7956-4275-49ec-91da-ee77f52e8166)

EL RETO MÁS IMPORTANTE FUERON LOS CÁLCULOS MANUALES QUE SE DEBÍAN REALIZAR POR LO QUE NO FUE POSIBLE DESARROLLAR ESTE PROYECTO Y SE ME ENCOMENDÓ REGRESAR AL PROYECTO ORIGINAL:

**DESCRIPCIÓN DEL PROBLEMA:** EN EL SECTOR DE LA SALUD, ES FUNDAMENTAL PARA LOS HOSPITALES Y PROVEEDORES DE ATENCIÓN MÉDICA GARANTIZAR QUE SE PAGUEN LAS FACTURAS DE LOS SERVICIOS PRESTADOS. SIN EMBARGO, ALGUNOS PACIENTES PUEDEN ENFRENTAR DIFICULTADES FINANCIERAS O TENER UN HISTORIAL DE PAGOS IRREGULAR, LO QUE PUEDE RESULTAR EN FACTURAS IMPAGAS O RETRASADAS. IDENTIFICAR A LOS PACIENTES QUE TIENEN UNA ALTA PROBABILIDAD DE PAGAR SUS FACTURAS PUEDE AYUDAR A PRIORIZAR LOS ESFUERZOS DE COBRO Y GESTIONAR MEJOR LOS RECURSOS FINANCIEROS DE LA INSTITUCIÓN MÉDICA.

**DATOS DISPONIBLES:** SE DISPONE DE UN CONJUNTO DE DATOS QUE INCLUYE INFORMACIÓN SOBRE DIFERENTES ASPECTOS DE LOS PACIENTES Y SU HISTORIAL DE PAGOS. LOS DATOS INCLUYEN:

    ●	ID DEL PACIENTE
    ●	FECHA DE NACIMIENTO
    ●	SEXO
    ●	ESTADO CIVIL
    ●	INGRESOS MENSUALES
    ●	RIESGO DE ENFERMEDAD
    ●	HISTORIAL DE PAGOS
    ●	NÚMERO DE FACTURAS IMPAGAS
    ●	FECHA DEL ÚLTIMO PAGO REALIZADO POR EL PACIENTE
    ●	PLAZO MÁXIMO DE PAGO PERMITIDO
    ●	PROPENSIÓN A PAGAR (VARIABLE OBJETIVO)
    
**METODOLOGÍA:** SE UTILIZARÁ UN ENFOQUE DE APRENDIZAJE AUTOMÁTICO SUPERVISADO PARA DESARROLLAR EL MODELO DE PROPENSIÓN A PAGAR. EL CONJUNTO DE DATOS SE DIVIDIRÁ EN CONJUNTOS DE ENTRENAMIENTO Y PRUEBA PARA ENTRENAR Y EVALUAR EL MODELO. SE PROBARÁN VARIOS ALGORITMOS DE CLASIFICACIÓN, COMO REGRESIÓN LOGÍSTICA, ÁRBOLES DE DECISIÓN, BOSQUES ALEATORIOS Y GRADIENT BOOSTING, PARA DETERMINAR CUÁL PROPORCIONA LAS MEJORES PREDICCIONES.

**MÉTRICAS DE EVALUACIÓN:** SE EVALUARÁ EL RENDIMIENTO DEL MODELO UTILIZANDO MÉTRICAS DE CLASIFICACIÓN, COMO PRECISIÓN, RECALL, F1-SCORE Y ÁREA BAJO LA CURVA ROC (AUC-ROC). ADEMÁS, SE REALIZARÁ UNA VALIDACIÓN CRUZADA Y SE AJUSTARÁN LOS HIPERPARÁMETROS DEL MODELO PARA MEJORAR SU RENDIMIENTO Y GENERALIZACIÓN.

**APLICACIÓN PRÁCTICA:** EL MODELO DESARROLLADO PUEDE APLICARSE EN ENTORNOS DE ATENCIÓN MÉDICA PARA IDENTIFICAR A LOS PACIENTES CON MAYOR RIESGO DE INCUMPLIMIENTO DE PAGO Y TOMAR MEDIDAS PROACTIVAS PARA ABORDAR SUS NECESIDADES FINANCIERAS. ESTO PUEDE AYUDAR A REDUCIR LOS COSTOS ASOCIADOS CON LAS CUENTAS INCOBRABLES Y MEJORAR LA EFICIENCIA EN LA GESTIÓN DE INGRESOS DE LAS INSTITUCIONES MÉDICAS.

SE UTILIZÓ LA BASE DE DATOS ANEXA.

PARA REALIZAR EL PREPROSESAMIENTO DE LOS DATOS UTILIZAMOS EL SIGUIENTE CÓDIGO


# 1. Importa las paqueterias
        import pandas as pd
        import numpy as np
        import sys
        import io

# 2. Importamos el conjunto de datos
        df= pd.read_csv("datasets/propensity_to_pay_healthcare.csv")

# 2.1 Vemos las dimensiones del dataframe
        rows = df.shape[0]
        cols = df.shape[1]
        print(f"El dataframe tiene {rows} filas y {cols} columnas")

# 2.2 Reporte de las columnas
        df.info()

# Crear un objeto StringIO
        buffer = io.StringIO()

# Obtener la información del DataFrame y escribirla en el buffer
        df.info(buf=buffer)

# Obtener el contenido del buffer como una cadena de texto
        info_str = buffer.getvalue()


        with open('resultados_preprocesamiento/dataframe_info.txt', 'w') as f:
    
# Escribimos la salida del buffe
        f.write(info_str)

# 2.3 Visualización de los primeros n registros del dataset

# Número de filas a visualizar
        n_filas = 10

# Abrir el archivo en modo escritura
        with open('resultados_preprocesamiento/dataframe_registros.txt', 'w') as f:
        # Convertir los primeros n registros a una cadena de texto
        registros_str = df.head(n_filas).to_string()
        # Escribir la cadena en el archivo
        f.write(registros_str)

# 2.4 Procesamiento de Columnas y filas con valores NA
# Elimino columnas si la mitad de sus registros son NA
# Para cada una de las columnas
        for columna in df.columns:
        
# Si la cantidad de NA es mayor a la mitad de registros
        if df[columna].isnull().sum() > len(df)/2:
        
# Elimina dicha columna
        df.drop(columns=[columna], inplace=True)

# Si hay presencia de valores NA 
# Opción 1: Elimino todas las filas que tengan algún Na en sus columnas
        #df.dropna(inplace=True)

# Opción 2: Reemplazo los valores NA por algun tipo de media
# Para esto en columnas numericas, puedes darle un listado
# de las columnas que te interesa sustituir
        columnas_numericas =  list(df.select_dtypes(include=['number']).columns)
        columnas_numericas.remove("ID Paciente")
        sustituir_na ="media"


        for columna in columnas_numericas:
    if sustituir_na == 'moda':
        df[columna].fillna(df[columna].mode(), inplace=True)
    elif sustituir_na == 'media':
        df[columna].fillna(df[columna].mean(), inplace=True)
    elif sustituir_na == 'mediana':
        df[columna].fillna(df[columna].median(), inplace=True)
    elif sustituir_na == 'media_recortada':
    
# Calcular la media recortada al 10%
        recortado = df[columna].quantile(0.1), df[columna].quantile(0.9)
        df[columna].fillna(df[columna].clip(*recortado).mean(), inplace=True)
    elif sustituir_na == 'media_winsorizada':
    
# Calcular la media winsorizada al 10%
        winsorizado = df[columna].quantile(0.1), df[columna].quantile(0.9)
        df[columna].fillna(df[columna].clip(*winsorizado).mean(), inplace=True)
    elif sustituir_na == 'media_huber':
    
# Calcular la media de Huber con un máximo de 100 iteraciones
        from statsmodels.robust.scale import huber
        valores_no_nan = df[columna].copy().dropna()
        # Calcular la media de Huber
        media_huber = huber(valores_no_nan)[0]
        df[columna].fillna(media_huber, inplace=True)

# 2.5 Procesamiento de Columnas con valores atípicos

        quantiles = df[columnas_numericas].quantile([0.1, 0.9])

# Extraer los valores cuantil inferior y superior para cada columna
        inferior = quantiles.loc[0.1]
        superior = quantiles.loc[0.9]


# Vamos a crear un dataframe con las mismas dimensiones del original
mascara = pd.DataFrame(True, index=df.index, columns=columnas_numericas)

# Luego, para cada columna numerica, se verifica si sus entradas

# estan dentro del rango [inferior, superior]
        for col in columnas_numericas:
            mascara[col] = df[col].between(inferior[col], superior[col])

# Combinar las máscaras de todas las columnas 
        mascara_combinada = mascara.all(axis=1)

# Filtrar el DataFrame usando la máscara combinada
        df_filtrado = df[mascara_combinada]

# Comparamos los filtrados 

        with open('resultados_preprocesamiento/dataframe_summary_pre.txt', 'w') as f:

# Convertir los primeros n registros a una cadena de texto
        registros_str = df[columnas_numericas].describe().to_string()

# Escribir la cadena en el archivo
        f.write(registros_str)
    
        with open('resultados_preprocesamiento/dataframe_summary_post.txt', 'w') as f:

# Convertir los primeros n registros a una cadena de texto
        registros_str = df_filtrado[columnas_numericas].describe().to_string()

# Escribir la cadena en el archivo
        f.write(registros_str)
    
# Guardamos el resultado
    df_filtrado.to_csv("resultados_preprocesamiento/train_dataset.csv")

**PARA REALIZAR EL ANALISIS GRÁFICO UTILIZAMOS EL SIGUIENTE CÓDIGO:**

# Importación de bibliotecas
        import os
        import sys
        import io
        import glob

# os: Proporciona una forma de utilizar funcionalidades dependientes del sistema operativo, como leer o escribir en el sistema de archivos.

# sys: Proporciona acceso a algunas variables y funciones que interactúan fuertemente con el intérprete de Python, como sys.argv para recibir argumentos de la línea de comandos.

# io: Proporciona herramientas para trabajar con flujos de entrada y salida, tanto para archivos como para datos en memoria.

# glob: Proporciona una forma de encontrar todos los nombres de ruta que coinciden con un patrón específico, útil para listar archivos con nombres que sigan un patrón.

    import pandas as pd

# pandas: Biblioteca esencial para manipulación y análisis de datos, proporciona estructuras de datos como DataFrame para trabajar con datos tabulares de manera eficiente.

    import matplotlib.pyplot as plt

# matplotlib.pyplot: Biblioteca para crear visualizaciones estáticas, animadas e interactivas en Python. plt es una interfaz similar a MATLAB para facilitar la creación de gráficos.

    import seaborn as sns

# seaborn: Biblioteca basada en matplotlib que proporciona una interfaz de alto nivel para dibujar gráficos estadísticos atractivos y con estilo.

    import numpy as np

# numpy: Biblioteca fundamental para el cálculo científico en Python. Proporciona soporte para arrays y matrices grandes, junto con una colección de funciones matemáticas de alto nivel para operar con estos arrays.

    from sklearn.decomposition import PCA

# sklearn.decomposition.PCA: Implementa el análisis de componentes principales (PCA).

    from sklearn.preprocessing import StandardScaler

# sklearn.preprocessing.StandardScaler: Estandariza las características eliminando la media y escalando a la varianza unitaria.


# Importación del archivo

# Buscar el archivo CSV en la carpeta "datasets"
    csv_files = glob.glob("datasets/*.csv")

# Verificar que hay exactamente un archivo CSV
        if len(csv_files) == 1:
            df = pd.read_csv(csv_files[0])
            print("Archivo CSV cargado exitosamente.")
        else:
            print("Error: No se encontró un archivo CSV único en la carpeta 'datasets'.")

        df.drop(columns="Unnamed: 32", inplace=True)

# Preparación de carpetas para guardar imágenes

# Crear las carpetas si no existen
        if not os.path.exists('imagenes'):
            os.makedirs('imagenes')
        if not os.path.exists('imagenes/var_categoricas'):
            os.makedirs('imagenes/var_categoricas')
        if not os.path.exists('imagenes/var_numericas'):
            os.makedirs('imagenes/var_numericas')

# Separar columnas numéricas y categóricas
        columnas_numericas = df.select_dtypes(include=['number']).columns
        columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns

# Configuración de matplotlib

        %matplotlib inline

# Crear una figura de ejemplo
        plt.figure(figsize=(8, 8))
        plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
        plt.title("Gráfico de ejemplo")
        plt.xlabel("Eje X")
        plt.ylabel("Eje Y")
        plt.show()

        %matplotlib notebook
        df[columnas_numericas[1]].value_counts().plot(kind='hist')
        plt.title(columnas_numericas[1])

# Configuración de tamaño de gráficos

# Tamaño en píxeles
        width_px = 1200
        height_px = 1200

# DPI
        dpi = 100

# Convertir a pulgadas
        width_inch = width_px / dpi
        height_inch = height_px / dpi

# Crear una figura con el tamaño calculado en pulgadas
        plt.figure(figsize=(width_inch, height_inch), dpi=dpi)
        plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
        plt.title("Gráfico de ejemplo")
        plt.xlabel("Eje X")
        plt.ylabel("Eje Y")
        plt.savefig('mi_figura.png')
        plt.show()

# Crear histogramas y gráficos de barras

        %matplotlib inline
        for col in columnas_numericas:
            plt.figure(figsize=(10, 6))
            df[col].dropna().value_counts().plot(kind='hist', edgecolor='k')
            plt.title(f'Histograma de {col}')
            plt.xlabel(col)
            plt.ylabel('Frecuencia')
            plt.tight_layout()
            plt.savefig(f'imagenes/var_numericas/histograma_{col}.png')
            plt.close()

        for col in columnas_categoricas:
            plt.figure(figsize=(10, 6))
            df[col].dropna().value_counts().plot(kind='bar', edgecolor='k')
            plt.title(f'Gráfico de barras de {col}')
            plt.xlabel(col)
            plt.ylabel('Frecuencia')
            plt.tight_layout()
            plt.savefig(f'imagenes/var_categoricas/barra_{col}.png')
            plt.close()

# Gráfico de valores NA

        cuenta_faltantes = df.isnull().sum()
        total_valores = df.isnull().count()
        porcentaje_faltantes = round((cuenta_faltantes / total_valores) * 100, 2)
        
        faltantes_df = pd.DataFrame({'cuenta': cuenta_faltantes, 'porcentaje': porcentaje_faltantes})

# Crear la carpeta "resultados_analisis_estadistico" si no existe
        if not os.path.exists('resultados_analisis_estadistico'):
            os.makedirs('resultados_analisis_estadistico')

# Guardar resultados de valores NA
        buffer = io.StringIO()
        sys.stdout = buffer
        print(faltantes_df)
        sys.stdout = sys.__stdout__
        contenido = buffer.getvalue()
        with open('resultados_analisis_estadistico/porcentaje_NA_por_columnas.txt', 'w') as archivo:
            archivo.write(contenido)
        buffer.close()

# Gráfico de barras horizontal para el porcentaje de valores faltantes

        %matplotlib notebook
        if not os.path.exists('imagenes/grafica_na'):
            os.makedirs('imagenes/grafica_na')
        
        barchart = faltantes_df.plot.barh(y='porcentaje')
        plt.yticks(fontsize=12)
        for indice, porcentaje in enumerate(porcentaje_faltantes):
            barchart.text(porcentaje + 4, indice, str(porcentaje) + "%", ha='right', va='center')
        
        plt.title("Porcentaje de valores NA desglosado por columnas", fontsize=18)
        plt.xlabel("Porcentaje", fontsize=12)
        plt.ylabel("Columnas", fontsize=12)
        plt.savefig('imagenes/grafica_na/grafico_valores_faltantes.png')
        
        #%% Comparaciones con Seaborn
        
        df_num = df[columnas_numericas]
        df_num.drop(columns="ID Paciente", inplace=True)
        
        if not os.path.exists('imagenes/pairplot'):
            os.makedirs('imagenes/pairplot')

# Comparación numérica vs. numérica
        sns.pairplot(df_num)
        plt.savefig('imagenes/pairplot/pairplot_numerical_vs_numerical.png')
        plt.close()

# Convertir columnas numéricas a categóricas

        df_categorico = df.drop(df.index).copy()
        for col in columnas_numericas:
            df_categorico[f'{col}'] = pd.cut(df[col], bins=4, labels=['Bajo', 'Medio', 'Alto', 'Muy Alto'])
        
        df_categorico["Propensión a Pagar"] = df["Propensión a Pagar"].copy()
        
        if not os.path.exists('imagenes/count_plot'):
            os.makedirs('imagenes/count_plot')
        
        sns.countplot(data=df_categorico, 
                      x="Ingresos Mensuales", 
                      hue='Propensión a Pagar')
        plt.savefig('imagenes/count_plot/countplot_categorical_vs_categorical.png')
        plt.close()

# Comparación categórica vs. numérica

        %matplotlib inline
        sns.set(style="whitegrid")
        if not os.path.exists('imagenes/boxplot'):
            os.makedirs('imagenes/boxplot')
        
        for columna in columnas_numericas:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x="Propensión a Pagar", y=columna)
            plt.title(f'Boxplot de "{columna}" para cada valor de "Propensión a Pagar"')
            plt.xlabel('Propensión a Pagar')
            plt.ylabel(columna)
            plt.savefig(f'imagenes/boxplot/boxplot_{columna}_vs_Propensión a Pagar.png')
            plt.close()

# Comparación numérica vs. numérica con hue categórico

        df_num["Propensión a Pagar"] = df["Propensión a Pagar"].copy()

        %matplotlib inline
        sns.set(style="whitegrid")
        sns.pairplot(df_num, hue='Propensión a Pagar')
        plt.savefig('imagenes/pairplot/pairplot_numerical_with_hue.png')
        plt.close()
        
        #%% Heatmap de las columnas numéricas
        
        if not os.path.exists('imagenes/heatmap'):
            os.makedirs('imagenes/heatmap')
        
        df_num.drop(columns="Propensión a Pagar",inplace=True)
        
        correlation_matrix = df_num.corr()
        sns.set(style="white")
        
        plt.figure(figsize=(12, 8))
        heatmap = sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=.5)
        plt.title('Heatmap de Correlación entre Columnas Numéricas')
        plt.savefig('imagenes/heatmap/heatmap_correlacion_numericas.png')
        plt.close()

# Análisis de Componentes Principales (PCA)
        if not os.path.exists('imagenes/PCA'):
            os.makedirs('imagenes/PCA')
            
        scaler = StandardScaler()
        df_num_scaled = scaler.fit_transform(df_num)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_num_scaled)
        
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        pca_df['Propensión a Pagar'] = df['Propensión a Pagar']
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PC1', 
                        y='PC2', 
                        hue='Propensión a Pagar', 
                        data=pca_df,
                        palette='viridis')
        plt.title('PCA de Columnas Numéricas')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.savefig('imagenes/PCA/pca_columnas_numericas.png')
        plt.close()


# Obtener los pesos de los dos primeros componentes
        pca_loadings = pca.components_

# Crear un DataFrame para los pesos
        pca_loadings_df = pd.DataFrame(pca_loadings.T, 
                                       columns=['PC1', 'PC2'], 
                                       index=columnas_numericas.drop("ID Paciente"))
# Añadir una columna con el nombre de las variables
        pca_loadings_df.reset_index(inplace=True)
        pca_loadings_df.rename(columns={'index': 'Variable'}, inplace=True)
        pca_loadings_df.to_csv('imagenes/PCA/pesos_pca.csv', index=False)

**PARA LA REALIZACIÓN DE MODELO DE APRENDIZAJE AUTOMÁTICO, UNA PROPUESTA DE CODIGO ES LA SIGUIENTE**

# Importa las bibliotecas necesarias
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, classification_report

# Cargamos la base
        data= pd.read_csv("datasets/propensity_to_pay_healthcare.csv")

# Calcula la edad a partir de la fecha de nacimiento
        data['Fecha de Nacimiento'] = pd.to_datetime(data['Fecha de Nacimiento'])
        data['Edad'] = (pd.to_datetime('today') - data['Fecha de Nacimiento']).astype('<m8[Y]')
        data.drop(columns=['Fecha de Nacimiento'], inplace=True)

# Codificación one-hot para variables categóricas (sexo, estado civil)
        data = pd.get_dummies(data, columns=['Sexo', 'Estado Civil'], drop_first=True)

# Divide los datos en conjuntos de entrenamiento y prueba
        X = data.drop(columns=['Propensión a Pagar'])
        y = data['Propensión a Pagar']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escala las características numéricas
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

# Crea y entrena el modelo RandomForest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

# Realiza predicciones en el conjunto de prueba
        y_pred = model.predict(X_test_scaled)

# Evalúa el rendimiento del modelo
        print("Exactitud (Accuracy):", accuracy_score(y_test, y_pred))
        print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))


# CONCLUSIONES
EN EL DIPLOMADO APRENDÍ BUENAS BASES PARA GUIAR A MI EQUIPO DE TRABAJO A LO QUE SE PUEDE Y NO HACER CON PYTHON POR LO QUE CONSIDERO EL OBJETIVO LOGRADO, CONSIDERO QUE PYTHON ES UNA HERRAMIENTA CAPAZ DE REALIZAR CASI CUALQUIER COSA, Y EL SER CAPAZ DE COMPRENDER LOS ALACANCES EN LA PRACTICA SIN DUDA ME AYUDARÁ MUCHO EN MI DESARROLLO PROFESIONAL, ES VERDAD QUE HUBO MOMENTOS DE MUCHA FRUSTRACIÓN, SIN EMBARGO, PROGRAMAR COMO EN TODO ES CUESTIÓN DE DEDICARLE HORAS VUELO PARA QUE SE LOGREN LOS RESULTADOS ESPERADOS.

**MUCHAS GRACIAS**





