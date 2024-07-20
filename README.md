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

SE UTILIZÓ LA BASE DE DATOS ANEXA:





PARA REALIZAR EL PREPROSESAMIENTO DE LOS DATOS Y EL ANÁLISIS EXPLORATORIO SE UTILIZÓ EL SIGUIENTE CÓDIGO:

#%%
# 1. Importa las paqueterias
import pandas as pd
import numpy as np
import sys
import io
#%%
# 2. Importamos el conjunto de datos
df= pd.read_csv("datasets/propensity_to_pay_healthcare.csv")
# 2.1 Vemos las dimensiones del dataframe
rows = df.shape[0]
cols = df.shape[1]
print(f"El dataframe tiene {rows} filas y {cols} columnas")
#%%
# 2.2 Reporte de las columnas
df.info()
#%%
# Crear un objeto StringIO
buffer = io.StringIO()

# Obtener la información del DataFrame y escribirla en el buffer
df.info(buf=buffer)

# Obtener el contenido del buffer como una cadena de texto
info_str = buffer.getvalue()


with open('resultados_preprocesamiento/dataframe_info.txt', 'w') as f:
    # Escribimos la salida del buffe
    f.write(info_str)
#%%
# 2.3 Visualización de los primeros n registros del dataset

# Número de filas a visualizar
n_filas = 10

# Abrir el archivo en modo escritura
with open('resultados_preprocesamiento/dataframe_registros.txt', 'w') as f:
    # Convertir los primeros n registros a una cadena de texto
    registros_str = df.head(n_filas).to_string()
    # Escribir la cadena en el archivo
    f.write(registros_str)
#%%#%%
# 2.4 Procesamiento de Columnas y filas con valores NA

# Elimino columnas si la mitad de sus registros son NA
# Para cada una de las columnas
for columna in df.columns:
    # Si la cantidad de NA es mayor a la mitad de registros
    if df[columna].isnull().sum() > len(df)/2:
        # Elimina dicha columna
        df.drop(columns=[columna], inplace=True)
#%%
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


#%% 2.5 Procesamiento de Columnas con valores atípicos

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
#%%
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
    
#%% Guardamos el resultado
df_filtrado.to_csv("resultados_preprocesamiento/train_dataset.csv")





