import numpy as np
from scipy import interpolate

def search_train_test(datos, data_lbl):
    # Nos devuelve un vector con todos los tramos menos el ultimo año (solo para aquellos tramos que tenga mas de dos vectores)
    # y otro solo con el ultimo año 
    etiquetas_tramos = np.unique(datos[:, 0])
    last_indexes_year = []
    test_data = []
    train_data_lbl = []

    for lbl in etiquetas_tramos:
        current_lbl_indexes = datos[:, 0] == lbl
        # Nos devuelve en True aquellas filas del tramo actual del "for"

        # Buscamos las filas donde aparece el 1ero. y ultimo registro de cada tramo
        # la vamos a usar para eliminar registros si hubo mejoras
        indice_minimo_tramo = np.min(np.argwhere(datos[:, 0] == lbl))
        indice_maximo_tramo = np.max(np.argwhere(datos[:, 0] == lbl))

        #if indice_maximo_tramo - indice_minimo_tramo > 1:
        # last_indexes_year es un array de los indices que contienen el ultimo año
        # datosTest es el vector con los datos del ultimo año
        last_indexes_year.append(np.arange(indice_maximo_tramo, indice_maximo_tramo+1))
        test_data.append(datos[indice_maximo_tramo])

    ####################################
    # creamos el array propiamente dicho
    last_indexes_year = [item for sublist in last_indexes_year for item in sublist]    
    last_indexes_year = list(set(last_indexes_year))  # eliminamos indices duplicados

    # creamos una lista de booleanos a True
    complete_indexes = np.ones(datos.shape[0], dtype=bool)
    
    # reemplazamos por los que tenemos que eliminar (False)
    complete_indexes[np.array(last_indexes_year)] = False
    
    # Limpio, tramo removido
    train_data = datos[complete_indexes, 0:]
    test_data = np.array(test_data)
    train_data_lbl = data_lbl[complete_indexes].copy()
    
    return train_data, test_data, train_data_lbl



def search_test_data_lbl(datos):
    # Nos devuelve un vector con todos los tramos menos el ultimo año (solo para aquellos tramos que tenga mas de dos vectores)
    # y otro solo con el ultimo año 

    etiquetas_tramos = np.unique(datos[:, 0])
    last_indexes_year = []
    test_data = []

    for lbl in etiquetas_tramos:
        current_lbl_indexes = datos[:, 0] == lbl
        # Nos devuelve en True aquellas filas del tramo actual del "for"

        # Buscamos las filas donde aparece el 1ero. y ultimo registro de cada tramo
        # la vamos a usar para eliminar registros si hubo mejoras
        indice_minimo_tramo = np.min(np.argwhere(datos[:, 0] == lbl))
        indice_maximo_tramo = np.max(np.argwhere(datos[:, 0] == lbl))

        #if indice_maximo_tramo - indice_minimo_tramo > 1:
        test_data.append(datos[indice_maximo_tramo])

    ####################################
    # Limpio, tramo removido
    test_data = np.array(test_data)
    
    return test_data


def search_data(datos, lbl):
    last_indexes_year = []
    data_paint = []
    yPredict_draw = []

    current_lbl_indexes = datos[:, 0] == lbl
    # Nos devuelve en True aquellas filas del tramo actual del "for"

    # Buscamos las filas donde aparece el 1ero. y ultimo registro de cada tramo
    # la vamos a usar para eliminar registros si hubo mejoras
    indice_minimo_tramo = np.min(np.argwhere(datos[:, 0] == lbl))
    indice_maximo_tramo = np.max(np.argwhere(datos[:, 0] == lbl))

    # last_indexes_year es un array de los indices que contienen el ultimo año
    # datosTest es el vector con los datos del ultimo año
    last_indexes_year.append(np.arange(indice_minimo_tramo, indice_maximo_tramo + 1))

    ####################################
    # creamos el array propiamente dicho
    last_indexes_year = [item for sublist in last_indexes_year for item in sublist]    
    last_indexes_year = list(set(last_indexes_year))  # eliminamos indices duplicados
    data_paint = datos[last_indexes_year, : 6]
    indice = np.lexsort((data_paint[:, 1], data_paint[:, 0]))
    data_paint = data_paint[indice]    
    yPredict_draw = [None] * (indice_maximo_tramo - indice_minimo_tramo+1)
    
    return data_paint, yPredict_draw

def search_anio(datos, data_original):
    # Nos devuelve un vector con todos los tramos menos el ultimo año (solo para aquellos tramos que tenga mas de dos vectores)
    # y otro solo con el ultimo año 

    etiquetas_tramos = np.unique(datos[:, 0])
    last_indexes_year = []
    test_data = []
    data_paint = []

    for lbl in etiquetas_tramos:
        current_lbl_indexes = datos[:, 0] == lbl
        # Nos devuelve en True aquellas filas del tramo actual del "for"

        # Buscamos las filas donde aparece el 1ero. y ultimo registro de cada tramo
        # la vamos a usar para eliminar registros si hubo mejoras
        indice_minimo_tramo = np.min(np.argwhere(datos[:, 0] == lbl))
        indice_maximo_tramo = np.max(np.argwhere(datos[:, 0] == lbl))

        if indice_maximo_tramo - indice_minimo_tramo > 1:
            # last_indexes_year es un array de los indices que contienen el ultimo año
            # datosTest es el vector con los datos del ultimo año
            last_indexes_year.append(np.arange(indice_maximo_tramo, indice_maximo_tramo+1))
            test_data.append(datos[indice_maximo_tramo])
            data_paint.append(data_original[indice_maximo_tramo])

    ####################################
    # creamos el array propiamente dicho
    last_indexes_year = [item for sublist in last_indexes_year for item in sublist]    
    last_indexes_year = list(set(last_indexes_year))  # eliminamos indices duplicados

    # creamos una lista de booleanos a True
    complete_indexes = np.ones(datos.shape[0], dtype=bool)
    
    # reemplazamos por los que tenemos que eliminar (False)
    complete_indexes[np.array(last_indexes_year)] = False
    
    # Limpio, tramo removido
    train_data = datos[complete_indexes, 0:]
    test_data = np.array(test_data)
    data_paint = np.array(data_paint)
    
    return train_data, test_data, data_paint


def search_test_data(datos, datos_lbl, tramo, i, fis, ahue):
    # Nos devuelve un vector con todos los tramos menos el ultimo año (solo para aquellos tramos que tenga mas de dos vectores)
    # y otro solo con el ultimo año 
    etiquetas_tramos = np.unique(datos[:, 0])
    indice_maximo_tramo = np.max(np.argwhere(datos[:, 0] == tramo))
    test_data = []

    anio = int(datos[indice_maximo_tramo, 1] + 1)
    deflex = int(datos[indice_maximo_tramo, 2])
    nAnio = round(float(datos[indice_maximo_tramo, 3] * (1.02 ** i)), 3)
    iri = float(datos_lbl[indice_maximo_tramo])
    
    test_data = [anio, deflex, nAnio, fis, ahue, iri]
    test_data = np.array(test_data)
    
    return test_data

def search_test_dataFis(datos, datos_lbl, tramo, i):
    
    # Nos devuelve un vector con todos los tramos menos el ultimo año (solo para aquellos tramos que tenga mas de dos vectores)
    # y otro solo con el ultimo año 
    etiquetas_tramos = np.unique(datos[:, 0])
    indice_maximo_tramo = np.max(np.argwhere(datos[:, 0] == tramo))
    test_data = []

    anio = int(datos[indice_maximo_tramo, 1] + 1)
    deflex = int(datos[indice_maximo_tramo, 2])
    nAnio = round(float(datos[indice_maximo_tramo, 3] * (1.02 ** i)), 3)
    fis = float(datos_lbl[indice_maximo_tramo])
    
    test_data = [anio, deflex, nAnio, fis]
    test_data = np.array(test_data)
    
    return test_data

def search_test_dataAhue(datos, datos_lbl, tramo, i, fis):
    
    # Nos devuelve un vector con todos los tramos menos el ultimo año (solo para aquellos tramos que tenga mas de dos vectores)
    # y otro solo con el ultimo año 
    etiquetas_tramos = np.unique(datos[:, 0])
    indice_maximo_tramo = np.max(np.argwhere(datos[:, 0] == tramo))
    test_data = []

    anio = int(datos[indice_maximo_tramo, 1] + 1)
    deflex = int(datos[indice_maximo_tramo, 2])
    nAnio = round(float(datos[indice_maximo_tramo, 3] * (1.02 ** i)), 3)
    ahue = float(datos_lbl[indice_maximo_tramo])
    
    test_data = [anio, deflex, nAnio, fis, ahue]
    test_data = np.array(test_data)
    
    return test_data

def search_dataFull(datos, datos_lbl, lbl):
    last_indexes_year = []
    data_paint = []
    yPredict_draw = []

    current_lbl_indexes = datos[:, 0] == lbl
    # Nos devuelve en True aquellas filas del tramo actual del "for"

    # Buscamos las filas donde aparece el 1ero. y ultimo registro de cada tramo
    # la vamos a usar para eliminar registros si hubo mejoras
    indice_minimo_tramo = np.min(np.argwhere(datos[:, 0] == lbl))
    indice_maximo_tramo = np.max(np.argwhere(datos[:, 0] == lbl))

    # last_indexes_year es un array de los indices que contienen el ultimo año
    # datosTest es el vector con los datos del ultimo año
    last_indexes_year.append(np.arange(indice_minimo_tramo, indice_maximo_tramo + 1))

    ####################################
    # creamos el array propiamente dicho
    last_indexes_year = [item for sublist in last_indexes_year for item in sublist]    
    #last_indexes_year = list(set(last_indexes_year))  # eliminamos indices duplicados
    data_paint = datos_lbl[last_indexes_year]
    yPredict_draw = [None] * (indice_maximo_tramo - indice_minimo_tramo+1)
    
    return data_paint, yPredict_draw


def search_test_dataFisFirts(datos, datos_lbl, tramo, i):
    
    # Nos devuelve un vector con todos los tramos menos el ultimo año (solo para aquellos tramos que tenga mas de dos vectores)
    # y otro solo con el ultimo año 
    etiquetas_tramos = np.unique(datos[:, 0])
    indice_maximo_tramo = np.min(np.argwhere(datos[:, 0] == tramo))
    test_data = []

    anio = int(datos[indice_maximo_tramo, 1] + 1)
    deflex = int(datos[indice_maximo_tramo, 2])
    nAnio = round(float(datos[indice_maximo_tramo, 3] * (1.02 ** i)), 3)
    fis = float(datos_lbl[indice_maximo_tramo])
    
    test_data = [anio, deflex, nAnio, fis]
    test_data = np.array(test_data)
    
    return test_data

def search_test_dataAhueFirts(datos, datos_lbl, tramo, i, fis):
    
    # Nos devuelve un vector con todos los tramos menos el ultimo año (solo para aquellos tramos que tenga mas de dos vectores)
    # y otro solo con el ultimo año 
    etiquetas_tramos = np.unique(datos[:, 0])
    indice_maximo_tramo = np.min(np.argwhere(datos[:, 0] == tramo))
    test_data = []

    anio = int(datos[indice_maximo_tramo, 1] + 1)
    deflex = int(datos[indice_maximo_tramo, 2])
    nAnio = round(float(datos[indice_maximo_tramo, 3] * (1.02 ** i)), 3)
    ahue = float(datos_lbl[indice_maximo_tramo])
    
    test_data = [anio, deflex, nAnio, fis, ahue]
    test_data = np.array(test_data)
    
    return test_data

def search_test_dataFirts(datos, datos_lbl, tramo, i, fis, ahue):
    # Nos devuelve un vector con todos los tramos menos el ultimo año (solo para aquellos tramos que tenga mas de dos vectores)
    # y otro solo con el ultimo año 
    etiquetas_tramos = np.unique(datos[:, 0])
    indice_maximo_tramo = np.min(np.argwhere(datos[:, 0] == tramo))
    test_data = []

    anio = int(datos[indice_maximo_tramo, 1] + 1)
    deflex = int(datos[indice_maximo_tramo, 2])
    nAnio = round(float(datos[indice_maximo_tramo, 3] * (1.02 ** i)), 3)
    iri = float(datos_lbl[indice_maximo_tramo])
    
    test_data = [anio, deflex, nAnio, fis, ahue, iri]
    test_data = np.array(test_data)
    
    return test_data