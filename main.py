# Vamos a predecir año a año la rugosidad, pasando el predicho para entrenar (año a año)
# Partimos del analisis_1_1.py

import os
import sys
import argparse
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#from sklearn.grid_search import GridSearchCV
#from sklearn.externals import joblib
import joblib

#from sklearn import preprocessing
#from sklearn.preprocessing import StandardScaler   #Estandarización
from sklearn import preprocessing

#import plotearContinuo_SVR
import plotear
import matplotlib
import funcionesC_
import funcionesC2_

def main(argv):
   """
   TransformApp entry point
   argv: command-line arguments (excluding this script's name)
   """

   '''
   El módulo "argparse" facilita la escritura de interfaces de línea de comandos fáciles de usar.
   El programa define qué argumentos requiere, y "argparse" descubrirá cómo analizar los que están fuera de sys.argv.
   El módulo "argparse" también genera automáticamente mensajes de ayuda y uso y emite errores cuando los usuarios dan al programa argumentos inválidos.
   '''
   parser = argparse.ArgumentParser(description="Script de modelado de evolucion de IRI")

   # Named arguments
   parser.add_argument("--verbose", "-v", help="Genera una salida detallada en consola", action="store_true")
   parser.add_argument("datos", help="Ruta al archivo de datos csv", metavar="DATA_PATH", nargs="?")
   parser.add_argument("--plot", "-p", help="Grafica las predicciones", action="store_true")

   args = parser.parse_args(argv)

   if not args.datos:
      print("Faltan argumentos posicionales")
      parser.print_usage()
      return 1

   if os.path.isfile(args.datos):
      data = np.loadtxt(args.datos, delimiter=",")
   else:
      # Cargo datos
      data = np.loadtxt('analisisConFis.csv', delimiter=",")
   
   tramo = 9      # Para graficar un tramo determinado
   anioAPred = 5   # Para predecir hasta un anio determinado
   
   # Nos quedamos con el nro. de tramo
   etiquetas_tramos=np.unique(data[:, 0])
   lastTramo = np.max(etiquetas_tramos)
   
   # Reemplazamos el año absoluto, por el relativo, buscando y restandoselo al anio el menor año de evaluacion.
   for lbl in np.unique(data[:, 0]):
      current_lbl_indexes = data[:, 0] == lbl
      # Nos devuelve en True aquellas filas del tramo actual del "for"

      # Buscamos las filas donde aparece el 1ero. y ultimo registro de cada tramo
      # la vamos a usar para eliminar registros si hubo mejoras
      indice_minimo_tramo = np.min(np.argwhere(data[:, 0] == lbl))
      anioMin = data[indice_minimo_tramo, 1]
      data[np.argwhere(data[:, 0] == lbl), 1] -= anioMin
      
   ####################################
   # completamos id, anio, deflexion, nAnio, fisuras, ahuellamiento y rugosidad para los anios no evaluados    
   data = funcionesC_.completaAnios(data, lastTramo)
   index = np.lexsort((data[:, 1], data[:, 0]))   # ordenamos porque sino no hace lo que tiene que hacer las sgtes. funciones
   data = data[index]
   
   # descarto tramo cuando hay mejoras -> pasandola como nuevo tramo, inicializando el año en 0
   data = funcionesC_.descartacion_tramo_mejoras(data, lastTramo)
   index = np.lexsort((data[:, 1], data[:, 0]))
   data = data[index]
   
   # En data tenemos los datos sin los errores de medicion, y sin las mejoras
   data_original = data.copy();	    # En data_original guardamos los datos originales antes de modificarse por las mejoras y demas
   dataFis_original = data[:, :5]
   dataAhue_original = data[:, :6]    
   
   ####################################
   # Suavizamos los valores de RUGOSIDAD (NO DE FISURAS, NI DE AHUELLAMIENTO) con una polinomica de grado 3, para minimizar ruidos
   data = funcionesC_.search_polilinea(data)
   index = np.lexsort((data[:, 1], data[:, 0]))
   data = data[index]
   
   data = funcionesC_.forzar_ascendente(data, lastTramo)
   index = np.lexsort((data[:, 1], data[:, 0]))
   data = data[index]
   
   # Forzamos que las mediciones de Fisuras sean ascendentes -> 
   data = funcionesC_.forzar_ascendenteFis(data, lastTramo)
   index = np.lexsort((data[:, 1], data[:, 0]))
   data = data[index]
   
   # Forzamos que las mediciones de AHUELLAMIENTO sean ascendentes -> 
   data = funcionesC_.forzar_ascendenteAhue(data, lastTramo)
   index = np.lexsort((data[:, 1], data[:, 0]))
   data = data[index]
   
   # Como vamos a predecir el Fisuras, vamos a quedarnos con el vector hasta Fisuras, sin incluir el IRI
   dataFis = data[:, :5].copy()
   testFis_data_lbl = dataFis.copy()
   
   dataFis_lbl = dataFis[:,-1].copy()
   dataFis_lbl_original = dataFis_original[:,-1].copy()
   dataFis_sin_roll = dataFis.copy()
   dataFis[:, -1] = np.roll(dataFis[:, -1], 1)   
   
   # Como vamos a predecir las AHUELLAMIENTO, vamos a quedarnos con el vector hasta AHUELLAMIENTO, sin incluir el IRI
   dataAhue = data[:, :6].copy()
   testAhue_data_lbl = dataAhue.copy()
   
   dataAhue_lbl = dataAhue[:,-1].copy()
   dataAhue_lbl_original = dataAhue_original[:,-1].copy()
   dataAhue_sin_roll = dataAhue.copy()
   dataAhue[:, -1] = np.roll(dataAhue[:, -1], 1)   
   
   #################################### 
   data_lbl = data[:,-1].copy()
   data_lbl_original = data_original[:,-1].copy()
   data_sin_roll = data.copy()
   data[:, -1] = np.roll(data[:, -1], 1)
   
   for tramoTrain in np.unique(data_sin_roll[:, 0]):  
      current_lbl_indexes = data_sin_roll[:, 0] == tramoTrain
      indice_minimo_tramo = np.min(np.argwhere(current_lbl_indexes))
      indice_maximo_tramo = np.max(np.argwhere(current_lbl_indexes))
      anioMax = data_sin_roll[indice_maximo_tramo, 1]
           
      #Hacemos la polilinea si tenemos mas de dos puntos
      if anioMax >= 1:           
         # itero por los casos de corte para arreglar el iri
         case = indice_minimo_tramo
         m = data_sin_roll[case + 1, -1] - data_sin_roll[case, -1]
         h = data_sin_roll[case, -1]
         # evaluo en la recta en el x anterior (-1)
         jj = -1 * m + h - 0.05	#Modificado el 03/04/18 porque 0.5 era MUUUCHO ruido
         data[case, -1] = jj   # metemos ruido gaussiano sigma 0.5
         
         # LO MISMO PARA Fisuras??
         if (dataFis_sin_roll[case, -1]>0):
            m2 = dataFis_sin_roll[case + 1, -1] - dataFis_sin_roll[case, -1]
            h2 = dataFis_sin_roll[case, -1]
            # evaluo en la recta en el x anterior (-1)
            jj2 = -1 * m2 + h2 - 0.05	#Modificado el 03/04/18 porque 0.5 era MUUUCHO ruido
            if jj2 < 0:
               jj2 = 0
         else:
            jj2 = 0
         dataFis[case, -1] = jj2   # metemos ruido gaussiano sigma 0.5
         
         # LO MISMO PARA AHUELLAMIENTO??
         if (dataAhue_sin_roll[case, -1]>0):
            m3 = dataAhue_sin_roll[case + 1, -1] - dataAhue_sin_roll[case, -1]
            h3 = dataAhue_sin_roll[case, -1]
            # evaluo en la recta en el x anterior (-1)
            jj3 = -1 * m3 + h3 - 0.05	#Modificado el 03/04/18 porque 0.5 era MUUUCHO ruido
            if jj3 < 0:
               jj3 = 0
         else:
            jj3 = 0
         dataAhue[case, -1] = jj3   # metemos ruido gaussiano sigma 0.5        
   
   ########## ARMAMOS CONJUNTO PARA FISURAS Y AHUELLAMIENTO #################################
   # Creamos dos vectores, train_data: con todos los tramos menos el ultimo año (solo para aquellos tramos que tenga mas de dos vectores)   
   trainFis_data = dataFis.copy()
   trainFis_data_lbl = dataFis_lbl.copy()
   
   trainAhue_data = dataAhue.copy()
   trainAhue_data_lbl = dataAhue_lbl.copy()
   
   #################################### ENTRENAMOS ####################################
   # ESTANDARIZACIÓN --------------------------------------------------------
   # Solo las características de entrada train_data necesitan estandarización
   # A veces puede ayudar a estandarizar el objetivo, pero a menudo no es tan útil como la estandarización de 
   #...las características de entrada.
   #std_x = StandardScaler()            
   #train_data_Normalizado = std_x.fit_transform(train_data)
   #test_data_Normalizado = std_x.fit_transform(test_data)
   
   train_data = data.copy()
   train_data_lbl = data_lbl.copy() 

   data_paint, yPredict_draw = funcionesC2_.search_dataFull(data_sin_roll, data_lbl, tramo)
   yPredict_RFRdraw = yPredict_draw.copy()
   yPredict_SVRdraw = yPredict_draw.copy()
   
   train_data = train_data[:, 1:]   #Vamos a eliminarle el id del tramo, es una informacion redundante 
   train_data_sinEstandarizar = train_data.copy()  # Nos sirve para el proximo año
   
   #El test_data no lo puedo armar porque necesito fisura/ahuellamiento para el año siguiente
   
   #####################################################
   # Primero entrenamos para predecir la fisura y el ahuellamiento!!
   trainFis_data = trainFis_data[:, 1:]   #Vamos a eliminarle el id del tramo, es una informacion redundante 
   trainFis_data_sinEstandarizar = trainFis_data.copy()  # Nos sirve para el proximo año
   
   trainAhue_data = trainAhue_data[:, 1:]   #Vamos a eliminarle el id del tramo, es una informacion redundante 
   trainAhue_data_sinEstandarizar = trainAhue_data.copy()  # Nos sirve para el proximo añ      o
   
   min_max_scaler = preprocessing.MinMaxScaler()
   i = 1   
   while i <= anioAPred+1:     
      trainFis_data = min_max_scaler.fit_transform(trainFis_data_sinEstandarizar)
      
      if i==1:
         testFis_dataSVR = funcionesC2_.search_test_dataFis(dataFis_sin_roll, trainFis_data_lbl, tramo, 1)
         testFis_dataRFR = testFis_dataSVR.copy()
      else:
         anio = int(test_data_sinEstandarizar[0, 0] + 1)
         deflex = int(test_data_sinEstandarizar[0, 1])
         nAnio = round(float(test_data_sinEstandarizar[0, 2] * (1.02 ** i)), 3)
      
         testFis_dataSVR = []
         testFis_dataSVR = [anio, deflex, nAnio, fisSVR]
         testFis_dataSVR = np.array(testFis_dataSVR)
      
         testFis_dataRFR = []
         testFis_dataRFR = [anio, deflex, nAnio, fisRFR]
         testFis_dataRFR = np.array(testFis_dataRFR)
      
      testFis_data = testFis_dataSVR.reshape(1,-1) 
      testFis_data = min_max_scaler.transform(testFis_data)
      
      testFis_data_sinEstandarizar = testFis_dataRFR.copy()
      testFis_data_sinEstandarizar = testFis_data_sinEstandarizar.reshape(1,-1)         
      
      if i==1:
         ########################## FISURA ########################################
         clfFis = SVR(kernel='linear', C=60, epsilon=3.1)
         clfFis.fit(trainFis_data, trainFis_data_lbl)
         
         rfcFis = RFR(random_state=0, n_estimators=1500)
         rfcFis.fit(trainFis_data_sinEstandarizar, trainFis_data_lbl)
         ##########################################################################
      
      yPredictFis_SVR = clfFis.predict(testFis_data)
      yPredictFis_RFR = rfcFis.predict(testFis_data_sinEstandarizar)            
      
      fisSVR = float(yPredictFis_SVR[0])
      fisRFR = float(yPredictFis_RFR[0])

      trainAhue_data = min_max_scaler.fit_transform(trainAhue_data_sinEstandarizar)
      if i==1:   
         testAhue_dataSVR = funcionesC2_.search_test_dataAhue(dataAhue_sin_roll, trainAhue_data_lbl, tramo, 1, fisSVR)      
         testAhue_dataRFR = funcionesC2_.search_test_dataAhue(dataAhue_sin_roll, trainAhue_data_lbl, tramo, 1, fisRFR)
      else:
         testAhue_dataSVR = []
         testAhue_dataSVR = [anio, deflex, nAnio, fisSVR, ahueSVR]
         testAhue_dataSVR = np.array(testAhue_dataSVR)

         testAhue_dataRFR = []
         testAhue_dataRFR = [anio, deflex, nAnio, fisRFR, ahueRFR]
         testAhue_dataRFR = np.array(testAhue_dataRFR)
         

      testAhue_data = testAhue_dataSVR.reshape(1,-1) 
      testAhue_data = min_max_scaler.transform(testAhue_data)

      testAhue_data_sinEstandarizar = testAhue_dataRFR.copy()
      testAhue_data_sinEstandarizar = testAhue_data_sinEstandarizar.reshape(1,-1)
      
      if i==1:
         ########################## AHUELLAMIENTO #################################
         clfAhue = SVR(kernel='linear', C=50, epsilon=0.8)        #1.30009454704   
         clfAhue.fit(trainAhue_data, trainAhue_data_lbl)
         
         rfcAhue = RFR(random_state=0, n_estimators=300)
         rfcAhue.fit(trainAhue_data_sinEstandarizar, trainAhue_data_lbl)
         ##########################################################################
         
      yPredictAhue_SVR = clfAhue.predict(testAhue_data)
      yPredictAhue_RFR = rfcAhue.predict(testAhue_data_sinEstandarizar)            
      
      ahueSVR = float(yPredictAhue_SVR[0])
      ahueRFR = float(yPredictAhue_RFR[0])
      
      train_data = min_max_scaler.fit_transform(train_data_sinEstandarizar)
      
      if i==1:
         test_dataSVR = funcionesC2_.search_test_data(data_sin_roll, train_data_lbl, tramo, 1, fisSVR, ahueSVR)
         test_dataRFR = funcionesC2_.search_test_data(data_sin_roll, train_data_lbl, tramo, 1, fisRFR, ahueRFR)
      else:
         test_dataSVR = []   
         test_dataSVR = [anio, deflex, nAnio, fisSVR, ahueSVR, iriSVR]
         test_dataSVR = np.array(test_dataSVR)
         
         test_dataRFR = []   
         test_dataRFR = [anio, deflex, nAnio, fisRFR, ahueRFR, iriRFR]
         test_dataRFR = np.array(test_dataRFR)         
         
         
      test_data = test_dataSVR.reshape(1,-1)
      test_data = min_max_scaler.transform(test_data)         
      
      test_data_sinEstandarizar = test_dataRFR.copy()
      test_data_sinEstandarizar = test_data_sinEstandarizar.reshape(1,-1)
      
      if i==1:
         ########################## RUGOSIDAD #################################
         '''
         clf = SVR(C=1000, kernel='rbf', epsilon=0.02, gamma=0.003) 
         clf.fit(train_data, train_data_lbl)

         rfc = RFR(random_state=1, n_estimators=300)
         rfc.fit(train_data_sinEstandarizar, train_data_lbl)
         
         joblib.dump(clf, 'clf.pkl', compress=9)
         joblib.dump(rfc, 'rfc.pkl', compress=9)
         '''
         clf = joblib.load('clf.pkl')
         rfc = joblib.load('rfc.pkl')
         
         ######################################################################
         

              
      yPredict_SVR = clf.predict(test_data)
      yPredict_RFR = rfc.predict(test_data_sinEstandarizar)
      
      iriSVR = float(yPredict_SVR[0])
      iriRFR = float(yPredict_RFR[0])
      
      yPredict_SVRdraw.append(iriSVR)
      yPredict_RFRdraw.append(iriRFR)
      
      
      
      i += 1
   
   print(data_paint)
   print(yPredict_SVRdraw)
   print(yPredict_RFRdraw)
   
   if args.plot:
      #Graficamos
      plt.plot(data_paint, c='k', marker="o", fillstyle='full', label='valores reales del tramo + predichos')
      plt.xlabel("Años")	 #Insertamos el titulo del eje X
      plt.ylabel("Rugosidad[m/km]")    #Insertamos el titulo del eje Y
   
      plt.plot(yPredict_SVRdraw, c='c', marker="o", fillstyle='full', label="SVR")
      plt.plot(yPredict_RFRdraw, c='r', marker="o", fillstyle='full', label="RFR")
   
      plt.grid()
      #plt.title("SVR")
      plt.legend(loc=0)
      plt.show()
      
if __name__ == "__main__":
   main(sys.argv[1:])
