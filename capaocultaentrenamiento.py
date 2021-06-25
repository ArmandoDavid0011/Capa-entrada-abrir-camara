import cv2 as cv
import os
import numpy as np
from time import time #Ver el tiempo en que se tarda una determinada función

dataRuta='D:/Documentos/CURSOS/PYTHON/Proyecto Python RF/Curso/reconocimientofacial1/Data' #hacer comparaciones de las fotos
listaData=os.listdir(dataRuta) # Ruta de nuestras fotos DATA
# print('Data',listaData)
ids=[] # para poder agregar las etiquetas
rostrosData=[] 
id=0 
#Medicion del tiempo
tiempoInicial=time() 
#Recodido de imagenes
for fila in listaData: 
    rutacompleta=dataRuta+'/'+ fila
    print('Inciando lectura...')
    for archivo in os.listdir(rutacompleta):
        
        print('Imagenes: ',fila + '/' + archivo)

        ids.append(id) #Trabajando con los ID
        rostrosData.append(cv.imread(rutacompleta+'/'+archivo,0)) #Agregar el texto o titulo para la diferencia
                
    id=id+1 #Clasificar imagenes agregandole un nuevo valor.
    tiempofinalLectura=time() #Empieza el bucle a trabajar
    tiempoTotalLectura=tiempofinalLectura-tiempoInicial # Saber cuanto tiempo se ha tardado en el proceso de lectura
    print('Tiempo total lectura: ',tiempoTotalLectura) #Sabremos que timepo ha tardado

entrenamiento=cv.face.EigenFaceRecognizer_create() #Aplicando proceso de entrenamiento
#Asignacion de nuestro primer modelo
print('Iniciando el entrenamiento...espere')
entrenamiento.train(rostrosData,np.array(ids)) 
tiempofinalEntramiento=time()
tiempoTotalEntrenamiento=tiempofinalEntramiento - tiempoTotalLectura
#controlar el tiempo
print('Tiempo entrenamiento total',tiempofinalEntramiento)
#Almacenando nuestro primer entrenamiento
entrenamiento.write('EntrenamientoEigenFaceRecognizer.xml' ) 
print('Entrenamiento concluido.!') 