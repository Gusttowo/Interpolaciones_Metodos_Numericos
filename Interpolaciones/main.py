import Interpolaciones as intpl
import numpy as np 
import matplotlib.pyplot as plt
from sympy import Symbol

# Valores conocidos
#Extraemos los datos del archivo de texto
x_values, y_values = intpl.extraer_datos('Interpolaciones\info_nueva.txt')

# Puntos en el que se quiere interpolar
xinter = np.array([np.pi/4, -np.pi/5, np.pi/6])


#----------interpolacion por base radial--------------------------------------------------------
#Definimos el parametro de forma llamado C
c = 0.2
matint=intpl.interpmat(x_values,c)

#Coeficientes de la interpolacion 
coef=np.linalg.solve(matint,y_values)

#Evaluacion de la superposicion de sobre un intervalo
x = np.linspace(-1, 1, 200, endpoint=True)
yinterp= intpl.rbfsuperposit(x, coef, x_values,c)

#Calculo de error para Base radial
ErrBR = np.sqrt(np.sum((yinterp - (intpl.f(x)))**2) / len(yinterp))
print('Parametro de forma: ', c)
print('Error RMS de la aproximación RBF: ', ErrBR)


#------------Interpolacion por Lagrange-------------------------------------------------------
Lagrange_intepolated = intpl.lagrange_interpolation(x_values, y_values, xinter)
print("El valor interpolado en x =", xinter, "es:", Lagrange_intepolated)

#Invocamos la función del modulo para generar la interpolacion de lagrange
y_plot = intpl.lagrange_interpolation(x_values, y_values, x)

ErrLG = np.sqrt(np.sum((y_plot - (intpl.f(x))**2) / len(y_plot)))
print('Error RMS de la aproximación Lagrange: ', ErrLG)
intpl.graficas(x,yinterp,y_plot,x_values,y_values, xinter, Lagrange_intepolated)




#-----------Trazadore lineales-------------------------------------------
y_trazL=intpl.Interpolante(y_values,x_values,xinter) #Llamamos la funcion Interpolante del Modelo y la guardamos en la variable y 

plt.figure()
plt.plot(x_values, y_values, label='Funcion interpolante') #se traza la linea entre los puntos
plt.scatter(xinter, y_trazL, color='green',label="Valor Interpolante") #se grafica el valor interpolante con un color verde 
plt.scatter(x_values,y_values,color='red',label="Puntos Conocidos") #Se grafican los puntos conocidos con un color rojo
plt.legend() #Se insertan las leyendas 
plt.grid(True) #en el grafico se ponen cuadriculas
plt.xlabel('Eje X') # Agregar etiquetas y título
plt.ylabel('Eje Y')
plt.title('Trazador Lineal')
plt.show()


#-----------Interpolación Polinomica--------------------------------
ultimo_dato = x_values[-1]

puntosx_inti,puntosy_inti,polinomiox_inti,polinomioy_inti = intpl.interpolacion(x_values,y_values)
inicio = 0.4
fin = ultimo_dato + 1
cantidad_numeros_deseados = len(polinomiox_inti)
paso = (fin - inicio) / (cantidad_numeros_deseados - 1)
x_jess = np.arange(inicio, fin + paso, paso)
Err = np.sqrt(np.sum((polinomioy_inti - (intpl.f(x_jess))**2) /len(polinomioy_inti)))
print("Error de nuestro polinomio", Err)


#-----------------Interpolación de Newton-------------------------------
# Calcular las diferencias divididas
coeficientes = intpl.calcular_diferencias_divididas(x_values, y_values) # calcular_diferencias_divididas(x, y) implementa el calculo de las diferencias divididas


#  GRAFICA
y_grafica = np.array([intpl.evaluar_polinomio_newton(xinter, x_values, coeficientes) for xinter in x]) 

# Graficar los datos originales y el polinomio de Newton
plt.figure() # asignamos el tamaño de la grafica
plt.plot(x, y_grafica, label='Polinomio de Newton', color='green')
plt.plot(x, (1 / (1 + 25 * x**2)), label='Función dada')
plt.scatter(x_values, y_values, color='red', label='Datos Originales')
plt.title('Datos y Aproximación por Polinomio de Newton y base radial')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

# Evaluar el polinomio en los puntos especificados
valores_evaluados = [intpl.evaluar_polinomio_newton(x_val, x_values, coeficientes) for x_val in xinter]

# Mostrar los resultados de la evaluación
valores_evaluados_dict = dict(zip(xinter, valores_evaluados)) # zip combiana elementos iterales para crear pares ordenados y dic crea un diccionario que asocia los resultados evualados en esos puntos
print(valores_evaluados_dict)
plt.scatter(xinter, valores_evaluados, marker='o', color='orange', label='Puntos evaluados')  # Puntos evaluados
plt.legend()
plt.show()


#-----------------Trazadores Cuadraticos-------------------------
x = Symbol('x')

# Calcular las ecuaciones y las derivadas
ecuaciones_array = intpl.funciones(x_values)
soluciones = intpl.ecuaciones(x_values,  y_values, ecuaciones_array)
derivadas_array = intpl.derivadas(ecuaciones_array)
resultados_derivadas = intpl.evaluar_derivadas(derivadas_array, x_values)
deriv_eva = intpl.evaluar_derivadas(derivadas_array, x_values)
incognis = intpl.incognitas(soluciones)

# Calcular las ecuaciones con los coeficientes reemplazados
ecuaciones_con_coeficientes = intpl.reemplazar_coeficientes(ecuaciones_array, soluciones, deriv_eva)

#Gráficos
# Crear una lista para almacenar los valores de x correspondientes a cada segmento
segmentos_x = [np.linspace(x_values[i], x_values[i+1], 20) for i in range(len(x_values) - 1)]

# Crear una lista para almacenar los valores de y correspondientes a cada segmento
segmentos_y = []

# Iterar sobre cada segmento y evaluar las ecuaciones
for i, segmento in enumerate(segmentos_x):
    # Evaluar la ecuación correspondiente al segmento actual
    y_ecuacion_segmento = np.array([ecuaciones_con_coeficientes[i].subs(x, val) for val in segmento])
    segmentos_y.append(y_ecuacion_segmento)

# Graficar las ecuaciones interpoladas en sus rangos específicos
for i, segmento in enumerate(segmentos_x):
    plt.plot(segmento, segmentos_y[i], label=f'Ecuación {i+1}', color=np.random.rand(3,))
plt.scatter(x_values, y_values, color='red', label='Datos de tabla')
plt.legend()
plt.title('Trazador Cuadrático')
plt.grid(True)
plt.show()