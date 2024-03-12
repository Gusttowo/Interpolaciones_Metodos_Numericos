import Interpolaciones as intpl
import numpy as np 
import matplotlib.pyplot as plt
from sympy import Symbol, lambdify

print('Se utilizan 6 métodos de interpolación los cuales son:\n-Funciones de Base Radial \n-Trazadores Lineales y Cuadraticos\n-Polinomico\n-Lagrange\n-Newton.\n\n')
print('Se nos pide evaluar cada método en la función 1/1 + 25x^2, teniendo como restricción seleccionar 16 puntos en un intervalo de [-1,1].',
'Adicional nos piden evaluar cada interpolacion en x=[pi/4, -pi/5, pi/6]. Con cada método de inteporlación aproximamos la función dada.\n')

# Valores conocidos
#Extraemos los datos del archivo de texto
data = np.loadtxt('Interpolaciones\datos.txt')
x_values = np.array(data)
y_values = intpl.funcion(x_values)

# Puntos en los que se quiere interpolar
xinter = np.array([np.pi/4, -np.pi/5, np.pi/6])

#----------interpolacion de Base Radial--------------------------------------------------------
#Definimos el parametro de forma llamado C
c = 0.01
matint=intpl.interpmat(x_values,c)

#Coeficientes de la interpolacion 
coef=np.linalg.solve(matint,y_values)

#Evaluacion de la superposicion de sobre un intervalo
x = np.linspace(-1,1,200, endpoint= True)
yinterp= intpl.rbfsuperposit(x, coef, x_values,c)

#Calculo de error para Base radial
ErrBR = np.sqrt(np.sum((yinterp - intpl.funcion(x))**2) / len(yinterp))
print('Parametro de forma: ', c)
print('Error RMS de la aproximación RBF es: ', ErrBR)


#------------Interpolacion por Lagrange-------------------------------------------------------
Lagrange_intepolated = intpl.lagrange_interpolation(x_values, y_values, xinter)
print("El valor interpolado en x =", xinter, "es:", Lagrange_intepolated)

#Invocamos la función del modulo para generar la interpolacion de lagrange
y_plot = intpl.lagrange_interpolation(x_values, y_values, x)

#Calculamos el error  entre las dos funciones
ErrLG = np.sqrt(np.sum((y_plot - intpl.funcion(x))**2) / len(y_plot))
print('Error RMS de la aproximación Lagrange: ', ErrLG)

#Generamos las gráficas
intpl.graficas(x,yinterp,y_plot,x_values,y_values, xinter, Lagrange_intepolated)


#-----------Trazadores lineales-------------------------------------------
#Llamamos la funcion Interpolante del Modelo y la guardamos en la variable y 
y_trazL=intpl.Interpolante(y_values,x_values,xinter)

plt.figure()
plt.plot(x, intpl.funcion(x), label= 'Función 1/1+25x^2', linestyle="solid") #se traza la linea entre los puntos
plt.scatter(xinter, y_trazL, color='green',label="Valor Interpolante") #se grafica el valor interpolante con un color verde 
plt.plot(x_values,y_values,color='red',label="Función interpolante", linestyle='dashed') #Se grafican los puntos conocidos con un color rojo
plt.scatter(x_values, y_values, label = 'Puntos conocidos')
plt.legend() #Se insertan las leyendas 
plt.grid(True) #en el grafico se ponen cuadriculas
plt.xlabel('Eje X') # Agregar etiquetas y título
plt.ylabel('Eje Y')
plt.title('Trazador Lineal')
plt.show()


#-----------Interpolación Polinomica--------------------------------
ultimo_dato = x_values[-1]

puntosx_inti,puntosy_inti,polinomiox_inti,polinomioy_inti = intpl.interpolacion(x_values,y_values, x)
inicio = 0.4
fin = ultimo_dato+1
cantidad_numeros_deseados = len(polinomiox_inti)
paso = (fin - inicio) / (cantidad_numeros_deseados - 1)
x_jess = np.arange(inicio, fin + paso, paso)
Err = np.sqrt(np.sum((polinomioy_inti - intpl.funcion(x_jess))**2)/len(polinomioy_inti))
print("Error de la interpolación polinomica es: ", Err)


#-----------------Interpolación de Newton-------------------------------
# Calcular las diferencias divididas
coeficientes = intpl.calcular_diferencias_divididas(x_values, y_values) # calcular_diferencias_divididas(x, y) implementa el calculo de las diferencias divididas

#  GRAFICA
y_grafica = np.array([intpl.evaluar_polinomio_newton(xinter, x_values, coeficientes) for xinter in x])

# Graficar los datos originales y el polinomio de Newton
plt.figure() # asignamos el tamaño de la grafica
plt.plot(x, y_grafica, label='Polinomio de Newton', color='green', linestyle='dashed')
plt.plot(x, intpl.funcion(x), label= 'Función 1/1+25x^2', linestyle="solid")
plt.scatter(x_values, y_values, color='red', label='Datos Originales')
plt.title('Datos y Aproximación por Polinomio de Newton')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

# Evaluar el polinomio en los puntos especificados
valores_evaluados = [intpl.evaluar_polinomio_newton(x_val, x_values, coeficientes) for x_val in xinter]

# Mostrar los resultados de la evaluación
# zip combina elementos iterales para crear pares ordenados y dic crea un diccionario que asocia los resultados evualados en esos puntos
valores_evaluados_dict = dict(zip(xinter, valores_evaluados)) 
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

intervalos = intpl.intervalos(x_values)

funciones = intpl.asignarIntervalos(ecuaciones_con_coeficientes, intervalos)

#Puntos de la interpolación
puntos_in = []
for x in xinter:
    for funcion, intervalo in zip(ecuaciones_con_coeficientes, intervalos):
        if intervalo[0] <= x <= intervalo[1]:
            punto_y = lambdify(Symbol('x'), funcion)(x)
            puntos_in.append(punto_y)
            break
        

#Graficar cada funcion
for funcion, intervalo in funciones:
    x_valuess = np.linspace(intervalo[0], intervalo[1], 200)
    y_valuess = [lambdify(Symbol('x'), funcion)(x_val) for x_val in x_valuess]
    plt.plot(x_valuess, y_valuess, linestyle = "dashed")
    plt.plot(x_valuess, intpl.funcion(x_valuess), linestyle="solid")
    
plt.xlabel('x')
plt.scatter(x_values, y_values, color='purple', label='Datos de tabla')
plt.scatter(xinter, puntos_in,color = 'red', label='Punto de interpolación')
plt.ylabel('f(x)')
plt.title('Trazador Cuadrático')
plt.legend()
plt.grid(True)
plt.show()
