import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

def f(x):
    return (1/(1 + 25 * x **2))


#----------------------Metodos para interpolacion de lagrange-------------------------------------------
#Esta función lee los datos de un archivo txt
def extraer_datos(datos):
    x_values = []
    y_values = []
    datos = np.loadtxt(datos)
    for i in range(len(datos)):
        x_values.append(datos[i][0])
    for j in range(len(datos)):
        y_values.append(datos[j][1])
    return x_values,y_values


def lagrange_interpolation(x_values, y_values, x):
    """
    Realiza la interpolación de Lagrange para encontrar el valor de f(x)
    en el punto dado x.

    Args:
    x_values: Lista de valores x conocidos.
    y_values: Lista de valores y conocidos corrspondientes a los valores x.
    x: El valor x para el cual se desea interpolar.

    Returns:
    El valor interpolado de f(x).
    """
    n = len(x_values)

    result = 0.0
    for i in range(n): #Iniciamos desde i hasta el tamaño del vector de x_values
        term = y_values[i] #Vamos a recorrer la array de los valores de y o f(xi)

    #Con este ciclo vamos a calcular cada uno de los terminos del polinomiio interpolante
        for j in range(n):
            #Debemos asegurar que j e i sean diferentes ya que si son iguales el denominador daria 0
            if i != j:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term #Acumulamos los polinomios interpolantes en la variable result
    return result


# ------------------Metodos para interpolacion de base radial----------------------------
#Función de evaluación de FBR Multicuadratica
def rbffunction(xev, xdat, c):
    rbfv = np.sqrt((xev - xdat)**2 + c**2)
    return rbfv

#Construimos de la matriz de interpolación
def interpmat(xdat, c):
    nd = len(xdat)
    mat1 = np.zeros((nd, nd),float)
    for i in range(nd):
        for j in range(nd):
            mat1[i,j] = rbffunction(xdat[i], xdat[j], c)
    return mat1

# Superposición de funciones de base radial
def rbfsuperposit(x, coef, xdat, c):
    y = np.zeros((len(x)))
    for i in range(len(x)):
        for j in range(len(xdat)):
            y[i] = y[i] + coef[j]*rbffunction(x[i], xdat[j], c)
    return y


#------------------Grafica para Lagrange y Funcion de Base Radial-----------------------------------------
def graficas (x,yinterp,y_plot,x_values,y_values, xinter, Lagrange_intepolated):
    plt.figure()
    plt.plot(x, (1 / (1 + 25 * x**2)), label= 'Función 1/1+25x**2')
    plt.plot(x, yinterp, label = 'Interpolación RBF')
    plt.plot(x, y_plot, label = 'Polinomio Lagrange')
    plt.scatter(x_values, y_values, color='red', label='Datos')

    #Marcamos los valores interpolados x=1.5 y x=5.7
    plt.scatter(xinter, Lagrange_intepolated, color='purple', label='Valores Interpolados')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.title('Interpolación con funciones de Base Radial vs Lagrange')
    plt.show()



#-----------Trazadore lineales-------------------------------------------
def Interpolante(fxi,x_i,x):# En esta funcion buscamos el valor interpolante 
    res=[]
    num=0
    for j in range(len(x)):
        if x_i[-1]>=x[j] and x_i[0]<=x[j]:
            for i in range(1,len(x_i)):
                if (x[j] <= x_i[i] and num!=x[j]):
                    num=x[j]
                    y=round(fxi[i]+((fxi[i]-fxi[i-1])/(x_i[i]-x_i[i-1]))*(x[j]-x_i[i]),4)
            res.append(y)
    return res


#-----------Interpolación Polinomica-----------------------------------
# Ingreso de datos, pero acá están presentados como listas
def interpolacion(x,y):
    # Procedimiento, pasamos a convertirlos en arreglos
    xi = np.array(x)
    fi = np.array(y)
    B = np.copy (y)

    #Matriz
    n = len(xi) #Tamaño de la matriz
    D = np.zeros((n,n),dtype=float) #Matriz, llena de ceros
    ultima = n-1

    #Para llenar las casillas
    i = 0 #Fila cero
    for i in range(0,n,1): #mover el valor de i, en el rango desde la primera hasta la ultima
        for j in range(0,n,1): #mover el valor de j, en el rango desde la primera hasta la ultima
            potencia = ultima -j #Se calcula el valor de la exponente
            D[i,j] = xi[i] **potencia #Posicion en la matriz

    # Calculo de coeficientes del polinomio,mediante algebra lineal
    coeficientes = np.linalg.solve(D,B)

    #Polinomio de Interpolación
    x = sym.Symbol('x') # X va a ser tomada como un simbolo
    polinomio = 0 #Comenzamos con polinomio vacío
    for i in range(0,n,1):
        potencia = (n-1)-i #Es relativo al valor de la ultima posición
        termino = coeficientes[i]*(x**potencia)
        polinomio = polinomio + termino

    #Para facilitar la evaluación del polinomio
    px = sym.lambdify(x,polinomio)

    #Evaluar polinomio
    a = np.min(xi) #Minimo
    b= np.max(xi) #Máximo
    pxi = np.linspace(a,b) #Serie de puntos muestreados
    pfi = px(pxi) #puntos de la función,usando la forma numerica del polinomio

    #Salida
    print ('Matriz')
    print (D[i,j])
    print ('Coeficientes:')
    print (coeficientes)
    print ('polinomio: ')
    print (polinomio)

    #crearemos una grafica
    plt.plot(xi,fi,'o', label='Puntos')
    plt.plot(pxi,pfi, label='Polinomio') #Trazamos la linea de los puntos
    plt.legend() #Mostrar todas las etiquetas
    plt.xlabel('xi') #Añadimos una etiqueta
    plt.ylabel('fi') #Añadimos una etiqueta
    plt.title('Polinomio de Interpolación')#Añadimos un titulo
    plt.grid(True)
    plt.show() #Para ver la gráfica

    puntosx = xi
    puntosy = fi
    polinomiox = pxi
    polinomioy = pfi
    print("X:1.5, Y:",px(1.5))
    print("X:5.7, Y: ",px(5.7))
    return puntosx,puntosy,polinomiox,polinomioy



#---------------Interpolación de Newton-----------------------
def calcular_diferencias_divididas(x, y):
    n = len(x) # es el número de puntos de datos el cual igualamos a el tamaño de la x convocada
    diferencias = np.zeros((n, n)) #Es una matriz para almacenar las diferencias divididas. Cada columna representa las diferencias de orden superior.
    diferencias[:, 0] = y #  Inicializa la primera columna con los valores de la función f(x)
#Entramos en el ciclo, que se define que para i en el rango de 1 a n
    for i in range(1, n): #columnas
        for j in range(n - i): # filas de diferencias
            diferencias[j, i] = (diferencias[j + 1, i - 1] - diferencias[j, i - 1]) / (x[j + i] - x[j]) 
    return diferencias[0,:]    # retorna la matriz  con los coeficiones de de la diferencias divididas            #construido de esta forma como un bloque  f(x)- f(y)                                                                                                                             x-y

def evaluar_polinomio_newton(x_val, x, coeficientes):
    n = len(coeficientes)# Inicializamos n con la longitud de los coeficientes  
    resultado = coeficientes[0] # Inicializamos resultado con el primer coeficiente.
    for i in range(1, n):
        producto = coeficientes[i] # Inicializamos producto con el coeficiente correspondiente al término i del polinomio.
        for j in range(i): 
            producto *= (x_val - x[j]) #multiplicamos el producto por la diferencia en cada iteracion del bucle
        resultado += producto #suma los productos y los acomula 
    return resultado #por ultimo lo retornamos


#--------------Trazadores Cuadraticos---------------
# Símbolos y variables
a, b, c, x = sym.symbols('a b c x')

# Definir la ecuación cuadrática genérica
ecuacion_generica = a * x**2 + b * x + c

def funciones(datos):
    num_ecuaciones = len(datos) - 1 # Número de ecuaciones

    # Crear un array NumPy para almacenar las ecuaciones
    ecuaciones_array = np.empty(num_ecuaciones, dtype=object)

    # Llenar el array con las ecuaciones cuadráticas
    for i in range(num_ecuaciones):
        ecuaciones_array[i] = ecuacion_generica.subs({a: sym.symbols(f'a{i+1}'), b: sym.symbols(f'b{i+1}'), c: sym.symbols(f'c{i+1}')})
        ecuaciones_array[i] = ecuaciones_array[i].subs('a1', 0)

    return ecuaciones_array

def ecuaciones(datos, f_x, ecuaciones_array):
    # Crear una lista para almacenar las soluciones
    soluciones = []
    # Procesar las ecuaciones
    for i in range(len(datos)):
        if i == 0:
            soluciones.append(sym.Eq(ecuaciones_array[0].subs(x, datos[0]), f_x[0])) # Guardar primera ecuación
        elif i == len(datos) - 1:
            soluciones.append(sym.Eq(ecuaciones_array[-1].subs(x, datos[-1]), f_x[-1]))  # Guardar última ecuación
        else:
            temp_soluciones = []
            ecuacion_i = sym.Eq(ecuaciones_array[i-1].subs(x, datos[i]), f_x[i])
            ecuacion_sig = sym.Eq(ecuaciones_array[i].subs(x, datos[i]), f_x[i])
            temp_soluciones.extend([ecuacion_i, ecuacion_sig])  # Agregar las ecuaciones a la lista temporal
            soluciones.extend(temp_soluciones)
    return soluciones

#calcular las derivadas
def derivadas(ecuaciones_array):
    derivadas_array = []
    for ecuacion in ecuaciones_array:
        derivada_ecuacion = sym.diff(ecuacion, x)
        derivadas_array.append(derivada_ecuacion)
    return derivadas_array

def evaluar_derivadas(derivadas_array, datos):
    deriv_eva = []
    for i in range(len(derivadas_array) - 1):
        derivada = sym.Eq((derivadas_array[i].subs('x', datos[i+1])), derivadas_array[i+1].subs('x', datos[i+1]))
        deriv_eva.append(derivada)
    return deriv_eva

def incognitas(soluciones):
    lista_a = []
    lista_b = []
    lista_c = []
    incog = []

# Separar los elementos
    for elemento in soluciones:
    # Obtener los símbolos presentes en la solución
        inco = elemento.free_symbols
        for ig in inco:
            nombre_incognita = str(ig)
            if nombre_incognita.startswith('a'):
                if nombre_incognita not in lista_a:
                    lista_a.append(nombre_incognita)
            elif nombre_incognita.startswith('b'):
                if nombre_incognita not in lista_b:
                    lista_b.append(nombre_incognita)
            elif nombre_incognita.startswith('c'):
                if nombre_incognita not in lista_c:
                    lista_c.append(nombre_incognita)

        incog = lista_a + lista_b + lista_c
    return incog

# Definición de la función reemplazar_coeficientes
def reemplazar_coeficientes(ecuaciones_array, soluciones, deriv_eva):
    # Extraer los símbolos de las ecuaciones
    coeficientes = incognitas(ecuaciones_array)

    # Concatenar las listas de soluciones y derivadas
    soluciones_y_derivadas = soluciones + deriv_eva

    # Resolver el sistema de ecuaciones para encontrar los valores de los coeficientes
    solucion = sym.solve(soluciones_y_derivadas, coeficientes)

    # Iterar sobre cada ecuación en el array
    for i, ecuacion in enumerate(ecuaciones_array):
        # Iterar sobre cada símbolo y valor en la solución
        for simbolo, valor in solucion.items():
            # Reemplazar el símbolo por el valor en la ecuación
            ecuacion = ecuacion.subs(simbolo, valor)
        # otra distinta
        ecuaciones_array[i] = ecuacion

    return ecuaciones_array