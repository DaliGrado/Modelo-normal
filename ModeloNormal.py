#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:10:46 2023

@author: DG
"""

import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp
import time
#%%
#np.random.seed(231)
np.random.seed(55)
locals_num = 20#65#40
houses = 100#200#123
W_capacidades= 1
# Definimos los índices i y j
I = list(range(houses))
J = list(range(locals_num))
W = list(range(W_capacidades))

#locales_totales = 9
dyj = [16, 32]#[15,30]
f_jw = [10000,20000]
cj = 300#[200,250,300,230,350] #np.random.randint(200,300,locals)
qi = np.random.randint(4,6+1, size=len(I))
nr = 900#[100,300,300,232,700]

opciones = ['Binary', 'Continuous']
eleccion = "Binary"#input("Por favor, elige una de las siguientes opciones: " + ', '.join(opciones) + ": ")
while eleccion not in opciones:
    print("Lo siento, esa no es una opción válida.")
    eleccion = input("Por favor, elige una de las siguientes opciones: " + ', '.join(opciones) + ": ")

#J_fic = [-999]

#creamos iterador
iterations = 0

#creeamos matriz de margen
#def create_random_matrix(I, J, A, B):
#    matrix = np.empty((I, J), dtype=int)  # No se agrega espacio para el valor ficticio aquí
#    for i in range(I):
#        row = np.random.choice(np.arange(A, B+1)[np.arange(A, B+1) != 0], size=J, replace=False)
#        matrix[i] = row
#    return matrix
#margin = create_random_matrix(houses, locals_num,-locals_num,2)
#def remove_last_element_from_sublists(matrix):
 #   return [sublist[:-1] for sublist in matrix]
#margin = remove_last_element_from_sublists(margin)

##############################################################################################
def generar_cuadricula(T, J, I):
    # Crear una cuadrícula de tamaño T x T
    cuadricula = np.zeros((T, T))

    # Generar ubicaciones aleatorias para J locales
    locales = np.random.rand(len(J), 2) * T

    # Generar ubicaciones aleatorias para I usuarios
    usuarios = np.random.rand(len(I), 2) * T

    # Generar valores de reserva aleatorios para usuarios en el rango [0, T]
    reservas = np.random.randint(0, (T+1)/1, size=len(I))

    # Calcular distancias euclidianas entre usuarios y locales
    distancias = np.zeros((len(I), len(J)))
    for i in range(len(I)):
        for j in range(len(J)):
            distancias[i, j] = np.linalg.norm(usuarios[i] - locales[j])

    # Crear matriz margin restando las reservas a las distancias
    margin = reservas.reshape(-1, 1) - distancias

    return cuadricula, locales, usuarios, reservas, distancias, margin

def dibujar_cuadricula(cuadricula, locales, usuarios):
    fig, ax = plt.subplots(figsize=(T/4, T/4))

    # Configurar el color de fondo a blanco para toda la figura
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Dibujar la cuadrícula
    plt.imshow(cuadricula, cmap='gray')

    # Dibujar locales como círculos con etiquetas
    for j, local in enumerate(locales):
        # Ajustar el tamaño de los puntos y la dispersión
        plt.scatter(local[1], local[0], marker='o', color='yellow', s=1000, alpha=0.7)
        plt.text(local[1], local[0], f'{j+1}', fontsize=12, ha='center', va='bottom', color='yellow')

    # Dibujar usuarios como 'X' con etiquetas
    for i, usuario in enumerate(usuarios):
        # Ajustar el tamaño de los puntos y la dispersión
        plt.scatter(usuario[1], usuario[0], marker='x', color='red', s=1000, alpha=0.7)
        plt.text(usuario[1], usuario[0], f'{i+1}', fontsize=12, ha='center', va='top', color='red')

    # Leyenda personalizada
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Posibles Locales'),
               plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=10, label='Usuarios Potenciales')]
    plt.legend(handles=handles, title='Leyenda')

    plt.grid()
    plt.show()
    
# Ejemplo de uso con T=10, J=5, I=10
T = 100#max(len(I), len(J))

cuadricula, locales, usuarios, reservas, distancias, margin = generar_cuadricula(T, J, I)
dibujar_cuadricula(cuadricula, locales, usuarios)


print("locales:")
print(locales)
#print(reservas)
#print(distancias)
print("MARGEEEEEEEEEENNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
print(margin)
for i in I:
    maximo = max(margin [i][j] for j in J)
    indice = np.where(margin[i] == maximo)
    #print(f"El maximo para {i+1} es el local {indice[0][0] + 1}")
indices = []
for i in range(len(margin)):
    row_indices = [j for j, marg in enumerate(margin[i]) if marg > 0]
    indices.append(row_indices)
# for i in I:
#     if i == 27 or i == 84:
#         print(margin[i])
#print(margin)
for i in I:
  if all(margin[i][j] <=0 for j in J):
    print(f"El individuo {i+1} no asistira nunca")
print(qi)

#%%

#El numero de locales de debe cambiar arriva en locals
#locales_propuestos = [25]
M = sum(max(margin[i]) for i in I)  # Calcula el valor máximo de preferencia
new_cj = [300]#[230,250,300] #np.random.randint(200,300,locals)
new_nr = [900]#[232,300,900]
relation_cost_dump =[1.5]#[1,1.5]
dump = 9#int(input("Model without dumping = 0, Model with dumping = any other number_key: "))
opciones = ['Binary', 'Continuous']
eleccion = 'Binary'#input("Por favor, elige una de las siguientes opciones: " + ', '.join(opciones) + ": ")
while eleccion not in opciones:
    print("Lo siento, esa no es una opción válida.")
    eleccion = input("Por favor, elige una de las siguientes opciones: " + ', '.join(opciones) + ": ")
#%%
##########################CODIGO GUROBI################################
#primero, definir la función del modelo de optimización que toma cj y nr como argumentos
def optimize_recycling_level_dum(cj, nr, dump, relation_values,binary=True):
    # Crear un modelo
    m = gp.Model("MiModelo")
    
    # Crear variables
    
    x = m.addVars(I,J,vtype=gp.GRB.BINARY)
    y = m.addVars(J,W,vtype=gp.GRB.BINARY)
    z = m.addVars(J,vtype=gp.GRB.CONTINUOUS)    
    
    # Función objetivo
    fixed_cost = gp.quicksum(f_jw[w]*y[j,w] for j in J for w in W)
    cost_of_manage = cj * (gp.quicksum(qi[i] * x[i, j] for i in I for j in J) - gp.quicksum(z[j] for j in J))
    dumping_cost = nr * relation_values * gp.quicksum(z[j] for j in J)
    not_recycle_cost = nr * gp.quicksum(qi[i] * (1 - gp.quicksum(x[i, j] for j in J)) for i in I)
    
    if dump != 0:
        m.setObjective(fixed_cost + cost_of_manage + dumping_cost + not_recycle_cost, sense=gp.GRB.MINIMIZE)
    else:
        m.setObjective(fixed_cost + cost_of_manage + not_recycle_cost, sense=gp.GRB.MINIMIZE)
    
    #Capacidad
    for j in J:
        m.addConstr(gp.quicksum(qi[i]*x[i,j] for i in I) <= gp.quicksum(dyj[w]*y[j,w] for w in W) + z[j])
        #if j != 15 or j != 6:

        #m.addConstr(y[j,1] == 0)
    
    #m.addConstr(gp.quicksum(y[j,1] for j in J) <= 2)

    #Locales
    m.addConstr(gp.quicksum(y[j,w] for j in J for w in W) <= locals_num)
    
    #Asiste a puntos abiertos
    # x < 1
    # Restricción de asignación de usuarios a locales
    for i in I:
        m.addConstr(gp.quicksum(x[i, j] for j in J) <= 1)
        for j in J:
            m.addConstr(x[i, j] <= gp.quicksum(y[j, w] for w in W))
    
    #preferencias
    for i in I:
        for j in J:
            #discutido para obligar a no asistir en caso de no existir dumping
            if dump == 0:
                m.addConstr(gp.quicksum(x[i, h] for h in J if margin[i][h] <= margin[i][j]) + gp.quicksum(y[j,w] for w in W) <= 1)
            #cuando Existe dumping se toma:
            else:
                if margin[i][j] >= 0:
                    m.addConstr(gp.quicksum(x[i, h] for h in J if margin[i][h] >= margin[i][j]) >= gp.quicksum(y[j,w] for w in W ))

    #preferencias aux
    for i in I:
        for j in J:
            if margin[i][j] <= 0:
                m.addConstr(x[i,j] == 0)
    
    
    m.optimize()

    if m.status == gp.GRB.OPTIMAL:
        print(f"Valor objetivo: {m.objVal}")
        warehouse_y = []
        for j in J:
            y_aux =[]
            for w in W:
                valor = y[j, w].X  # Obtener el valor de la variable y[j, w]
                y_aux.append(valor)
            warehouse_y.append(y_aux)
            
            
        for j, elem in enumerate(warehouse_y):
            if (elem[0] >= 1) :#or elem[1] >= 1) :
                if elem[0] >= 1:
                    print(f"Abrimos con capacidad 16 en el punto {j+1}")
                #if elem[1] >= 1:
                 #   print(f"Abrimos con capacidad 32 en el punto {j+1}")
            else:
                print(f"El punto {j+1} no se abre")
        
        matriz_x = np.zeros((len(I), len(J)))
        for i, fila in enumerate(I):
            for j, columna in enumerate(J):
                matriz_x[i, j] = x[i, j].X
        
        houses_x = sum(x[i, j].X for i in I for j in J)
        no_asiste_indices = [i for i in I if all(x[i, j].X == 0 for j in J)]
        no_asiste_count = len(no_asiste_indices)
        MARGEN = sum(x[i,j].X * margin[i][j] for i in I for j in J)
        
        if dump != 0:

          z_dump=sum(z[j].X for j in J)
          reciclado= sum(qi[i]*x[i,j].X for i in I for j in J) - z_dump
          landfill= sum(qi[i] for i in I) - (reciclado + z_dump)
        else:
          #for i in I:
            landfill = sum(qi[i] for i in I)-sum(qi[i]*x[i,j].X for i in I for j in J)
            reciclado = sum(qi[i]*x[i,j].X for i in I for j in J)
        print("#############################################################################")
    
        print(f"con {cj} como costo de adm y {nr} como costo NR, se abren: {sum(y[j,w].X for j in J for w in W)} locales")
        print(f"La relacion de dumping es: {relation_values}")
        #print("Locales disponibles: "+ str(locals))
        if dump != 0:
            print(f"Dumping: {z_dump}")
            print(f"Landfill: {landfill}" )
            print(f"Reciclado: {reciclado}")
        else:
          print(f"Landfill: {landfill}" )
          print(f"Reciclado: {reciclado}")
         
        
        print(f"Cantidad de casas que reciclan: {houses_x} de {houses}")
        #print("La cantidad reciclada es de "+str(reciclado))
        print(f"Número de usuarios que no asisten a ningún punto j: {no_asiste_count}")
        for i in I:
          for j in J:
            if x[i,j].X > 0:
              print(f"El individuo {i+1} asiste al punto {j+1} y lleva {qi[i]*x[i,j].X}")
        for j in J:
            print(f"El estado del punto {j+1} es {sum(y[j,w].X for w in W)}")
    
        if dump == 0:
          tot = sum (qi[i] for i in I)
          reci = sum(qi[i]*x[i,j].X for i in I for j in J)
          print(f"El valor no reciclado es {(tot-reci)*nr}")
          print(sum(qi[i] for i in I))
          print(f"El valor manage es {sum(x[i,j].X*qi[i]for i in I for j in J)*cj }")
          print(f"El valor de costos fijos es {sum(sum(f_jw[w]*warehouse_y[j,w].X for w in W) for j in J)}")
          #print(f"Obj = {(sum(qi[i] for i in I) - sum(x[i,j]*qi[i] for i in I for j in J))*nr + sum(x[i,j]*qi[i] for i in I for j in J)*cj + sum(y[j,w] for w in W for j in J)*fj }")
        #else:
    
        objective_value = m.objVal
        print(f"El valor de la funcion objetivo es: ${objective_value}")
        print(cj)
        valores_y = {j: {w: y[j, w].X for w in W} for j in J}
        #valores_z = {j: z[j].X for j in J}
        if dump != 0:
          print(f"Total de basura a repartir: {sum(qi[i]for i in I)}")
          vector_de_unos = np.ones(len(I))
          en_j = {i: sum(x[i,j].X for j in J) for i in I}
          no_van = {i: vector_de_unos[i] - en_j[i] for i in I}
          norec= sum(qi[i]*no_van[i] for i in I)*nr
          por_j = []
          for j in J:
            print(f"Al punto {j+1} asisten: {sum(x[i,j].X for i in I)} personas")
            por_j.append(sum(x[i,j].X*qi[i] for i in I))
    
          return landfill,reciclado,z_dump,warehouse_y,objective_value,sum(f_jw[w]*valores_y[j][w] for j in J for w in W ), cj*((sum(x[i,j].X*qi[i] for i in I for j in J)) - sum(z[j].X for j in J)),  nr*relation_values*sum(z[j].X for j in J), norec, por_j, matriz_x,MARGEN
        else:
          vector_de_unos = np.ones(len(I))
          en_j = {i: sum(x[i,j].X for j in J) for i in I}
          no_van = {i: vector_de_unos[i] - en_j[i] for i in I}
          norec= sum(qi[i]*no_van[i] for i in I)*nr
          por_j = []
          for j in J:
          #  print(f"Al punto {j+1} asisten: {sum(model.x[i,j].value for i in I)} personas")
            por_j.append(sum(x[i,j].X*qi[i] for i in I))
          #return landfill,reciclado,warehouse_y,objective_value,sum(f_jw[w]*y[j,w]for j in J for w in W ),cj*sum(x[i,j]*qi[i] for i in I for j in J), norec, por_j, matriz_x

    else:
        print("Infactible")

land_wel_dump = []
fun_obj = []

for iteration in range(len(new_cj)):
    if dump != 0:
        obj_value = []
        for relation in relation_cost_dump:
            start = time.time()
            result = optimize_recycling_level_dum(new_cj[iteration], new_nr[iteration], dump, relation, binary=eleccion == 'Binary')
            if result is not None:
                landfill, reciclado, z, warehouse_y, objective,fijo,variable,dumpingg,norecicla, cantidad_por_j, i_j, MARGEN = result
                var = [landfill, reciclado, z]
                obj = objective
                land_wel_dump.append(var)
                obj_value.append(obj)
                print(f"El costo de abir es: ${fijo}, administrar: ${variable},dumping: ${dumpingg},social: ${norecicla}")
            else:
                print("No se encontró una solución óptima.")
            final = time.time()
            print("El tiempo de cálculo es " + str(int(final - start)) + " segundos, o " + str(int((final - start)/60))+ " minutos")
        fun_obj.append(obj_value)

    else:
          start = time.time()
          result = optimize_recycling_level_dum(new_cj[iteration], new_nr[iteration], dump, 0, binary=eleccion)
          if result is not None:
              landfill, reciclado, warehouse_y, objective,fijo,variable,norecicla, cantidad_por_j, i_j = result
              obj = objective
              fun_obj.append(obj)
              print(f"El costo de abir es: ${fijo}, administrar: ${variable},social: ${norecicla}")
          else:
              print("No se encontró una solución óptima.")
          final = time.time()
          print("El tiempo de cálculo es " + str(int(final - start)) + " segundos, o " + str(int((final - start)/60))+ " minutos")

      #for i in I:
          #print(sum(value(model.x[i, j]) for j in J))


locales_totales = warehouse_y
Totales_por_j = cantidad_por_j
landfill = landfill
cada_i_j = i_j
print(cada_i_j)
print(locales_totales)
print(Totales_por_j)
if dump !=0:
  print(MARGEN)
#%%
print(locales)
  #%% GRAFICOS#
  
separacion = 0.5
# Crear un gráfico con Flechas
plt.figure(figsize=(T/4, T/4))

# Dibujar los locales
for j, local in enumerate(locales):

    estado = locales_totales[j]
    if sum(estado[w] for w in W) == 1:
      if 1 == 0:#warehouse_y[j][1] == 1:
        if Totales_por_j[j] <= dyj[1]:
          color = "green"
          size = 10*T + 10*T * Totales_por_j[j]  # Ajustar el tamaño según la cantidad depositada
          plt.scatter(local[1], local[0], c=color, s=size, alpha=0.7)
        else:
          color = "yellow"
          size = 10*T + 10*T * Totales_por_j[j]  # Ajustar el tamaño según la cantidad depositada
          plt.scatter(local[1], local[0], c=color, s=size, alpha=0.7)
          color_1 = "green"
          size_1 = 10*T + 10*T * dyj[1]  # Ajustar el tamaño según la cantidad depositada
          plt.scatter(local[1], local[0], c=color_1, s=size_1, alpha=0.7)
      else:
        if Totales_por_j[j] <= dyj[0]:
          color = "green"
          size = 10*T + 10*T * Totales_por_j[j]  # Ajustar el tamaño según la cantidad depositada
          plt.scatter(local[1], local[0], c=color, s=size, alpha=0.7)
        else:
          color = "yellow"
          size = 10*T + 10*T * Totales_por_j[j]  # Ajustar el tamaño según la cantidad depositada
          plt.scatter(local[1], local[0], c=color, s=size, alpha=0.7)
          color_1 = "green"
          size_1 = 10*T + 10*T * dyj[0]  # Ajustar el tamaño según la cantidad depositada
          plt.scatter(local[1], local[0], c=color_1, s=size_1, alpha=0.7)

    else:
        color = "red"
        size = 10*T + 10*T * Totales_por_j[j]  # Ajustar el tamaño según la cantidad depositada
        plt.scatter(local[1], local[0], c=color, s=size, alpha=0.7)


    #color = 'green' if estado == 1 else 'red'
    #size = 10*T + 10*T * Totales_por_j[j]  # Ajustar el tamaño según la cantidad depositada
    #plt.scatter(local[1], local[0], c=color, s=size, alpha=0.7)

    # Agregar etiqueta de cantidad depositada
    plt.text(local[1], local[0], str(int(Totales_por_j[j])), fontsize=0.3*T, ha='center', va='center', color='white')
    # Agregar índice del local
    plt.text(local[1], local[0] - separacion*3, f'{j+1}', fontsize=0.1*T, ha='right', va='bottom', color='black')

plt.scatter(20, 5, c="purple", s= 10*T + 10*T*landfill, alpha=0.7)
plt.text(20,5 + 5*separacion, "Landfill", fontsize=0.2*T, ha='center', va='top', color='black')
plt.text(20 ,5 , str(landfill), fontsize=0.3*T, ha='center', va='center', color='black')

# Dibujar los usuarios y flechas
for i, usuario in enumerate(usuarios):
    asiste = np.any(cada_i_j[i, :])  # Verificar si el usuario asiste a algún local
    color_usuario = 'blue' if asiste else 'red'
    marker = 'X'
    plt.scatter(usuario[1], usuario[0], c=color_usuario, marker=marker, s=4*T, label=f'{i+1}')

    # Agregar flechas para indicar la asistencia a locales abiertos
    for j, local_estado in enumerate(locales_totales):
        if asiste and local_estado == 1 and cada_i_j[i, j] == 1:
            plt.arrow(usuario[1], usuario[0], locales[j][1] - usuario[1], locales[j][0] - usuario[0],
                      head_width=0.2, head_length=0.2, fc='darkgoldenrod', ec='darkgoldenrod', alpha=0.7)

    # Agregar etiqueta de usuario y local al que asiste
    if asiste:
        for j, local_estado in enumerate(locales_totales):
            if local_estado == 1 and cada_i_j[i, j] == 1:
                plt.text(usuario[1], usuario[0]-0.4*5, f'{i+1},{j+1}', fontsize=0.1*T, ha='right', va='bottom', color='black')

    else:
        for j, local_estado in enumerate(locales_totales):
            if local_estado == 0 and cada_i_j[i,j] == 0:
                plt.text(usuario[1], usuario[0]-0.4*5, f'{i+1}', fontsize=0.1*T, ha='right', va='bottom', color='black')
    # Agregar cantidad disponible en la esquina superior izquierda de cada usuario
    #plt.text(usuario[1], usuario[0]+0.4*5, f'{qi[i]}', fontsize=0.1*T, ha='left', va='top', color='black')

# Configuración del gráfico
#plt.title('Usuarios, Locales y Asistencia')
#plt.xlabel('Coordenada X')
#plt.ylabel('Coordenada Y')
plt.grid()
plt.xlim(0, T)
plt.ylim(0, T)

# Leyenda personalizada
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Abierto'),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Cerrado'),
           plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="yellow", markersize=10, label="Dumping"),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Landfill'),
           plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='blue', markersize=10, label='Usuario Asiste')]
#plt.legend(handles=handles, title='Legenda')

# Mostrar el gráfico
plt.show()


