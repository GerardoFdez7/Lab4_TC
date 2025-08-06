# Laboratorio 4 - Teoría de la Computación

## Algoritmo de Thompson: Expresiones Regulares → AFN

Este proyecto implementa un algoritmo completo que convierte expresiones regulares a Autómatas Finitos No Deterministas (AFN) usando el algoritmo de Thompson, incluyendo visualización y simulación.

### Características

- **Algoritmo Shunting Yard**: Conversión de infix a postfix
- **Árbol Sintáctico**: Construcción de árbol a partir de notación postfix
- **Algoritmo de Thompson**: Construcción de AFN a partir del árbol sintáctico
- **Visualización**: Dibujo automático del árbol y AFN usando matplotlib y networkx
- **Simulación**: Verificación de cadenas contra el AFN generado
- **Lectura de Archivos**: Procesamiento de múltiples expresiones desde archivo

### Requisitos del Sistema

1. **Python 3.7 o superior**

   - Descargar desde: https://www.python.org/downloads/
   - Asegúrese de marcar "Add Python to PATH" durante la instalación

2. **Instalar dependencias**:

```bash
pip install -r requirements.txt
```

O manualmente:

```bash
pip install matplotlib>=3.5.0 numpy>=1.21.0 networkx>=2.8.0
```

### Uso

```bash
python main.py
```

El programa procesará automáticamente las expresiones regulares del archivo `expresiones_regulares.txt`.

### Formato del Archivo de Entrada

El archivo `expresiones_regulares.txt` debe contener pares de líneas:

- **Línea impar**: Expresión regular
- **Línea par**: Cadenas de prueba separadas por comas

```
(a* | b*)+
a,b,aa,bb,ab,ba,aaa,bbb,
((ε | a) | b*)*
,a,b,aa,bb,ab,ba,aaa,bbb
(a | b)* abb (a | b)*
abb,aabb,abba,aabba,babb,babba,a,b,ab
0? (1?)? 0*
,0,1,00,01,10,11,000,001,010,100
```

**Nota**: Para representar la cadena vacía (ε), use una cadena vacía en la lista (como al inicio de algunas líneas).

### Salida del Programa

Para cada expresión regular, el programa muestra:

1. **Preprocesamiento**: Expresión después del preprocesamiento
2. **Notación Postfix**: Conversión usando Shunting Yard
3. **Árbol Sintáctico**: Estructura del árbol en formato texto
4. **Visualización del Árbol**: Gráfico del árbol sintáctico
5. **Construcción del AFN**: Aplicación del algoritmo de Thompson
6. **Visualización del AFN**: Gráfico del autómata generado
7. **Simulación**: Prueba de cadenas contra el AFN
   - El programa usa las cadenas de prueba del archivo
   - Muestra "sí" si la cadena es aceptada, "no" en caso contrario

### Estructura del Código

#### Clases Principales

- `TreeNode`: Representa nodos del árbol sintáctico
- `NFAState`: Representa estados del AFN
- `NFA`: Representa el Autómata Finito No Determinista completo

#### Funciones Principales

- `infix_to_postfix()`: Algoritmo Shunting Yard
- `postfix_to_syntax_tree()`: Construcción del árbol sintáctico
- `simplify_extensions()`: Simplificación de `+` y `?`
- `thompson_construction()`: Algoritmo de Thompson para construir AFN
- `visualize_tree()`: Visualización gráfica del árbol
- `visualize_nfa()`: Visualización gráfica del AFN
- `process_regex()`: Procesamiento completo de una expresión

### Ejemplos Incluidos

1. `(a* | b*)+` - Unión de cerraduras de Kleene con extensión +
2. `((ε | a) | b*)*` - Cerradura de Kleene con unión y epsilon
3. `(a | b)* abb (a | b)*` - Patrón con concatenación
4. `0? (1?)? 0*` - Múltiples extensiones anidadas

### Algoritmo de Thompson

El algoritmo de Thompson construye un AFN a partir del árbol sintáctico usando las siguientes reglas:

- **Símbolo terminal**: Crea dos estados conectados por el símbolo
- **Concatenación**: Conecta el final del primer AFN con el inicio del segundo
- **Unión**: Crea nuevos estados inicial y final con transiciones ε
- **Cerradura de Kleene**: Añade transiciones ε para permitir repetición

### Notas Técnicas

- **Extensiones**: `+` se convierte en `aa*`, `?` se convierte en `(a|ε)`
- **Operadores**: Soporte para `|`, `*`, `+`, `?`, `.` (concatenación)
- **Visualización del Árbol**: Colores diferentes para hojas (azul), unarios (verde), binarios (rojo)
- **Visualización del AFN**: Estados iniciales (verde), finales (rojo), normales (azul)
- **Simulación**: Usa cerradura epsilon para manejar transiciones ε
