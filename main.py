import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import networkx as nx
from collections import defaultdict, deque

class TreeNode:
    """Clase para representar un nodo del árbol sintáctico"""
    def __init__(self, value, node_type="leaf"):
        self.value = value
        self.node_type = node_type  # leaf, unary, binary
        self.left = None
        self.right = None
        self.children = []  # Para operadores con múltiples operandos
    
    def add_child(self, child):
        """Añade un hijo al nodo"""
        self.children.append(child)
    
    def __str__(self):
        return f"{self.node_type}: {self.value}"

class NFAState:
    """Clase para representar un estado del AFN"""
    def __init__(self, state_id, is_final=False):
        self.state_id = state_id
        self.is_final = is_final
        self.transitions = defaultdict(list)  # símbolo -> lista de estados
    
    def add_transition(self, symbol, target_state):
        """Añade una transición desde este estado"""
        self.transitions[symbol].append(target_state)
    
    def __str__(self):
        return f"State {self.state_id}{'(F)' if self.is_final else ''}"
    
    def __repr__(self):
        return self.__str__()

class NFA:
    """Clase para representar un Autómata Finito No Determinista"""
    def __init__(self):
        self.states = {}
        self.start_state = None
        self.final_states = set()
        self.alphabet = set()
        self.state_counter = 0
    
    def create_state(self, is_final=False):
        """Crea un nuevo estado"""
        state = NFAState(self.state_counter, is_final)
        self.states[self.state_counter] = state
        if is_final:
            self.final_states.add(self.state_counter)
        self.state_counter += 1
        return state
    
    def set_start_state(self, state):
        """Establece el estado inicial"""
        self.start_state = state.state_id
    
    def add_transition(self, from_state, symbol, to_state):
        """Añade una transición al AFN"""
        from_state.add_transition(symbol, to_state.state_id)
        if symbol != 'ε':
            self.alphabet.add(symbol)
    
    def epsilon_closure(self, states):
        """Calcula la cerradura epsilon de un conjunto de estados"""
        closure = set(states)
        stack = list(states)
        
        while stack:
            current = stack.pop()
            if current in self.states:
                for next_state in self.states[current].transitions.get('ε', []):
                    if next_state not in closure:
                        closure.add(next_state)
                        stack.append(next_state)
        
        return closure
    
    def simulate(self, input_string):
        """Simula el AFN con una cadena de entrada"""
        if self.start_state is None:
            return False
        
        current_states = self.epsilon_closure({self.start_state})
        
        for symbol in input_string:
            next_states = set()
            for state_id in current_states:
                if state_id in self.states:
                    for next_state in self.states[state_id].transitions.get(symbol, []):
                        next_states.add(next_state)
            
            current_states = self.epsilon_closure(next_states)
            
            if not current_states:
                return False
        
        # Verificar si algún estado actual es final
        return bool(current_states.intersection(self.final_states))

def get_precedence(c):
    """
    Calculate precedence for regex operators.
    Precedences for REs:
    '(' -> 1
    '|' -> 2
    '.' -> 3
    '?' -> 4
    '*' -> 4
    '+' -> 4
    '^' -> 5
    """
    precedence_map = {
        '(': 1,
        '|': 2,
        '.': 3,
        '?': 4,
        '*': 4,
        '+': 4,
        '^': 5
    }
    return precedence_map.get(c, 0)

def preprocess_regex(regex):
    """
    Preprocess regex to handle escaped characters and convert extensions.
    Converts '+' to 'aa*' pattern and '?' to '(a|ε)' pattern.
    """
    result = []
    i = 0
    
    while i < len(regex):
        char = regex[i]
        
        if char == '\\' and i + 1 < len(regex):
            result.append(char)
            result.append(regex[i + 1])
            i += 2
            continue
        else:
            result.append(char)
            i += 1
    
    return ''.join(result)

def format_regex(regex):
    """
    Format regex by adding explicit concatenation operators ('.').
    Properly handles character classes, escaped characters, and multi-character tokens.
    """
    all_operators = ['|', '?', '+', '*', '∗', '^']
    binary_operators = ['^', '|']
    result = []
    
    cleaned_regex = regex.replace(' ', '')
    
    i = 0
    while i < len(cleaned_regex):
        char = cleaned_regex[i]
        
        # Handle escaped characters
        if char == '\\' and i + 1 < len(cleaned_regex):
            next_char = cleaned_regex[i + 1]
            if next_char in ['n', 't', 'r', 's', 'd', 'w']:
                token = char + next_char
                result.append(token)
                i += 2
            elif next_char in ['(', ')', '{', '}', '[', ']', '+', '*', '?', '|', '^', '.']:
                result.append(next_char) 
                i += 2
            else:
                token = char + next_char
                result.append(token)
                i += 2
        elif char == '[':
            j = i + 1
            while j < len(cleaned_regex) and cleaned_regex[j] != ']':
                j += 1
            if j < len(cleaned_regex):
                token = cleaned_regex[i:j+1]
                result.append(token)
                i = j + 1
            else:
                result.append(char)
                i += 1
        elif char == '{':
            j = i + 1
            while j < len(cleaned_regex) and cleaned_regex[j] != '}':
                j += 1
            if j < len(cleaned_regex): 
                token = cleaned_regex[i:j+1]
                result.append(token)
                i = j + 1
            else:
                result.append(char)
                i += 1
        else:
            result.append(char)
            i += 1
    
    final_result = []
    for i in range(len(result)):
        token = result[i]
        final_result.append(token)
        
        # Check if we need to add concatenation operator
        if i + 1 < len(result):
            next_token = result[i + 1]
            
            if (token != '(' and 
                token not in binary_operators and
                next_token != ')' and 
                next_token not in all_operators):
                final_result.append('.')
    
    return ''.join(final_result)

def infix_to_postfix(regex):
    """
    Convert infix regex to postfix using Shunting Yard algorithm.
    """
    postfix = []
    stack = []
    formatted_regex = format_regex(regex)
    operators = {'|', '.', '?', '*', '+', '^'}
    
    for c in formatted_regex:
        if c == '(':
            stack.append(c)
        elif c == ')':
            while stack and stack[-1] != '(':
                postfix.append(stack.pop())
            if stack:
                stack.pop()
        elif c in operators:
            # Handle operators
            while (stack and 
                   stack[-1] != '(' and
                   get_precedence(stack[-1]) >= get_precedence(c)):
                postfix.append(stack.pop())
            stack.append(c)
        else:
            postfix.append(c)
    
    # Pop remaining operators from stack
    while stack:
        postfix.append(stack.pop())
    
    return ''.join(postfix)

def postfix_to_syntax_tree(postfix):
    """
    Convierte una expresión en notación postfix a un árbol sintáctico
    """
    stack = []
    operators = {'|', '.', '?', '*', '+', '^'}
    
    for char in postfix:
        if char in operators:
            # Operador unario
            if char in {'*', '+', '?'}:
                if stack:
                    operand = stack.pop()
                    node = TreeNode(char, "unary")
                    node.left = operand
                    stack.append(node)
            # Operador binario
            elif char in {'|', '.', '^'}:
                if len(stack) >= 2:
                    right = stack.pop()
                    left = stack.pop()
                    node = TreeNode(char, "binary")
                    node.left = left
                    node.right = right
                    stack.append(node)
        else:
            # Operando (carácter)
            node = TreeNode(char, "leaf")
            stack.append(node)
    
    return stack[0] if stack else None

def copy_tree(node):
    """
    Crea una copia profunda de un árbol sintáctico
    """
    if node is None:
        return None
    
    new_node = TreeNode(node.value, node.node_type)
    new_node.left = copy_tree(node.left)
    new_node.right = copy_tree(node.right)
    
    return new_node

def simplify_extensions(tree):
    """
    Simplifica las extensiones '+' y '?' en el árbol
    '+' se convierte en 'aa*'
    '?' se convierte en '(a|ε)'
    """
    if tree is None:
        return None
    
    if tree.node_type == "unary":
        if tree.value == '+':
            # Simplificar primero el operando
            operand = simplify_extensions(tree.left)
            
            # Crear copia para el nodo estrella
            operand_copy = copy_tree(operand)
            star_node = TreeNode('*', "unary")
            star_node.left = operand_copy
            
            concat_node = TreeNode('.', "binary")
            concat_node.left = operand
            concat_node.right = star_node
            
            return concat_node
        elif tree.value == '?':
            # Simplificar primero el operando
            operand = simplify_extensions(tree.left)
            epsilon_node = TreeNode('ε', "leaf")
            
            or_node = TreeNode('|', "binary")
            or_node.left = operand
            or_node.right = epsilon_node
            
            return or_node
        else:
            tree.left = simplify_extensions(tree.left)
            return tree
    elif tree.node_type == "binary":
        tree.left = simplify_extensions(tree.left)
        tree.right = simplify_extensions(tree.right)
        return tree
    else:
        return tree

def thompson_construction(tree):
    """
    Construye un AFN usando el algoritmo de Thompson a partir de un árbol sintáctico
    """
    if tree is None:
        return None
    
    nfa = NFA()
    
    def build_nfa(node):
        """
        Función recursiva que construye el AFN para un nodo del árbol
        Retorna (estado_inicial, estado_final)
        """
        if node is None:
            return None, None
            
        if node.node_type == "leaf":
            # Caso base: símbolo terminal
            start = nfa.create_state()
            end = nfa.create_state(is_final=True)
            
            if node.value == 'ε':
                nfa.add_transition(start, 'ε', end)
            else:
                nfa.add_transition(start, node.value, end)
            
            return start, end
        
        elif node.node_type == "unary":
            if node.value == '*':
                # Cerradura de Kleene
                inner_start, inner_end = build_nfa(node.left)
                
                # Verificar que el operando sea válido
                if inner_start is None or inner_end is None:
                    return None, None
                
                start = nfa.create_state()
                end = nfa.create_state(is_final=True)
                
                # Transiciones epsilon
                nfa.add_transition(start, 'ε', inner_start)
                nfa.add_transition(start, 'ε', end)
                nfa.add_transition(inner_end, 'ε', inner_start)
                nfa.add_transition(inner_end, 'ε', end)
                
                # El estado final interno ya no es final
                inner_end.is_final = False
                nfa.final_states.discard(inner_end.state_id)
                
                return start, end
        
        elif node.node_type == "binary":
            if node.value == '.':
                # Concatenación
                left_start, left_end = build_nfa(node.left)
                right_start, right_end = build_nfa(node.right)
                
                # Verificar que ambos lados sean válidos
                if left_start is None or left_end is None or right_start is None or right_end is None:
                    return None, None
                
                # Conectar el final del izquierdo con el inicio del derecho
                nfa.add_transition(left_end, 'ε', right_start)
                
                # El estado final izquierdo ya no es final
                left_end.is_final = False
                nfa.final_states.discard(left_end.state_id)
                
                return left_start, right_end
            
            elif node.value == '|':
                # Unión
                left_start, left_end = build_nfa(node.left)
                right_start, right_end = build_nfa(node.right)
                
                # Verificar que ambos lados sean válidos
                if left_start is None or left_end is None or right_start is None or right_end is None:
                    return None, None
                
                start = nfa.create_state()
                end = nfa.create_state(is_final=True)
                
                # Transiciones epsilon desde el nuevo inicio
                nfa.add_transition(start, 'ε', left_start)
                nfa.add_transition(start, 'ε', right_start)
                
                # Transiciones epsilon hacia el nuevo final
                nfa.add_transition(left_end, 'ε', end)
                nfa.add_transition(right_end, 'ε', end)
                
                # Los estados finales internos ya no son finales
                left_end.is_final = False
                right_end.is_final = False
                nfa.final_states.discard(left_end.state_id)
                nfa.final_states.discard(right_end.state_id)
                
                return start, end
        
        return None, None
    
    start_state, final_state = build_nfa(tree)
    if start_state:
        nfa.set_start_state(start_state)
    
    return nfa

def visualize_nfa(nfa, title="AFN"):
    """
    Visualiza el AFN usando networkx y matplotlib
    """
    G = nx.DiGraph()
    
    # Agregar nodos
    for state_id, state in nfa.states.items():
        if state.is_final:
            G.add_node(state_id, node_type='final')
        elif state_id == nfa.start_state:
            G.add_node(state_id, node_type='start')
        else:
            G.add_node(state_id, node_type='normal')
    
    # Agregar aristas con etiquetas
    edge_labels = {}
    for state_id, state in nfa.states.items():
        for symbol, targets in state.transitions.items():
            for target in targets:
                edge_key = (state_id, target)
                if edge_key in edge_labels:
                    edge_labels[edge_key] += f", {symbol}"
                else:
                    edge_labels[edge_key] = symbol
                    G.add_edge(state_id, target)
    
    # Configurar la visualización
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Dibujar nodos con diferentes colores
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get('node_type', 'normal')
        if node_type == 'start':
            node_colors.append('lightgreen')
        elif node_type == 'final':
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightblue')
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
    
    # Dibujar etiquetas de las aristas
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def get_tree_height(node):
    """Calcula la altura del árbol"""
    if node is None:
        return 0
    if node.node_type == "leaf":
        return 1
    elif node.node_type == "unary":
        return 1 + get_tree_height(node.left)
    else:  # binary
        return 1 + max(get_tree_height(node.left), get_tree_height(node.right))

def get_tree_width(node):
    """Calcula el ancho del árbol (número de hojas)"""
    if node is None:
        return 0
    if node.node_type == "leaf":
        return 1
    elif node.node_type == "unary":
        return get_tree_width(node.left)
    else:  # binary
        return get_tree_width(node.left) + get_tree_width(node.right)

def visualize_tree(node, ax, x=0, y=0, width=1):
    """
    Visualiza el árbol sintáctico usando matplotlib
    """
    if node is None:
        return
    
    # Dibujar el nodo actual
    if node.node_type == "leaf":
        color = 'lightblue'
    elif node.node_type == "unary":
        color = 'lightgreen'
    else:  # binary
        color = 'lightcoral'
    
    # Crear caja para el nodo
    box = FancyBboxPatch((x-0.1, y-0.05), 0.2, 0.1,
                        boxstyle="round,pad=0.01",
                        facecolor=color,
                        edgecolor='black',
                        linewidth=1)
    ax.add_patch(box)
    
    # Añadir texto del nodo
    ax.text(x, y, node.value, ha='center', va='center', fontsize=10, fontweight='bold')
    
    if node.node_type == "unary":
        # Nodo unario
        child_x = x
        child_y = y - 0.3
        ax.plot([x, child_x], [y-0.05, child_y+0.05], 'k-', linewidth=1)
        visualize_tree(node.left, ax, child_x, child_y, width)
        
    elif node.node_type == "binary":
        # Nodo binario
        left_width = get_tree_width(node.left)
        right_width = get_tree_width(node.right)
        total_width = left_width + right_width
        
        if total_width > 0:
            left_x = x - (right_width / total_width) * width * 0.5
            right_x = x + (left_width / total_width) * width * 0.5
        else:
            left_x = x - 0.2
            right_x = x + 0.2
            
        child_y = y - 0.3
        
        # Conectar con hijos
        ax.plot([x, left_x], [y-0.05, child_y+0.05], 'k-', linewidth=1)
        ax.plot([x, right_x], [y-0.05, child_y+0.05], 'k-', linewidth=1)
        
        visualize_tree(node.left, ax, left_x, child_y, width * 0.5)
        visualize_tree(node.right, ax, right_x, child_y, width * 0.5)

def print_tree(node, prefix="", is_left=True):
    """
    Imprime el árbol en formato texto
    """
    if node is None:
        return
    
    print(prefix + ("└── " if is_left else "┌── ") + str(node))
    
    if node.node_type == "unary":
        print_tree(node.left, prefix + ("    " if is_left else "│   "), True)
    elif node.node_type == "binary":
        print_tree(node.left, prefix + ("    " if is_left else "│   "), True)
        print_tree(node.right, prefix + ("    " if is_left else "│   "), False)

def read_regex_from_file(filename):
    """
    Read regex and test strings from a text file.
    Expected format: regex on one line, test strings on next line (comma-separated)
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]
        
        regex_data = []
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                regex = lines[i]
                test_strings_line = lines[i + 1]
                # Parse test strings (comma-separated)
                test_strings = [s.strip() for s in test_strings_line.split(',')]
                # Convert empty string to actual empty string
                test_strings = ['' if s == '' else s for s in test_strings]
                regex_data.append((regex, test_strings))
            else:
                # If no test strings provided, use default ones
                regex_data.append((lines[i], ['a', 'b', 'ab', 'ba', 'aa', 'bb', '']))
        
        return regex_data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def process_regex(regex, index, test_strings=None):
    """
    Procesa una expresión regular completa: infix -> postfix -> árbol sintáctico -> AFN
    """
    print(f"\n{'='*60}")
    print(f"EXPRESIÓN REGULAR #{index}: {regex}")
    print(f"{'='*60}")
    
    # Paso 1: Preprocesamiento
    preprocessed = preprocess_regex(regex)
    print(f"1. Después del preprocesamiento: {preprocessed}")
    
    # Paso 2: Conversión infix a postfix
    postfix = infix_to_postfix(preprocessed)
    print(f"2. Notación postfix: {postfix}")
    
    # Paso 3: Crear árbol sintáctico
    tree = postfix_to_syntax_tree(postfix)
    print(f"3. Árbol sintáctico creado")
    
    # Paso 4: Simplificar extensiones
    simplified_tree = simplify_extensions(tree)
    print(f"4. Extensiones simplificadas")
    
    # Paso 5: Mostrar árbol en texto
    print(f"\n5. Árbol sintáctico (formato texto):")
    print_tree(simplified_tree)
    
    # Paso 6: Visualizar árbol
    print(f"\n6. Visualización del árbol:")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-get_tree_height(simplified_tree) * 0.4, 0.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    visualize_tree(simplified_tree, ax, 0, 0, 2)
    plt.title(f'Árbol Sintáctico - Expresión #{index}: {regex}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Paso 7: Construir AFN usando Thompson
    print(f"\n7. Construcción del AFN usando algoritmo de Thompson:")
    nfa = thompson_construction(simplified_tree)
    print(f"   AFN construido con {len(nfa.states)} estados")
    print(f"   Estado inicial: {nfa.start_state}")
    print(f"   Estados finales: {nfa.final_states}")
    print(f"   Alfabeto: {nfa.alphabet}")
    
    # Paso 8: Visualizar AFN
    print(f"\n8. Visualización del AFN:")
    visualize_nfa(nfa, f'AFN - Expresión #{index}: {regex}')
    
    # Paso 9: Simulación del AFN
    print(f"\n9. Simulación del AFN:")
    if test_strings is None:
        test_strings = get_test_strings()
    else:
        print(f"   Usando cadenas de prueba del archivo:")
        for s in test_strings:
            display_string = s if s else "ε (cadena vacía)"
            print(f"     '{display_string}'")
    
    for test_string in test_strings:
        result = nfa.simulate(test_string)
        status = "sí" if result else "no"
        display_string = test_string if test_string else "ε (cadena vacía)"
        print(f"   Cadena '{display_string}': {status}")
    
    return simplified_tree, nfa

def get_test_strings():
    """
    Solicita al usuario cadenas de prueba para simular el AFN
    """
    test_strings = []
    print("\n   Ingrese cadenas para probar el AFN (presione Enter sin texto para terminar):")
    
    while True:
        try:
            test_string = input("   Cadena: ").strip()
            if not test_string:
                break
            test_strings.append(test_string)
        except (EOFError, KeyboardInterrupt):
            break
    
    # Si no se ingresaron cadenas, usar algunas por defecto
    if not test_strings:
        test_strings = ["a", "b", "ab", "ba", "aa", "bb", "abb", ""]
        print("   Usando cadenas de prueba por defecto:")
        for s in test_strings:
            display_string = s if s else "ε (cadena vacía)"
            print(f"     '{display_string}'")
    
    return test_strings

def main():
    """
    Función principal que demuestra el algoritmo completo
    """
    print("ALGORITMO DE THOMPSON: EXPRESIONES REGULARES -> AFN")
    print("="*70)
    
    # Leer directamente desde el archivo expresiones_regulares.txt
    filename = "expresiones_regulares.txt"
    regex_data = read_regex_from_file(filename)
    
    if regex_data is None:
        print(f"Error: No se pudo leer el archivo '{filename}'")
        return
    
    print(f"Leyendo expresiones regulares y cadenas de prueba desde: {filename}")
    print("Expresiones encontradas:")
    for i, (regex, test_strings) in enumerate(regex_data, 1):
        print(f"  {i}. {regex}")
        print(f"     Cadenas de prueba: {', '.join([s if s else 'ε' for s in test_strings])}")
    
    # Procesar cada expresión regular
    results = []
    for i, (regex, test_strings) in enumerate(regex_data, 1):
        tree, nfa = process_regex(regex, i, test_strings)
        results.append((tree, nfa))
    
    print(f"\n{'='*70}")
    print("PROCESAMIENTO COMPLETADO")
    print(f"Se procesaron {len(regex_data)} expresiones regulares")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
