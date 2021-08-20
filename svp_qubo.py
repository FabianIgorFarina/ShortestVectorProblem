from sympy import *
import numpy as np
import pandas as pd
import itertools

def max_value_of_base(base, power):
    if power > 1:
        return (base-1)*base**power + max_value_of_base(base, power-1)
    return base + 1

def calculate_gram_matrix(lattice):
    (d1, d2) = shape(lattice)
    return Matrix([[lattice.row(i)*lattice.row(j).T for i in range(0, d1)] for j in range(0,d2)])


def max_num_qubits_per_qudit(lattice, qudit_type):
    (d1, d2) = shape(lattice)
    if qudit_type == 'BINARY':
        return int(np.ceil((3.0/2.0)*shape(lattice)[0]*np.log2(shape(lattice)[0])+shape(lattice)[0]+np.log2(float(np.absolute(lattice.det())))))
    elif qudit_type == 'HAMMING':
        return int(2.0*np.power(d1, 5.0/2.0)*np.power(np.absolute(lattice.det()), (1.0/d1)))
    elif qudit_type == 'TERNARY':
        return int(1+np.floor(np.log(np.ceil((3.0/2.0)*d1*np.log2(d1)+d1+np.log2(float(np.absolute(lattice.det())))))/log(2.0)))*2
    elif qudit_type == 'REAL_TERNARY':
        return int(1+np.floor(np.log(np.ceil((3.0/2.0)*d1*np.log2(d1)+d1+np.log2(float(np.absolute(lattice.det())))))/log(2.0)))
    else:
        return qudit_type


def symbols_to_dict(hamiltonian, num_qubits):
    Qu = hamiltonian.as_coefficients_dict()

    constant = 0
    linear = {}
    quadratic = {}
    Qu_ = iter(Qu)
    for i in range(0, len(Qu)):
        key = next(Qu_)
        if i == 0:
            constant = Qu[key]
        elif not '*' in str(key):
            linear[int(str(key)[1:])] = Qu[key]
        elif not '**' in str(key):
            keys = str(key).split('*')
            quadratic[(int(keys[0][1:]), int(keys[1][1:]))] = Qu[key]
        else:
            keys = str(key).split('*')
            quadratic[(int(keys[0][1:]), int(keys[0][1:]))] = Qu[key]

    return [constant, linear, quadratic]


def initialize_coefficient_function(lattice, qudit_type):
    shape_qubits = [num_qudits, dim_qudits] = [shape(lattice)[0], max_num_qubits_per_qudit(lattice, qudit_type)]
    num_qubits = num_qudits*dim_qudits
    gram_matrix = calculate_gram_matrix(lattice)
    sym = symbols('s:'+str(shape_qubits[0]*shape_qubits[1]))
    return shape_qubits, num_qubits, gram_matrix, sym


def hamiltonian_from_operators(shape_qubits, gram_matrix, qudit_operatator, sym, num_qubits, replace=True):
    hamiltonian = 0
    for i in range(0,shape_qubits[0]):
        for j in range(0,shape_qubits[0]):
            hamiltonian += gram_matrix[i,j]*qudit_operatator[i]*qudit_operatator[j]
    
    if replace:
        replacements = [(sym[i]**2, sym[i]) for i in range(num_qubits)]
        return hamiltonian.expand().subs(replacements)
    return hamiltonian.expand()


def qubo_coefficients_binary(lattice):
    """
    Calculates the coefficients for SVP as QUBO with binary qudits. Returns the constant, liear and quadratic terms.
    """
    lattice = Matrix(lattice)
    shape_qubits, num_qubits, gram_matrix, sym = initialize_coefficient_function(lattice, 'BINARY')
    
    qudit_operatator = []
    for i in range(0,shape_qubits[0]):
        expr = sym[i*shape_qubits[1]]
        for j in range(1, shape_qubits[1]):
            expr += (2**j)*sym[i*shape_qubits[1]+j]
        expr -= 2**(shape_qubits[1]-1)
        qudit_operatator.append(expr)
    
    hamiltonian = hamiltonian_from_operators(shape_qubits, gram_matrix, qudit_operatator, sym, num_qubits)

    return symbols_to_dict(hamiltonian, num_qubits) # [constant, linear, quadratic]


def qubo_hamiltonian_binary(lattice):
    """
    Calculates the hamiltonian for SVP as QUBO with binary qudits. Returns the constant, liear and quadratic terms.
    """
    lattice = Matrix(lattice)
    shape_qubits, num_qubits, gram_matrix, sym = initialize_coefficient_function(lattice, 'BINARY')
    
    qudit_operatator = []
    for i in range(0,shape_qubits[0]):
        expr = sym[i*shape_qubits[1]]
        for j in range(1, shape_qubits[1]):
            expr += (2**j)*sym[i*shape_qubits[1]+j]
        expr -= 2**(shape_qubits[1]-1)
        qudit_operatator.append(expr)
    
    return sym, hamiltonian_from_operators(shape_qubits, gram_matrix, qudit_operatator, sym, num_qubits)


def qubo_coefficients_hamming(lattice):
    """
    Calculates the coefficients for SVP as QUBO with hamming qudits. Returns the constant, liear and quadratic terms.
    """
    lattice = Matrix(lattice)
    shape_qubits, num_qubits, gram_matrix, sym = initialize_coefficient_function(lattice, 'HAMMING')
    
    qudit_operatator = []
    for i in range(0,shape_qubits[0]):
        expr = sym[i*shape_qubits[1]]
        for j in range(1, shape_qubits[1]):
            expr += sym[i*shape_qubits[1]+j]
        expr -= int(shape_qubits[1]/2.0)
        qudit_operatator.append(expr)
    
    hamiltonian = hamiltonian_from_operators(shape_qubits, gram_matrix, qudit_operatator, sym, num_qubits)

    return symbols_to_dict(hamiltonian, num_qubits) # [constant, linear, quadratic]


def qubo_hamiltonian_hamming(lattice):
    """
    Calculates the hamiltonian for SVP as QUBO with hamming qudits. Returns the constant, liear and quadratic terms.
    """
    lattice = Matrix(lattice)
    shape_qubits, num_qubits, gram_matrix, sym = initialize_coefficient_function(lattice, 'HAMMING')
    
    qudit_operatator = []
    for i in range(0,shape_qubits[0]):
        expr = sym[i*shape_qubits[1]]
        for j in range(1, shape_qubits[1]):
            expr += sym[i*shape_qubits[1]+j]
        expr -= int(shape_qubits[1]/2.0)
        qudit_operatator.append(expr)
    
    return sym, hamiltonian_from_operators(shape_qubits, gram_matrix, qudit_operatator, sym, num_qubits)


def qubo_coefficients_ternary(lattice):
    """
    Calculates the coefficients for SVP as QUBO with ternary and partially hamming qudits. Returns the constant, liear and quadratic terms.
    """
    lattice = Matrix(lattice)
    shape_qubits, num_qubits, gram_matrix, sym = initialize_coefficient_function(lattice, 'TERNARY')

    qudit_operatator = []
    for i in range(0,shape_qubits[0]):
        expr = sym[i*shape_qubits[1]] + sym[i*shape_qubits[1]+1]
        for j in range(2, shape_qubits[1],2):
            expr += int(3**(j/2))*(sym[i*shape_qubits[1]+j] + sym[i*shape_qubits[1]+j+1])
        expr -= int(max_value_of_base(3, shape_qubits[1])/2.0)
        qudit_operatator.append(expr)

    hamiltonian = hamiltonian_from_operators(shape_qubits, gram_matrix, qudit_operatator, sym, num_qubits)

    return symbols_to_dict(hamiltonian, num_qubits) # [constant, linear, quadratic]


def qubo_hamiltonian_ternary(lattice):
    """
    Calculates the hamiltonian for SVP as QUBO with ternary and partially hamming qudits. Returns the constant, liear and quadratic terms.
    """
    lattice = Matrix(lattice)
    shape_qubits, num_qubits, gram_matrix, sym = initialize_coefficient_function(lattice, 'TERNARY')

    qudit_operatator = []
    for i in range(0,shape_qubits[0]):
        expr = sym[i*shape_qubits[1]] + sym[i*shape_qubits[1]+1]
        for j in range(2, shape_qubits[1],2):
            expr += int(3**(j/2))*(sym[i*shape_qubits[1]+j] + sym[i*shape_qubits[1]+j+1])
        expr -= int(max_value_of_base(3, shape_qubits[1]-1)/2.0)
        qudit_operatator.append(expr)

    return sym, hamiltonian_from_operators(shape_qubits, gram_matrix, qudit_operatator, sym, num_qubits)


def qubo_coefficients_real_ternary(lattice):
    """
    Calculates the coefficients for SVP as QUBO with qutrits Returns the constant, liear and quadratic terms.
    """
    lattice = Matrix(lattice)
    shape_qubits, num_qubits, gram_matrix, sym = initialize_coefficient_function(lattice, 'REAL_TERNARY')

    qudit_operatator = []
    for i in range(0,shape_qubits[0]):
        expr = sym[i*shape_qubits[1]]
        for j in range(1, shape_qubits[1]):
            expr += int(3**(j))*(sym[i*shape_qubits[1]+j])
        expr -= int(max_value_of_base(3, shape_qubits[1])/2.0)
        qudit_operatator.append(expr)
    
    hamiltonian = hamiltonian_from_operators(shape_qubits, gram_matrix, qudit_operatator, sym, num_qubits, False)
    
    return symbols_to_dict(hamiltonian, num_qubits) # [constant, linear, quadratic]

def qubo_hamiltonian_real_ternary(lattice):
    """
    Calculates the hamiltonian for SVP as QUBO with qutrits. Returns the constant, liear and quadratic terms.
    """
    lattice = Matrix(lattice)
    shape_qubits, num_qubits, gram_matrix, sym = initialize_coefficient_function(lattice, 'REAL_TERNARY')

    qudit_operatator = []
    for i in range(0,shape_qubits[0]):
        expr = sym[i*shape_qubits[1]]
        for j in range(1, shape_qubits[1]):
            expr += int(3**(j))*(sym[i*shape_qubits[1]+j])
        expr -= int(max_value_of_base(3, shape_qubits[1]-1)/2.0)
        qudit_operatator.append(expr)
    
    return sym, hamiltonian_from_operators(shape_qubits, gram_matrix, qudit_operatator, sym, num_qubits, False)


def sample_qubits(sym, hamiltonian, qubits_per_qudit=1):
    Energy = lambdify([sym], hamiltonian, 'numpy')
    num_qubits = len(sym)
    if qubits_per_qudit == -1:
        qubits_per_qudit = num_qubits
    labels = [i for i in range(0,num_qubits)]
    labels.append('energy')
    row_DataFrame = {}
    for label in labels:
        row_DataFrame[label] = 0
    df = pd.DataFrame(columns=labels)
    for num in itertools.product([0,1], repeat=num_qubits):
        for i in range(0, len(num)):
            row_DataFrame[i] = num[i]
        row_DataFrame['energy'] = Energy(num)
        df_temp = pd.DataFrame(row_DataFrame, columns=labels, 
            index=[str([sum([num[i]*2**(i%qubits_per_qudit) 
            for i in range(j*qubits_per_qudit, (j+1)*(qubits_per_qudit))]) - 2**(qubits_per_qudit-1)
            for j in range(0, int(num_qubits/qubits_per_qudit))])])
        df = pd.concat([df, df_temp])
    return df


def sample_qubits_binary(lattice):
    sym, hamiltonian = qubo_hamiltonian_binary(lattice)
    num_qubits = len(sym)
    qubits_per_qudit = max_num_qubits_per_qudit(Matrix(lattice), 'BINARY')
    Energy = lambdify([sym], hamiltonian, 'numpy')
    labels = [i for i in range(0,num_qubits)]
    labels.append('energy')
    row_DataFrame = {}
    for label in labels:
        row_DataFrame[label] = 0
    df = pd.DataFrame(columns=labels)
    for num in itertools.product([0,1], repeat=num_qubits):
        for i in range(0, len(num)):
            row_DataFrame[i] = num[i]
        row_DataFrame['energy'] = Energy(num)
        df_temp = pd.DataFrame(row_DataFrame, columns=labels, 
            index=[str([sum([num[i]*2**(i%qubits_per_qudit) 
            for i in range(j*qubits_per_qudit, (j+1)*(qubits_per_qudit))]) - 2**(qubits_per_qudit-1)
            for j in range(0, int(num_qubits/qubits_per_qudit))])])
        df = pd.concat([df, df_temp])
    return df


def sample_qutrits(sym, hamiltonian, qutrits_per_qudit=1):
    Energy = lambdify([sym], hamiltonian, 'numpy')
    num_qutrits = len(sym)
    if qutrits_per_qudit == -1:
        qutrits_per_qudit = num_qutrits
    labels = [i for i in range(0,num_qutrits)]
    labels.append('energy')
    row_DataFrame = {}
    for label in labels:
        row_DataFrame[label] = 0
    df = pd.DataFrame(columns=labels)
    for num in itertools.product([0,1,2], repeat=num_qutrits):
        for i in range(0, len(num)):
            row_DataFrame[i] = num[i]
        row_DataFrame['energy'] = Energy(num)
        df_temp = pd.DataFrame(row_DataFrame, columns=labels, index=[str([sum([num[i]*3**(i%qutrits_per_qudit) 
            for i in range(j*qutrits_per_qudit, (j+1)*(qutrits_per_qudit))]) - int(max_value_of_base(3, qutrits_per_qudit-1)/2.0)
            for j in range(0, int(num_qutrits/qutrits_per_qudit))])])
        df = pd.concat([df, df_temp])
    return df


def sample_qutrits_ternary(lattice):
    sym, hamiltonian = qubo_hamiltonian_real_ternary(lattice)
    num_qutrits = len(sym)
    qutrits_per_qudit = max_num_qubits_per_qudit(Matrix(lattice), 'REAL_TERNARY')
    Energy = lambdify([sym], hamiltonian, 'numpy')
    labels = [i for i in range(0,num_qutrits)]
    labels.append('energy')
    row_DataFrame = {}
    for label in labels:
        row_DataFrame[label] = 0
    df = pd.DataFrame(columns=labels)
    for num in itertools.product([0,1,2], repeat=num_qutrits):
        for i in range(0, len(num)):
            row_DataFrame[i] = num[i]
        row_DataFrame['energy'] = Energy(num)
        df_temp = pd.DataFrame(row_DataFrame, columns=labels, index=[str([sum([num[i]*3**(i%qutrits_per_qudit) 
            for i in range(j*qutrits_per_qudit, (j+1)*(qutrits_per_qudit))]) - int(max_value_of_base(3, qutrits_per_qudit-1)/2.0)
            for j in range(0, int(num_qutrits/qutrits_per_qudit))])])
        df = pd.concat([df, df_temp])
    return df


def sample_qudits(sym, hamiltonian, qudit_values, qudits_per_qudit=-1):
    Energy = lambdify([sym], hamiltonian, 'numpy')
    num_qudits = len(sym)
    if qudits_per_qudit == -1:
        qudits_per_qudit = num_qudits
    labels = [i for i in range(0,num_qudits)]
    labels.append('energy')
    row_DataFrame = {}
    for label in labels:
        row_DataFrame[label] = 0
    df = pd.DataFrame(columns=labels)
    for num in itertools.product(qudit_values, repeat=num_qudits):
        for i in range(0, len(num)):
            row_DataFrame[i] = num[i]
        row_DataFrame['energy'] = Energy(num)
        df_temp = pd.DataFrame(row_DataFrame, columns=labels, index=[str([sum([num[i]*len(qudit_values)**(i%qudits_per_qudit) 
            for i in range(j*qudits_per_qudit, (j+1)*(qudits_per_qudit))]) - int(max_value_of_base(len(qudit_values), qudits_per_qudit)/2.0)
            for j in range(0, int(num_qudits/qudits_per_qudit))])])
        df = pd.concat([df, df_temp])
    return df