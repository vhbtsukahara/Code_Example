import pandas as pd
from scipy.io import loadmat
import numpy as np
from numpy import size, zeros, linspace, mean, where
from statistics import median
from matplotlib import pyplot as plt
from pgmpy.estimators import HillClimbSearch, K2Score
from graphviz import Digraph
from datetime import datetime
import pymc as pm


# Creating df1 for testing purposes
columns_df1 = ['Hospital', 'Identificador', 'Marca', 'Modelo', 'Tipo Equipamento', 'Contrato', 'N.º Série', 'Idade']
data_df1 = [
    ['Hospital A', 'EQ001', 'Marca X', 'Modelo A', 'Ventilador', 'SIM', 'SN001', 3],
    ['Hospital B', 'EQ002', 'Marca Y', 'Modelo B', 'Ventilador', 'NÃO', 'SN002', 7],
    ['Hospital C', 'EQ003', 'Marca Z', 'Modelo C', 'Monitor', 'SIM', 'SN003', 12],
    ['Hospital A', 'EQ004', 'Marca X', 'Modelo A', 'Ventilador', 'SIM', 'SN004', 15],
    ['Hospital D', 'EQ005', 'Marca W', 'Modelo D', 'Ultrassom', 'NÃO', 'SN005', 22]
]
df1 = pd.DataFrame(data_df1, columns=columns_df1)

# Creating df2 for testing purposes
columns_df2 = ['Identificador', 'Tipo de manutenção']
data_df2 = [
    ['EQ001', 'Manutenção Corretiva'],
    ['EQ001', 'Manutenção Programada'],
    ['EQ002', 'Manutenção Corretiva'],
    ['EQ003', 'Manutenção Corretiva'],
    ['EQ004', 'Manutenção Programada'],
    ['EQ005', 'Manutenção Corretiva'],
    ['EQ005', 'Manutenção Corretiva']
]
df2 = pd.DataFrame(data_df2, columns=columns_df2)

# Saving the DataFrames to Excel files to test the algorithm
df1.to_excel('listaeqptos_ciec_test.xlsx', index=False)
df2.to_excel('monitor_os_test.xlsx', index=False)

# Update plotting parameters for better visualization
plt.rcParams.update({'font.size': 22})  # Set font size for plots
plt.rcParams['figure.figsize'] = (18, 12)  # Set figure size
plt.rcParams['font.size'] = 24  # Set font size (again, potentially redundant)
plt.rcParams['image.cmap'] = 'plasma'  # Set color map for images
plt.rcParams['axes.linewidth'] = 2  # Set linewidth for plot axes

# Function to scale a dataset based on its discrete version
def scaled_dataset(df, df_discrete):
    DF = df_discrete.copy()  # Create a copy of the discrete dataframe
    cols = df.columns  # Get column names of the original dataframe
    for col in cols:
        discrete_values = list(set(df_discrete[col]))  # Get unique values from the discrete column
        for value in discrete_values:
            discrete = df_discrete[df_discrete[col] == value]  # Filter rows with the current discrete value
            continuous = df[df.index.isin(discrete.index)]  # Get corresponding rows from the original dataframe
            cont_val = median(continuous[col])  # Calculate the median of the continuous values
            DF.loc[DF[col] == value, col] = cont_val  # Replace the discrete values with the median
    return DF  # Return the scaled dataframe

# Function to adaptively bin data into discrete bins
def adaptive_bins(df, bins):
    teste = df.copy()  # Create a copy of the dataframe
    for h in teste.columns:
        x = list(teste[h])  # Get the column as a list

        x_max = max(x)  # Find the maximum value
        x_min = min(x)  # Find the minimum value
        N_MIN = 4  # Minimum number of bins (must be more than 1)
        N_MAX = bins  # Maximum number of bins
        N = range(N_MIN, N_MAX)  # Create a range of bin numbers
        N = np.array(N)
        D = (x_max - x_min) / N  # Calculate the bin size for each option
        C = zeros(shape=(size(D), 1))  # Initialize cost function array

        # Computation of the cost function for optimal bin size
        for i in range(size(N)):
            edges = linspace(x_min, x_max, N[i] + 1)  # Define the edges of the bins
            ki = plt.hist(x, edges)[0]  # Count the number of events in each bin
            plt.clf()  # Clear the plot to avoid overlapping
            plt.close()  # Close the plot
            k = mean(ki)  # Calculate the mean of the event counts
            v = sum((ki - k) ** 2) / N[i]  # Calculate the variance of event counts
            C[i] = (2 * k - v) / ((D[i]) ** 2)  # Compute the cost function

        # Select the optimal bin size
        cmin = min(C)  # Find the minimum cost value
        idx = where(C == cmin)  # Find the index of the minimum cost
        idx = int(idx[0])
        edges = linspace(x_min, x_max, N[idx] + 1)  # Define the optimal bin edges
        edges = edges.tolist()
        edges[len(edges) - 1] = edges[len(edges) - 1] + 0.5  # Adjust the last edge slightly

        label = np.arange(0, len(edges) - 1)  # Create labels for each bin
        label = label.tolist()
        teste[h] = list(pd.cut(x=teste[h], bins=edges, labels=label, right=False))  # Assign labels to bins
        teste[h] = teste[h].astype(np.int64)  # Convert labels to integers

    DF = scaled_dataset(df, teste)  # Scale the dataset based on the binned data

    return DF  # Return the discretized and scaled dataframe

# Function to generate a Bayesian Network from the dataframe
def bn_generator(df, n_parents=None):
    data = df.copy()  # Create a copy of the dataframe
    est = HillClimbSearch(data)  # Initialize the Hill Climb search for structure learning
    if n_parents:
        best_model = est.estimate(max_indegree=n_parents)  # Estimate the best model with a limit on parents
    else:
        best_model = est.estimate(max_iter=40)  # Estimate the best model with a default max iteration
    edges = list(best_model.edges())  # Extract the edges from the learned model

    return edges  # Return the edges of the Bayesian Network

# Function to generate a graph visualization of the Bayesian Network
def graph_generator(edges, df, filename):
    nodes = []
    for i in df.columns:
        nodes.append(i)  # Add all column names as nodes
    f = Digraph('finite_state_machine', filename)  # Initialize a graph
    f.attr(rankdir='LR', size='8,5')  # Set graph attributes
    f.concentrate = True  # Enable concentration of edges
    f.attr('node')
    for i in nodes:
        f.node(i, color='black', shape='circle', style='filled', fillcolor='lightcyan')  # Add nodes with attributes

    model = list(edges)
    for i in range(len(model)):
        f.edge(model[i][0], model[i][1])  # Add edges to the graph
    f.view()  # Render the graph

# Function to compute joint probability distribution of given nodes
def joint_distribution(nodes, df, numBins):
    data = np.array(df[nodes])  # Extract data for the given nodes
    jointProbs, edges = np.histogramdd(data, bins=numBins, density=True)  # Calculate joint histogram
    jointProbs /= jointProbs.sum()  # Normalize the joint probability distribution

    return jointProbs  # Return the joint probability distribution

# Function to compute conditional entropy H(Y|Z) for a set of variables
def H_conditional(y, Z, df, bins):
    tuples = [i for i in range(bins)]  # Create a range of possible values for each variable
    tuples = [tuples for j in range(len(Z) + 1)]  # Create a list of ranges for all variables including y and Z
    tuples = [element for element in itertools.product(*tuples)]  # Generate all combinations of possible values
    nodes_num = Z + [y]  # Nodes involved in the numerator
    nodes_den = Z  # Nodes involved in the denominator
    num = joint_distribution(nodes_num, df, bins)  # Calculate joint distribution for numerator
    den = joint_distribution(nodes_den, df, bins)  # Calculate joint distribution for denominator

    # Calculate conditional entropy
    Hy_given_z = [0 if (num[pos] == 0 or den[pos[:-1]] == 0) else num[pos] * log2(num[pos] / den[pos[:-1]]) for pos in tuples]

    return -sum(Hy_given_z)  # Return the conditional entropy

# Function to compute entropy of a set of labels
def entropy(labels, base=2):
    value, counts = np.unique(labels, return_counts=True)  # Get unique values and their counts
    return H(counts, base=base)  # Calculate entropy using scipy's entropy function

# Function to calculate link strength between two variables in a Bayesian Network
def link_strenght(var1, var2, df, edges, bins):
    G = DAG(edges)  # Create a Directed Acyclic Graph from the edges
    parents = G.get_parents(node=var2)  # Get parents of var2
    if var1 not in parents:
        return f"Erro: {var1} precisa ser pai de {var2}"  # Error if var1 is not a parent of var2
    if len(parents) == 1:
        return entropy(df[var2]) - H_conditional(var2, [var1] + parents, df, bins)  # Calculate link strength for a single parent

    parents = [x for x in parents if x != var1]  # Exclude var1 from parents
    LS = H_conditional(var2, parents, df, bins) - H_conditional(var2, [var1] + parents, df, bins)  # Calculate link strength

    return LS  # Return the link strength
