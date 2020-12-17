# Instructions
#######################
# python social-circles-discovery.py or python3 social-circles-discovery.py
#######################

# imports
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from multiprocessing import Pool

# Reading Signed network.
signed_network = nx.read_weighted_edgelist('soc-sign-Slashdot090221.txt.gz', comments='#', create_using=nx.DiGraph(), nodetype = int)

# Reading node features.
node_features ={}
no_of_features = 100
line_no = 0
with open("embedded-soc-sign-slashdot") as nf: 
    Lines = nf.readlines() 
    for line in Lines:
        #skip first line
        if line_no > 0:
            # splitting by space
            values = line.split()
            values = values[:no_of_features+1]
            index = 0
            # reading node features
            for val in values:
                if index == 0:
                    # reading nodeIds for first time
                    if line_no == 1:
                        node_features["nodeId"] = [int(val)]
                    else:
                        node_features["nodeId"].append(int(val))
                
                elif index > 0:
                    # reading features for the first time
                    if line_no == 1:
                        node_features["feature"+str(index)] = [float(val)]
                    else:
                        node_features["feature"+str(index)].append(float(val))
                index+=1
        line_no += 1

node_features_df = pd.DataFrame(node_features)

# Function to determine optimal number of clusters or number of active centers using elbow method

def elbowMethod(node_features_df):
  # To give equal importance to all features, we need to scale the continuous features. 
  # We will be using scikit-learn’s MinMaxScaler as the feature matrix is a mix of binary and continuous features . 
  mms = MinMaxScaler()
  mms.fit(node_features_df)
  node_features_df_transformed = mms.transform(node_features_df)

  Sum_of_squared_distances = []
  K = range(1,50)
  for k in K:
      km = KMeans(n_clusters=k, n_jobs=-1)
      km = km.fit(node_features_df_transformed)
      Sum_of_squared_distances.append(km.inertia_)

  plt.plot(K, Sum_of_squared_distances, 'bx-')
  plt.xlabel('k')
  plt.ylabel('Sum_of_squared_distances')
  plt.title('Elbow Method For Optimal k')
  plt.show()

# From above elbow plot method, it looks like optimal value of K is 7.
K = 7

# Extracting node Ids.
nodeIds = list(node_features_df["nodeId"])

# Function to calculate profile similarities.
# based on euclidean distance
def profSimilarity(nodeId, active_center, active_centers):
    nodeId_index = nodeIds.index(nodeId)
    ac_index = active_centers.index(active_center)
    # setting to 1 to avoid division by zero error
    sum = 1
    for i in range(0, no_of_features):
        sq_diff = (node_features_df["feature"+str(i+1)][nodeId_index] - node_features_df["feature"+str(i+1)][ac_index])**2
        sum += sq_diff

    # returning inverse as high value means less similarity.
    return 1/math.sqrt(sum)

# Function to determine whether edge exists or not.
def edgeExists(node1, node2):
    if signed_network.has_edge(node1, node2):
        return 1
    else:
        return 0

# Function to calculate strength of ties.
# For the computation,
# we have borrowed the idea of base node similarity,57
# where more the number of links a user shares with its
# neighbors, less becomes the strength of ties existing
def strengthOfTies(node1, node2):
    sum = 0
    for degree in list(dict(signed_network.out_degree([node1])).values()):
        sum+=degree
    for degree in list(dict(signed_network.in_degree([node2])).values()):
        sum+=degree

    sum -= 1
    
    if sum <= 0:
      return 0
    
    return 1/sum

# Function to find residual area (those neighbors of given nodes that are not in the social circle).
def residualArea(x,circle):
    residual = list(signed_network.neighbors(x))
    for re in residual:
        if re in circle:
            residual.remove(re)
    return residual

# Function to calculate degree centrality.
def degreeCentrality(x,circle):
    degree = 0 # in-degree + out-degree
    for c in circle:
        if signed_network.has_edge(x, c):
            degree+=1
        if signed_network.has_edge(c, x):
            degree+=1
    # error if len(circle) = 1
    if len(circle) ==1:
        deg_cen = degree
    else:
        deg_cen = degree/(len(circle)-1)
    return deg_cen

# Funtion to discover social circle using K-means.

def algorithm1(nodeIds,active_centers, add_trust_ftr = False):
    social_circles = {}
    nodeIdsWAC = list(set(nodeIds) - set(active_centers))
    for i in range (0, K):
        active_center = active_centers[i]
        social_circles[str(active_center)] = []
    for nodeId in nodeIdsWAC:
        maxS = 0
        # active_center which will be most similar to given node
        selectedAC = -1
        for i in range (0, K):
            active_center = active_centers[i]
            p1 = 0 # edge exists from active center to node
            p2 = 0 # edge exists from node to active center
            p3 = 0
            p4_1 = 0 # strength of ties between active center and node  
            p4_2 = 0 # strength of ties between node and active center 
            p5_1 = 0 # trust between active center and node
            p5_2 = 0 # trust between node and active center

            p1 = edgeExists(active_center, nodeId)
            p2 = edgeExists(nodeId, active_center)

            if p1 or p2:
              p3 = profSimilarity(nodeId, active_center, active_centers)

            if p1:
              p4_1 = strengthOfTies(active_center, nodeId)
              if add_trust_ftr:
                p5_1 = signed_network.get_edge_data(active_center, nodeId, default={'weight':0})['weight']
            
            if p2:
                p4_2 = strengthOfTies(nodeId, active_center)
                if add_trust_ftr:
                  p5_2 = signed_network.get_edge_data(nodeId, active_center, default={'weight':0})['weight']            

            if maxS < p1 + p2 + p3 + p4_1 + p4_2 + p5_1 + p5_2:
                maxS = p1 + p2 + p3 + p4_1 + p4_2 + p5_1 + p5_2
                selectedAC = active_center
      
        if selectedAC != -1:
            social_circles[str(selectedAC)].append(nodeId)

    return social_circles

# Initializing variables for Genetic Algorithm.

#center selection
##population size = 20
N = 20
population = []
random.seed(0)

# Generating random population (sets of active centers) randomly.

for i in range(0,N):
    selected = random.sample(nodeIds,K)
    if selected not in population:
        population.append(selected)

# Algorithm2_part1 takes one group of centers and return the fitness of it.

def algorithm2_part1(pop_n, add_trust_ftr = False):
    Xi = pop_n
    Cij = algorithm1(nodeIds,Xi, add_trust_ftr)
    
    obj=0
    for k in range(0,K):
        #initialize k=1 and obj=0
        xi = Xi[k]
        
        residual =  residualArea(xi,Cij[str(xi)])
        #deg_cen
        deg_cen_C = degreeCentrality(xi,Cij[str(xi)])
        deg_cen_R = degreeCentrality(xi,residual)
        
        #prof_sim
        prof_sim_C = 0
        for c in Cij[str(xi)]:
            prof_sim_C+=profSimilarity(c, xi,Xi)
        prof_sim_C = prof_sim_C/len(Cij)
        
        prof_sim_R = 0
        for r in residual:
            prof_sim_R+=profSimilarity(r, xi,Xi)
        if len(residual) != 0:
            prof_sim_R = prof_sim_R/len(residual)
        
        #str_C
        str_C = 0
        for c in Cij[str(xi)]:
            str_C+= strengthOfTies(c, xi)
            str_C+= strengthOfTies(xi, c)
        str_C = str_C/len(Cij)
        
        str_R = 0
        for r in residual:
            str_R+= strengthOfTies(r, xi)
            str_R+= strengthOfTies(xi, r)
        if len(residual) != 0:
             str_R = str_R/len(residual)

        #str_C
        trust_C = 0
        trust_R = 0
        if add_trust_ftr:
          for c in Cij[str(xi)]:
              trust_C += signed_network.get_edge_data(c, xi, default={'weight':0})['weight']
              trust_C += signed_network.get_edge_data(xi, c, default={'weight':0})['weight']
          trust_C = trust_C/len(Cij)
          
        if add_trust_ftr:  
          for r in residual:
              trust_R += signed_network.get_edge_data(r, xi, default={'weight':0})['weight']
              trust_R += signed_network.get_edge_data(xi, r, default={'weight':0})['weight']
          if len(residual) != 0:
              trust_R = trust_R/len(residual)
       
        
        obj+=deg_cen_C - deg_cen_R + prof_sim_C - prof_sim_R + str_C - str_R + trust_C - trust_R
    
    
    return obj/K

# Calculating fitness value of generated population.
#initialize n=1 and fitness = 0
fitness=[]
for n in range(0,N):
    #pick up the ith row from X_ij and Cij
    fit_val = algorithm2_part1(population[n])
    print("Population # " + str(n)+ " fitness value: " + str(fit_val))
    print(population[n])
    fitness.append(fit_val)

# Function to check the format of generated Q offspring (new set of active centers generated after crossover and mutation).
def check(matrix):
    for i in matrix:
        if len(i)!=8:
            print("error in row: ",i)

# Applying Crossover and mutation operation to determine best set of active centers.
# Part two
##Augmented matrix Q containing fitness value for each set of active center
Q=[]
for i in range(0,len(fitness)):
    Qi = population[i].copy()
    Qi.append(fitness[i])
    Q.append(Qi)

# re-sort in descending order
# since there are K elements in each set then we have fitness value at K index
fit_val_index = K
Q_desc= sorted(Q,key=lambda x:x[fit_val_index],reverse=True)

# Callback function for Globally optimum version of GA (gurantees best set of active centers but convergences slowly).

def parallel_insider_algorithm2_part2(i,Q_desc,K,nodeIds,signed_network,node_features_df, add_trust_ftr = False):
    import random
    #print(i)
    # Q_high = list(filter(lambda x:x[K]>1,Q_desc))
    # if len(Q_high) >10:
    #     Q_high = Q_high[:10]

    Q_high = Q_desc[:10]
    #print(Q_high)
    Q1 = random.choice(Q_high)

    #print(Q1,Q2)
    x1 = Q1[:-1]
    x1_fit = Q1[-1]
    Q2 = random.choice(Q_high)
    x2 = Q2[:-1]
    x2_fit = Q2[-1]
    while x2 == x1:
        Q2 = random.choice(Q_high)
        x2 = Q2[:-1]
        x2_fit = Q2[-1]
    #random.seed(0)
    randc_pos = random.randint(1,K)

    x1_new = x1[:randc_pos]+x2[randc_pos:]
    x2_new = x2[:randc_pos]+x1[randc_pos:]

    x1_new_fit = algorithm2_part1(x1_new, add_trust_ftr)
    x2_new_fit = algorithm2_part1(x2_new, add_trust_ftr)

    max_fit = max(x1_fit,x2_fit,x1_new_fit,x2_new_fit)
    value="null"
    if(x1_fit == max_fit):
        x = x1
        value="x1"
        #print("x1 is the best")
    elif (x2_fit == max_fit):
        x = x2
        value="x2"
        #print("x2 is the best")
    elif (x1_new_fit == max_fit):
        x = x1_new
        value="x1_new"
        #print("x1 new is the best")
    else:
        x = x2_new
        value="x2_new"
        #print("x2 new is the best")

###################original thought###################################        
#         x1 = Q_desc[i][:-1]
#         x2 = Q_desc[i+1][:-1]

#         #cross over
#         #generate a random int randc_pos
#         random.seed(0)
#         randc_pos = randint(1,K)

#         x1_new = x1[:randc_pos]+x2[randc_pos:]
#         x2_new = x2[:randc_pos]+x1[randc_pos:]
#########################################################
    #mutation
    #generate a random position randm_pos [1,k] and rand_id [1,n]
    randm_pos = random.randint(1,K)
    rand_id = random.choice(nodeIds)

    #x = x1_new
    #x = Q_desc[i]
    #repair chomorosome if values of two alleles of a chromosome occurs
    if rand_id in (x[:randm_pos-1]+x[randm_pos:]):
        x_new =x
    else:
        x_new = x[:randm_pos-1]+[rand_id]+x[randm_pos:]
        
        #if len(x_new) !=7:
        #    
        #    print("In x_new i =",i,"random pos ",randm_pos," randc pos ",
        #          randc_pos,"x has more ", x,"x takes the value",value,
        #          "x2: ", x2, "Q2:", Q2,
        #         "x1:", x1)

    x_new_fit = algorithm2_part1(x_new, add_trust_ftr)
    #print("This is x_new_fit:", str(x_new_fit),"This is Q_desc[i][-1]",str(Q_desc[i][-1]))
    #print(i,": end")
    if (x_new_fit > Q_desc[i][-1]):
        #print("Fit better: i is ",i," ", x_new + [x_new_fit])
        return x_new + [x_new_fit]
    else:
        #print("Orgin better: i is ",i," ", Q_desc[i])
        return Q_desc[i]

def run_global_optimum_ga(add_trust_ftr = False):
    
    print("Running global optimum Genetic Algorithm for best set of active centers.")

    iteration = 0
    Q_desc_temp = Q_desc.copy()
    Q_desc_new_global=[]
    pre_fit = 0
    while iteration <10:
        if (Q_desc_new_global != []):
            Q_desc_temp = Q_desc_new_global
            pre_fit = Q_desc_temp[0][-1]
            Q_desc_new_global = []
        pool = Pool()
        result_async = [pool.apply_async(parallel_insider_algorithm2_part2, 
                                         args = (i,Q_desc_temp,K,nodeIds,signed_network,node_features_df, add_trust_ftr)) for i in range(N)] 
        Q_desc_new_global = [r.get() for r in result_async] 
        Q_desc_new_global=sorted(Q_desc_new_global,key=lambda x:x[-1],reverse=True)
        check(Q_desc_new_global)
        #print(Q_desc_new_global)
        if Q_desc_new_global[0][-1] == pre_fit:
            iteration +=1
        else:
            iteration=1
        print("Best fitness value is ",str(Q_desc_new_global[0][-1]),", and iteration currently is ", str(iteration))
        print(Q_desc_new_global[0][:K])
    
    return Q_desc_new_global[0][:K]

# Getting globally optimum social circles without considering link/trust feature.
ac_wo_trust = run_global_optimum_ga()
print("Best Set of Active Centers without trust feature:")
print(ac_wo_trust)
print("Generating Social Circles.")
sc_wo_trust = algorithm1(nodeIds, ac_wo_trust)
print("Social Circles without trust:")
print(sc_wo_trust)

# Function to calculate net values of the properties (degree centrality, strenght of ties, profile similarity, objective function value).

def get_net_values(social_circle, active_centers):
  net_deg_cen_C = 0
  net_deg_cen_R = 0
  net_str_C = 0
  net_str_R = 0
  net_prof_sim_C = 0
  net_prof_sim_R = 0
  net_obj_val = 0

  for active_center in social_circle:
    circle = social_circle[active_center]
    
    residual =  residualArea(int(active_center), circle)     
    
    deg_cen_C = degreeCentrality(int(active_center),circle)
    net_deg_cen_C += deg_cen_C
    deg_cen_R = degreeCentrality(int(active_center),residual)
    net_deg_cen_R += deg_cen_R 
    
    prof_sim_C = 0
    for c in circle:
      prof_sim_C +=profSimilarity(c, int(active_center), active_centers)
    prof_sim_C = prof_sim_C/len(circle)
    net_prof_sim_C += prof_sim_C

    prof_sim_R = 0
    for r in residual:
      prof_sim_R += profSimilarity(r, int(active_center), active_centers)
    if len(residual) != 0:
      prof_sim_R = prof_sim_R/len(residual)
    net_prof_sim_R += prof_sim_R
              
    str_C = 0
    for c in circle:
      str_C += strengthOfTies(c, int(active_center))
      str_C += strengthOfTies(int(active_center), c)
    str_C = str_C/len(circle)
    net_str_C += str_C

    str_R = 0
    for r in residual:
      str_R+= strengthOfTies(r, int(active_center))
      str_R+= strengthOfTies(int(active_center), r)
    if len(residual) != 0:
      str_R = str_R/len(residual)
    net_str_R += str_R
    net_obj_val += deg_cen_C - deg_cen_R + prof_sim_C - prof_sim_R + str_C - str_R
  
  net_deg_cen_C /= K
  net_deg_cen_R /= K
  net_str_C /= K
  net_str_R /= K
  net_prof_sim_C /= K
  net_prof_sim_R /= K
  net_obj_val /= K

  return net_deg_cen_C, net_deg_cen_R, net_str_C, net_str_R, net_prof_sim_C, net_prof_sim_R, net_obj_val

# Getting net values for social circles without trust feature.
net_deg_cen_C, net_deg_cen_R, net_str_C, net_str_R, net_prof_sim_C, net_prof_sim_R, net_obj_val = get_net_values(sc_wo_trust, ac_wo_trust)
print("Net Degree Centrality for circle without trust feature: " + str(net_deg_cen_C))
print("Net Degree Centrality for residual without trust feature: " + str(net_deg_cen_R))
print("Net Strength of ties for circle without trust feature: " + str(net_str_C))
print("Net Strength of ties for residual without trust feature: " + str(net_str_R))
print("Net Pofile similarity for circle without trust feature: " + str(net_prof_sim_C))
print("Net Pofile similarity for residual without trust feature: " + str(net_prof_sim_R))
print("Net Objective value without trust feature: " + str(net_obj_val))


# Resetting population values generated reandomly in previous successfull runs
# Part two
##Augmented matrix Q containing fitness value for each set of active center
Q=[]
for i in range(0,len(fitness)):
    Qi = population[i].copy()
    Qi.append(fitness[i])
    Q.append(Qi)

fit_val_index = K
Q_desc= sorted(Q,key=lambda x:x[fit_val_index],reverse=True)

# Getting globally optimum social circles by considering link/trust feature.

ac_w_trust = run_global_optimum_ga(add_trust_ftr=True)
print("Best Set of Active Centers with trust feature:")
print(ac_w_trust)
print("Generating Social Circles.")
sc_w_trust = algorithm1(nodeIds, ac_w_trust)
print("Social Circles with trust:")
print(sc_w_trust)

# Getting net values for social circles with trust feature.
net_deg_cen_C_trust, net_deg_cen_R_trust, net_str_C_trust, net_str_R_trust, net_prof_sim_C_trust, net_prof_sim_R_trust, net_obj_val_trust = get_net_values(sc_w_trust, ac_w_trust)
print("Net Degree Centrality for circle with trust feature: " + str(net_deg_cen_C_trust))
print("Net Degree Centrality for residual with trust feature: " + str(net_deg_cen_R_trust))
print("Net Strength of ties for circle with trust feature: " + str(net_str_C_trust))
print("Net Strength of ties for residual with trust feature: " + str(net_str_R_trust))
print("Net Pofile similarity for circle with trust feature: " + str(net_prof_sim_C_trust))
print("Net Pofile similarity for residual with trust feature: " + str(net_prof_sim_R_trust))
print("Net Objective value with trust feature: " + str(net_obj_val_trust))

# Functions for measuring evaluation measures to assess goodness of clusters.

from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
import numpy as np
from sklearn.cluster import KMeans

def silhouetteCoefficient(X, labels):
  return metrics.silhouette_score(X, labels, metric='euclidean')

def calinskiHarabasz(X, labels):
  return metrics.calinski_harabasz_score(X, labels)

def daviesBouldin(X, labels):
  return metrics.davies_bouldin_score(X, labels)

# Preprocessing social circles data for comparison.

nodes_ftrs_wo_trust = []
nodes_clstr_lbls_wo_trust = []

for ac in sc_wo_trust:
  members = sc_wo_trust[ac]
  #extracting features
  ac_features = node_features_df[node_features_df['nodeId'] == int(ac)][0:].values[0][1:]
  nodes_ftrs_wo_trust.append(ac_features)
  nodes_clstr_lbls_wo_trust.append(int(ac))
  for member in members:
    # extracting feature values for each member in social circle
    node_ftrs = node_features_df[node_features_df['nodeId'] == member][0:].values[0][1:]
    nodes_ftrs_wo_trust.append(node_ftrs)
    nodes_clstr_lbls_wo_trust.append(int(ac))


nodes_ftrs_w_trust = []
nodes_clstr_lbls_w_trust = []

for ac in sc_w_trust:
  members = sc_w_trust[ac]
  #extracting features
  ac_features = list(node_features_df[node_features_df['nodeId'] == int(ac)][0:].values[0][1:])
  nodes_ftrs_w_trust.append(ac_features)
  nodes_clstr_lbls_w_trust.append(int(ac))
  for member in members:
    # extracting feature values for each member in social circle
    node_ftrs = list(node_features_df[node_features_df['nodeId'] == member][0:].values[0][1:])
    nodes_ftrs_w_trust.append(node_ftrs)
    nodes_clstr_lbls_w_trust.append(int(ac))

# Comparing clusters formed with trust feature and without trust feature.

print("Silhouette Coefficient Scores:")
print("With trust feature: " + str(silhouetteCoefficient(nodes_ftrs_w_trust, nodes_clstr_lbls_w_trust)))
print("Without trust feature: " + str(silhouetteCoefficient(nodes_ftrs_wo_trust, nodes_clstr_lbls_wo_trust)))

print("\nCalinski Harabasz Scores:")
print("With trust feature: " + str(calinskiHarabasz(nodes_ftrs_w_trust, nodes_clstr_lbls_w_trust)))
print("Without trust feature: " + str(calinskiHarabasz(nodes_ftrs_wo_trust, nodes_clstr_lbls_wo_trust)))

print("\nDavies Bouldin Scores:")
print("With trust feature: " + str(daviesBouldin(nodes_ftrs_w_trust, nodes_clstr_lbls_w_trust)))
print("Without trust feature: " + str(daviesBouldin(nodes_ftrs_wo_trust, nodes_clstr_lbls_wo_trust)))