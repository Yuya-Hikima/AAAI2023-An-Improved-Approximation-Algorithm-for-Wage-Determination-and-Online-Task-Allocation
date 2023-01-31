import pandas as pd
import numpy as np
import random
import math
import networkx as nx
import time
import csv
import pickle
import os
import GPyOpt
import pulp
from pulp import value
from numpy import linalg
import sys
import pickle
from pulp import PULP_CBC_CMD
import statistics
import copy
from scipy import stats

#set a seed for random parameters
random.seed(314)

#----------
# Setting
#----------
args=sys.argv
#input parameters
m=int(args[1])
n=int(args[2])
T=int(args[3])
#lower and upper bounds of task budgets
rand_low=int(args[4])
rand_up=int(args[5])
#generate task budgets
b_u=[random.randint(rand_low, rand_up) for i in range(m)]
b_max=int(max(b_u))

#the parameter of the OM-CRA method
max_iteration_OMCRA=int(args[6])
residual_end_condition=10**(-5)

#running time of bayesian optimization
bo_run_time=int(args[7])
#running time of random search
rs_run_time=int(args[8])
#number of simulations
num_simulation=int(args[9])

#given constant parameters
#the value of allowable error in the proposed method
epsilon=0.00001
#the number of simulations in Monte Carlo method
num_Monte_Carlo=10**3
#the number of simulations for calcilationf \beta in Dickerson's matching strategy
beta_calc_num=10000

#the parameters of the sigmoid function p_vt
beta=1.0
gamma=0.1*math.sqrt(3)/math.pi

#the parameters of the r_v (, that is, the probability that node v appears)
r_vt_difference=0.01
base_price_low=0.5
base_price_difference=0.3

#the value of allowable error of z (flow amouts) in the proposed method
z_epsilon=10**(-7)

#the parameter of the PDHG method (in the OM-CRA method).
tau_OMCRA=0.1
sigma_OMCRA=0.1
alpha_OMCRA=0.7
eta_OMCRA=0.7
c_OMCRA=0.7


#Preprocessing of the data
with open('work/Reward_matrix', 'rb') as web:
    Reward_matrix = pickle.load(web)
df = pd.read_csv("work/trec-rf10-data.csv")
df_v=df.values
topicID_set=list(set(df_v[:,0]))
workerID_set=list(set(df_v[:,1]))
num_worker=len(workerID_set)
num_task=len(df_v)

#------------------
# define functions
#------------------
def find_value_from_list(lst, value):
    return [i for i, x in enumerate(lst) if x == value]

def generate_random_num_based_on_given_pd(pd):
    # if pd= [0.1, 0.2, 0.3]^\top, then return {0/1/2/3} with prob {0.1, 0.2, 0.3 ,0.4(=1-0.1-0.2-0.3)}.
    cumulative_dist = np.cumsum(pd).tolist()
    cumulative_dist.append(1.0)
    random_num = np.random.rand()
    cumulative_dist.append(random_num)
    return sorted(cumulative_dist).index(random_num)

def approximated_objective_value_calculate_dickersons_matching_strategy(m,T,r_vt,Pr,z,beta_ut,W,P,gamma_app_ratio):
    #return the approximated objective value by Monte Carlo method for input parameters when dickerson's matching strategy is used.
    total_rewards_list=[]
    average_rewards_list=[]
    match_num_list=[]
    for h in range(num_Monte_Carlo):
        total_rewards=0
        remove_index_u_list=[]
        rewards_list=[]
        remmain_index_list=list(range(m))
        for k in range(T):
            arrive_index=generate_random_num_based_on_given_pd(r_vt[:,k])
            #determin if the arriving worker will accept the wage
            tmp=np.random.rand()
            if tmp < Pr[arrive_index,k]:
                #determine the node u to match by dickerson's matching strategy
                tmp=generate_random_num_based_on_given_pd(list(z[remmain_index_list,arrive_index,k]*gamma_app_ratio/(Pr[arrive_index,k]*r_vt[arrive_index,k]*beta_ut[remmain_index_list,k])))
                if tmp==len(remmain_index_list):
                    continue
                matching_index=remmain_index_list[tmp]
                total_rewards+=W[matching_index,arrive_index]+P[arrive_index,k]
                rewards_list.append(W[matching_index,arrive_index]+P[arrive_index,k])
                remove_index_u_list.append(matching_index)
                remmain_index_list.remove(matching_index)
                if remmain_index_list==[]:
                    break
        total_rewards_list.append(total_rewards)
        if len(remove_index_u_list) !=0:
            average_rewards_list.append(statistics.mean(rewards_list))
        match_num_list.append(len(remove_index_u_list))
    return [sum(total_rewards_list)/len(total_rewards_list),statistics.mean(average_rewards_list),statistics.mean(match_num_list)]

def beta_ut_calculate(T,m,r_vt,Pr,z,gamma_app_ratio):
    #compute the beta required for dickerson's matching strategy
    beta_ut=np.ones([m,T])
    remmain_index_list=[]
    for h in range(beta_calc_num):
        remmain_index_list.append(list(range(m)))
    for t in range(T):
        sum_u=np.zeros(m)
        for h in range(beta_calc_num):
            arrive_index=generate_random_num_based_on_given_pd(r_vt[:,t])
            tmp=np.random.rand()
            if tmp < Pr[arrive_index,t]:
                if remmain_index_list[h]==[]:
                    continue
                tmp=generate_random_num_based_on_given_pd(list(z[remmain_index_list[h],arrive_index,t]*gamma_app_ratio/(Pr[arrive_index,t]*r_vt[arrive_index,t]*beta_ut[remmain_index_list[h],t])))
                if tmp==len(remmain_index_list[h]):
                    continue
                matching_index=remmain_index_list[h][tmp]
                remmain_index_list[h].remove(matching_index)
                sum_u[matching_index]+=1
        if t!=T-1:
            for u in range(m):
                beta_ut[u,t+1]=beta_ut[u,t]-(1.0/beta_calc_num)*sum_u[u]
    return beta_ut

def val_calc(gamma,beta,to_am,Flow_i,delta,r_v,p_flag):
    #Return the difference in cost per unit of flow when delta is added to the current flow amount to_am for given parameters.
    #Return ``inf" if the flow amount cannot be increased by delta due to the capacity constraint.
    if p_flag==1:
        if Flow_i!=0:
            if r_v-(Flow_i+delta)>0:
                val = (gamma*to_am*np.log((Flow_i+delta)/(r_v-(Flow_i+delta)))*(Flow_i+delta)+beta*to_am*(Flow_i+delta)-(gamma*to_am*np.log(Flow_i/(r_v-Flow_i))*Flow_i+beta*to_am*Flow_i))/delta
            else:
                val= np.infty
        else:
            if r_v-(Flow_i+delta)>0:
                val = (gamma*to_am*np.log((Flow_i+delta)/(r_v-(Flow_i+delta)))*(Flow_i+delta)+beta*to_am*(Flow_i+delta))/delta
            else:
                val= np.infty
    else:
        if Flow_i-delta>0:
            if Flow_i!=0:
                if r_v>Flow_i-delta:
                    val = (gamma*to_am*np.log((Flow_i-delta)/(r_v-(Flow_i-delta)))*(Flow_i-delta)+beta*to_am*(Flow_i-delta)-(gamma*to_am*np.log(Flow_i/(r_v-Flow_i))*Flow_i+beta*to_am*Flow_i))/delta
                else:
                    val= np.infty
            else:
                if r_v-(Flow_i-delta)>0:
                    val = (gamma*to_am*np.log((Flow_i-delta)/(r_v-(Flow_i-delta)))*(Flow_i-delta)+beta*to_am*(Flow_i-delta))/delta
                else:
                    val= np.infty
        elif Flow_i-delta==0:
            val = (0-(gamma*to_am*np.log(Flow_i/(r_v-Flow_i))*Flow_i+beta*to_am*Flow_i))/delta
        else:
            val=np.infty
    return val

def get_target_min_index(min_index, distance, unsearched_nodes):
    #Return the index of the node in the unsearched nodes that is closest to one of the searched nodes.
    start = 0
    while True:
        index = distance.index(min_index, start)
        found = index in unsearched_nodes
        if found:
            return index
        else:
            start = index + 1

def approximated_objective_value_calculate_alaei_matching_strategy(m,T,r_vt,Acc_Pr,z,W,P,E,b_u):
    #return the approximated objective value by Monte Carlo method for input parameters when alaei's matching strategy is used.
    total_rewards_list=[]
    average_rewards_list=[]
    match_num_list=[]
    for h in range(num_Monte_Carlo):
        total_rewards=0
        remove_index_u_list=[]
        rewards_list=[]
        remmain_index_list=list(range(m))
        b_u_hat=b_u.copy()
        for k in range(T):
            arrive_index=generate_random_num_based_on_given_pd(r_vt[:,k])
            #determin if the arriving worker will accept the wage
            tmp=np.random.rand()
            if tmp < Acc_Pr[arrive_index]:
                #determine the node u to match by alaei's matching strategy.
                tmp=generate_random_num_based_on_given_pd(list(z[remmain_index_list,arrive_index]/(Acc_Pr[arrive_index]*r_vt[arrive_index,k])))
                if tmp==len(remmain_index_list):
                    continue
                matching_index=remmain_index_list[tmp]
                #print('choose node')
                if W[matching_index,arrive_index]+P[arrive_index]+E[matching_index,int(b_u_hat[matching_index])-1,k+1]>E[matching_index,int(b_u_hat[matching_index]),k+1]:
                    total_rewards+=W[matching_index,arrive_index]+P[arrive_index]
                    rewards_list.append(W[matching_index,arrive_index]+P[arrive_index])
                    b_u_hat[matching_index]=b_u_hat[matching_index]-1
                    if b_u_hat[matching_index]==0:
                        remove_index_u_list.append(matching_index)
                        remmain_index_list.remove(matching_index)
                        if remmain_index_list==[]:
                            break
                #else:
                    #print('unmatched')
            #else:
                #print('denied')
        total_rewards_list.append(total_rewards)
        if len(rewards_list) !=0:
            average_rewards_list.append(statistics.mean(rewards_list))
        match_num_list.append(len(remove_index_u_list))
    return [sum(total_rewards_list)/len(total_rewards_list),statistics.mean(average_rewards_list),statistics.mean(match_num_list)]

def approximated_objective_value_calculate_alaei_matching_strategy_tdef(m,T,r_vt,Acc_Pr,z,W,P,E,b_u):
    #return the approximated objective value by Monte Carlo method for input parameters when alaei's matching strategy is used.
    total_rewards_list=[]
    average_rewards_list=[]
    match_num_list=[]
    for h in range(num_Monte_Carlo):
        total_rewards=0
        remove_index_u_list=[]
        rewards_list=[]
        remmain_index_list=list(range(m))
        b_u_hat=b_u.copy()
        reward_flag=0
        for k in range(T):
            arrive_index=generate_random_num_based_on_given_pd(r_vt[:,k])
            #determine if the arriving worker will accept the wage
            tmp=np.random.rand()
            if tmp < Acc_Pr[arrive_index,k]:
                #determine the node u to match by alaei's matching strategy.
                tmp=generate_random_num_based_on_given_pd(list(z[remmain_index_list,arrive_index,k]/(Acc_Pr[arrive_index,k]*r_vt[arrive_index,k])))
                if tmp==len(remmain_index_list):
                    continue
                matching_index=remmain_index_list[tmp]
                #print('choose node')
                if W[matching_index,arrive_index]+P[arrive_index,k]+E[matching_index,int(b_u_hat[matching_index])-1,k+1]>E[matching_index,int(b_u_hat[matching_index]),k+1]:
                    total_rewards+=W[matching_index,arrive_index]+P[arrive_index,k]
                    reward_flag=1
                    rewards_list.append(W[matching_index,arrive_index]+P[arrive_index,k])
                    b_u_hat[matching_index]=b_u_hat[matching_index]-1
                    if b_u_hat[matching_index]==0:
                        remove_index_u_list.append(matching_index)
                        remmain_index_list.remove(matching_index)
                        if remmain_index_list==[]:
                            break
                #else:
                    #print('unmatched')
            #else:
                #print('denied')
        total_rewards_list.append(total_rewards)
        if reward_flag!=0:
            average_rewards_list.append(statistics.mean(rewards_list))
        else:
            average_rewards_list.append(0)
        match_num_list.append(len(remove_index_u_list))
    return [sum(total_rewards_list)/len(total_rewards_list),statistics.mean(average_rewards_list),statistics.mean(match_num_list)]


#--------------
# experiments
#--------------
# Lists to store results
proposed_value_list=[]
proposed_time_list=[]
OMCRA_value_list=[]
OMCRA_time_list=[]
rs_approx_value_list=[]
rs_approx_time_list=[]
bo_value_list=[]
bo_time_list=[]

# repeat for # of ``num_simulation''
for setting_tmp in range(num_simulation):
    #Generate the problem
    Task_list=random.sample(range(num_task), m)
    Worker_list=random.sample(range(num_worker), n)
    W=np.zeros([m,n])
    tmp_y=0
    for i in Worker_list:
        tmp_x=0
        for j in Task_list:
            topic_k=find_value_from_list(topicID_set,df_v[j,0])
            W[tmp_x,tmp_y]=Reward_matrix[topic_k[0],i]
            tmp_x+=1
        tmp_y+=1

    #generate the r_vt (, that is, the probability that node v appears at time t)
    r_v_rand=r_vt_difference+np.random.rand(n).reshape([n,1])*(1-r_vt_difference)
    sums_along_cols = r_v_rand.sum(axis=0)
    r_v=r_v_rand/sums_along_cols
    r_vt=np.ones([1,T])*r_v

    epsilon_proposed =np.min(r_v)*0.01

    #generate the parameter of p_vt (p_vt(x) is the probability that node v accepts the wage x at time t)
    base_price=base_price_low+base_price_difference*np.random.rand(n)

    #----------------
    #proposed method
    #----------------
    start_time=time.time()
    #generate the graph of min-cost flow problem
    G = nx.DiGraph()
    #add nodes
    #u
    G.add_nodes_from(range(m))
    #v
    G.add_nodes_from(range(m,m+n))
    #s
    G.add_node(m+n)
    #d
    G.add_node(m+n+1)

    #Set quantities that do not match flow constraints (, that is, equations (2)--(4) in our paper)
    excess=np.zeros(m+n+2)
    #excess[n+m]=0
    #excess[n+m+1]=0

    #set the amount of delta (amount to adjust current flow)
    delta=n+0.5
    #set the flow amount (primal variable) for each edge and the potential for each node (dual variable)
    Flow=np.zeros([m+n+2,m+n+2])
    potential=np.zeros(m+n+2)

    #Matrix representing the cost of increasing the flow of each edge by delta
    Cost_matrix=np.ones([m+n+2,m+n+2])*np.inf

    #Matrix representing the remaining capacity of each edge
    Cap_matrix=np.zeros([m+n+2,m+n+2])

    for i in range(m):
        for j in range(n):
            Cap_matrix[i, m+j]=1
            Cap_matrix[m+j,i]=0
            G.add_edge(i, m+j)
            G.add_edge(m+j,i)
            Cost_matrix[i,m+j]=-W[i,j]
            Cost_matrix[m+j,i]=W[i,j]

    for i in range(m):
        Cap_matrix[m+n,i]=b_u[i]/T
        Cap_matrix[i,m+n]=0
        Cost_matrix[m+n,i]=0
        Cost_matrix[i,m+n]=0
        G.add_edge(m+n,i)
        G.add_edge(i,m+n)

    val=np.infty
    for j in range(n):
        Cap_matrix[m+j,m+n+1]=1-epsilon_proposed
        Cap_matrix[m+n+1,m+j]=epsilon_proposed
        Cost_matrix[m+j,m+n+1]=val
        Cost_matrix[m+n+1,m+j]=val
        excess[m+j]-=epsilon_proposed
        excess[m+n+1]+=epsilon_proposed
        Flow[m+j,m+n+1]+=epsilon_proposed
        Flow[m+n+1,m+j]-=epsilon_proposed
        G.add_edge(m+j,m+n+1)
        G.add_edge(m+n+1,m+j)

    G.add_edge(m+n,m+n+1)
    Cap_matrix[m+n,m+n+1]=0
    Cost_matrix[m+n,m+n+1]=0

    G.add_edge(m+n+1,m+n)
    Cap_matrix[m+n+1,m+n]=n
    Cost_matrix[m+n+1,m+n]=0

    #Capacity scaling method
    while delta>0.0000001:
        #delta-scaling phase
        for i in range(m):
            for j in range (n):
                if Cost_matrix[i,m+j]-potential[i]+potential[m+j] < -epsilon and Cap_matrix[i,m+j]>=delta:
                    Flow[i,m+j]+=delta
                    Flow[m+j,i]-=delta
                    excess[i] -= delta
                    excess[m+j] += delta
                    Cap_matrix[i,m+j]-=delta
                    Cap_matrix[m+j,i]+=delta

                if Cost_matrix[m+j,i]-potential[m+j]+potential[i] < -epsilon and Cap_matrix[m+j,i]>=delta:
                    Flow[i,m+j]-=delta
                    Flow[m+j,i]+=delta
                    excess[i] += delta
                    excess[m+j] -= delta
                    Cap_matrix[i,m+j]+=delta
                    Cap_matrix[m+j,i]-=delta

        for i in range(m):
            if Cost_matrix[m+n,i]-potential[m+n]+potential[i] < -epsilon and Cap_matrix[m+n,i]>=delta:
                Flow[m+n,i]+=delta
                Flow[i,m+n]-=delta
                excess[m+n] -= delta
                excess[i] += delta
                Cap_matrix[m+n,i]-=delta
                Cap_matrix[i,m+n]+=delta

            if Cost_matrix[i,m+n]-potential[i]+potential[m+n] < -epsilon and Cap_matrix[i,m+n]>=delta:
                Flow[m+n,i]-=delta
                Flow[i,m+n]+=delta
                excess[m+n] += delta
                excess[i] -= delta
                Cap_matrix[m+n,i]+=delta
                Cap_matrix[i,m+n]-=delta

        for j in range (n):
            if Cost_matrix[m+j,m+n+1]-potential[m+j]+potential[m+n+1] < -epsilon and Cap_matrix[m+j,m+n+1]>=delta:
                Flow[m+j,m+n+1]+=delta
                Flow[m+n+1,m+j]-=delta
                excess[m+j] -= delta
                excess[m+n+1] += delta
                Cap_matrix[m+j,m+n+1]-=delta
                Cap_matrix[m+n+1,m+j]+=delta
                Cost_matrix[m+j,m+n+1]=val_calc(gamma,beta,base_price[j],Flow[m+j,m+n+1],delta,r_v[j],1)
                Cost_matrix[m+n+1,m+j]=val_calc(gamma,beta,base_price[j],Flow[m+j,m+n+1],delta,r_v[j],-1)

            if Cost_matrix[m+n+1,m+j]-potential[m+n+1]+potential[m+j] < -epsilon and Cap_matrix[m+n+1,m+j]>=delta:
                Flow[m+j,m+n+1]-=delta
                Flow[m+n+1,m+j]+=delta
                excess[m+n+1] -= delta
                excess[m+j] += delta
                Cap_matrix[m+j,m+n+1]+=delta
                Cap_matrix[m+n+1,m+j]-=delta
                Cost_matrix[m+n+1,m+j]=val_calc(gamma,beta,base_price[j],Flow[m+n+1,m+j],delta,r_v[j],1)
                Cost_matrix[m+j,m+n+1]=val_calc(gamma,beta,base_price[j],Flow[m+n+1,m+j],delta,r_v[j],-1)

        if Cost_matrix[m+n,m+n+1]-potential[m+n]+potential[m+n+1] < -epsilon and Cap_matrix[m+n,m+n+1]>=delta:
            Flow[m+n,m+n+1]+=delta
            Flow[m+n+1,m+n]-=delta
            excess[m+n] -= delta
            excess[m+n+1] += delta
            Cap_matrix[m+n,m+n+1]-=delta
            Cap_matrix[m+n+1,m+n]+=delta

        if Cost_matrix[m+n+1,m+n]-potential[m+n+1]+potential[m+n] < -epsilon and Cap_matrix[m+n+1,m+n]>=delta:
            Flow[m+n,m+n+1]-=delta
            Flow[m+n+1,m+n]+=delta
            excess[m+n+1] -= delta
            excess[m+n] += delta
            Cap_matrix[m+n,m+n+1]+=delta
            Cap_matrix[m+n+1,m+n]-=delta


        #shortest path phase
        #Dijkstras's algorithm
        while len(list(*np.where(excess >= delta)))>0 and len(list(*np.where(excess <= -delta)))>0:
            start_node=list(*np.where(excess >= delta))[0]
            node_num = n+m+2
            unsearched_nodes = list(range(node_num))
            distance = [math.inf] * node_num
            previous_nodes = [math.inf] * node_num
            distance[start_node] = 0
            searched_nodes=[]

            while(len(unsearched_nodes) != 0):
                posible_min_distance = math.inf
                for node_index in unsearched_nodes:
                    if posible_min_distance > distance[node_index]:
                        posible_min_distance = distance[node_index]
                target_min_index = get_target_min_index(posible_min_distance, distance, unsearched_nodes)
                if distance[target_min_index]==np.inf:
                    error #debug
                unsearched_nodes.remove(target_min_index)
                searched_nodes.append(target_min_index)
                if excess[target_min_index] <= -delta:
                    end_node=target_min_index
                    break
                neighbor_node_list = list(G.succ[target_min_index])
                deb_tmp=0
                for neighbor_node in neighbor_node_list:
                    if neighbor_node in unsearched_nodes:
                        if distance[neighbor_node] - epsilon > distance[target_min_index] + Cost_matrix[target_min_index,neighbor_node] -potential[target_min_index]+potential[neighbor_node] and Cap_matrix[target_min_index,neighbor_node] >= delta:
                            distance[neighbor_node] = distance[target_min_index] + Cost_matrix[target_min_index,neighbor_node]-potential[target_min_index]+potential[neighbor_node] # 過去に設定されたdistanceよりも小さい場合はdistanceを更新
                            previous_nodes[neighbor_node] =  target_min_index
                            deb_tmp+=1
            #update potential
            for i in range(n+m+2):
                if i in searched_nodes:
                    potential[i] -= distance[i]
                else:
                    potential[i] -= distance[end_node]

            #update flow
            tmp_node=end_node
            x=0
            tmp_kk=0
            while tmp_node!=start_node:
                Flow[previous_nodes[tmp_node],tmp_node]+=delta
                Flow[tmp_node,previous_nodes[tmp_node]]-=delta
                Cap_matrix[previous_nodes[tmp_node],tmp_node] -= delta
                Cap_matrix[tmp_node,previous_nodes[tmp_node]] += delta
                tmp_node=previous_nodes[tmp_node]
                tmp_kk+=1
                if tmp_kk>1000:
                    error #debug

            #update Cost_matrix (, which represents the cost of increasing the flow of each edge by delta)
            for j in range(n):
                Cost_matrix[m+j,n+m+1]=val_calc(gamma,beta,base_price[j],Flow[m+j,m+n+1],delta,r_v[j],1)
                Cost_matrix[m+n+1,m+j]=val_calc(gamma,beta,base_price[j],Flow[m+j,m+n+1],delta,r_v[j],-1)

            #update excess, which represents quantities that do not match flow constraints
            excess[start_node]-=delta
            excess[end_node]+=delta

        #update delta
        delta=0.5*delta
        #update Cost_matrix
        for j in range(n):
            Cost_matrix[m+j,m+n+1]=val_calc(gamma,beta,base_price[j],Flow[m+j,m+n+1],delta,r_v[j],1)
            Cost_matrix[m+n+1,m+j]=val_calc(gamma,beta,base_price[j],Flow[m+j,m+n+1],delta,r_v[j],-1)

    #Calculate the price corresponding to the flow
    price_proposed=np.zeros(n)
    for j in range(n):
        price_proposed[j]=-gamma*base_price[j]*np.log(Flow[m+j,m+n+1]/(r_v[j]-Flow[m+j,m+n+1]))-beta*base_price[j]

    #z_proposed and z_vd_proposed are needed to compute E later
    z_proposed=np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            z_proposed[i,j]=Flow[i,m+j]

    z_vd_proposed=np.zeros(n)
    for j in range(n):
        z_vd_proposed[j]=sum(z_proposed[:,j])

    #calculate E, which is the expected total rewards we get from task u with remaining capacity b_u at time t
    #Alaei's matching strategy uses this E.
    E=np.zeros([m,b_max+1,T+1])
    for i in range(m):
        for t in range(T):
            current_t=T-1-t
            for r in range(int(b_u[i])+1):
                tmp=0
                if r==0:
                    E[i,r,current_t]=0
                else:
                    for j in range(n):
                        tmp+=z_proposed[i,j]*max(W[i,j]+price_proposed[j]+E[i,r-1,current_t+1],E[i,r,current_t+1])
                    E[i,r,current_t]=tmp+(1-z_vd_proposed[j])*E[i,r,current_t+1]

    #CPU time
    proposed_time=time.time()-start_time

    # Calculate the result
    Acc_Pr=(1-(1/(1+np.exp(-(price_proposed.reshape(-1,1)+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1))))))
    [proposed_value,proposed_time,zz]=approximated_objective_value_calculate_alaei_matching_strategy(m,T,r_vt,Acc_Pr,z_proposed,W,price_proposed,E,b_u)

    proposed_value_list.append(proposed_value)
    proposed_time_list.append(proposed_time)

    #-------------
    #OMCRA method
    #-------------
    #preprocessing
    W_dupli=np.zeros([int(sum(b_u)),n])
    array_num=0
    for i in range(m):
        k=0
        while k < b_u[i]:
            W_dupli[array_num,:]=W[i,:]
            array_num+=1
            k+=1
    m_dupli=int(sum(b_u))

    # Starts OM-CRA method
    start_time=time.time()
    #Initial points of the primal variables and dual variables
    z_k=np.zeros([m_dupli,n,T])
    la_k=np.zeros([m_dupli])
    z_u_k=np.sum(z_k,axis=(1,2))
    #the iteration number
    k=1
    #the primal residual
    p=1
    #the dual residual
    d=1
    while k<max_iteration_OMCRA and (np.linalg.norm(p)> residual_end_condition or np.linalg.norm(d)>residual_end_condition):
        #Update primal variables (line 4 and 5 in Algorithm 1 of the paper)
        #z_star means z_{k+1}.
        z_star=np.zeros([m_dupli,n,T])
        #Solve Eq (7) for all v, t
        for v in range(n):
            for t in range(T):
                a=z_k[:,v,t]+tau_OMCRA*(W_dupli[:,v]-la_k)
                s=r_vt[v,t]/2.0
                delta=s/2.0
                while delta>r_vt[v,t]*10**(-8):
                    f_vt_dash_s=beta*base_price[v]+gamma*base_price[v]*np.log(s/(r_vt[v,t]-s)) +gamma*base_price[v]*r_vt[v,t]/(r_vt[v,t]-s)
                    left=a-tau_OMCRA*f_vt_dash_s
                    left_plus=left*(left>0)
                    if sum(left_plus)>s:
                           s+=delta
                    else:
                           s-=delta
                    delta=delta/2
                tmp=a-tau_OMCRA*f_vt_dash_s
                z_star[:,v,t]=tmp*(tmp>0)
        #Update dual variables (line 6 in Algorithm 1 of the paper)
        z_u=np.sum(z_star,axis=(1,2))
        la_tmp=la_k+sigma_OMCRA*(2*z_u-z_u_k-1)
        la_star=la_tmp*(la_tmp>0)

        #Update step sizes (line 7--14 in Algorithm 1 of the paper)
        z_d=z_star-z_k
        la_d=la_star-la_k
        z_u_d=np.sum(z_d,axis=(1,2))
        #line 7 in Algorithm 1 of the paper
        if (c_OMCRA/(2.0*tau_OMCRA))*np.sum(np.power(z_d,2))+(c_OMCRA/(2.0*sigma_OMCRA))*np.sum(np.power(la_d,2)) <= 2.0*np.inner(la_d,z_u_d):
            tau_OMCRA=0.5*tau_OMCRA
            sigma_OMCRA=0.5*sigma_OMCRA
        #line 9 and 10 in Algorithm 1 of the paper
        p_sum=0
        for t in range(T):
            for v in range(n):
                p_tmp=-z_d[:,v,t]/tau_OMCRA+la_d
                p_sum+=sum(np.power(p_tmp,2))
        p=np.sqrt(p_sum)
        d=-la_d/sigma_OMCRA+z_u_d
        #line 11--14 in Algorithm 1 of the paper
        if 2*p< np.linalg.norm(d):
            tau_OMCRA=tau_OMCRA*(1-alpha_OMCRA)
            sigma_OMCRA=sigma_OMCRA/(1-alpha_OMCRA)
            alpha_OMCRA=alpha_OMCRA*eta_OMCRA
        elif p > 2*np.linalg.norm(d):
            tau_OMCRA=tau_OMCRA/(1-alpha_OMCRA)
            sigma_OMCRA=sigma_OMCRA*(1-alpha_OMCRA)
            alpha_OMCRA=alpha_OMCRA*eta_OMCRA
        z_u_k=z_u
        z_k=z_star.copy()
        la_k=la_star.copy()
        k=k+1
        print(k,np.linalg.norm(p),np.linalg.norm(d))

    # Calculate x_vt (by Proposition 4 of the paper)
    z_star=z_star*(z_star>0)
    z_vt_sum_star=np.sum(z_star,axis=0)
    X_vt_OMCRA=np.zeros([n,T])
    for v in range(n):
        for t in range(T):
            if z_vt_sum_star[v,t]>z_epsilon:
                X_vt_OMCRA[v,t]=-beta*base_price[v]-gamma*base_price[v]*np.log(z_vt_sum_star[v,t]/(r_vt[v,t]-z_vt_sum_star[v,t]))
            else:
                X_vt_OMCRA[v,t]=0
    #calculate p_vt(x_vt)
    p_vt_OMCRA=(1-(1/(1+np.exp(-(X_vt_OMCRA+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1))))))

    # To obtain the 1/2 approximation matching strategy [Dickerson et al., 2018], calculate beta_ut.
    gamma_app_ratio=0.5
    beta_ut=beta_ut_calculate(T,m_dupli,r_vt,p_vt_OMCRA,z_star,gamma_app_ratio)

    # Calculate results
    OMCRA_time=time.time()-start_time
    [OMCRA_value,add_value,remove_average_num]=approximated_objective_value_calculate_dickersons_matching_strategy(m_dupli,T,r_vt,p_vt_OMCRA,z_star,beta_ut,W_dupli,X_vt_OMCRA,gamma_app_ratio)

    OMCRA_value_list.append(OMCRA_value)
    OMCRA_time_list.append(OMCRA_time)

    #------------
    #BO-A method
    #------------
    #We use GPyOpt.methods.BayesianOptimization to search for x_vt.
    ##Settting of set to search x_vt
    bounds=[]
    for k in range(n*T):
        dic ={}
        key = "name"
        dic[key] = f"x{k}"
        key = "type"
        dic[key] = "continuous"
        key = "domain"
        dic[key] = (-1.0,-0.0)
        bounds.append(dic)

    ## Settting of the oracle.
    max_value=0
    def f_approx(x):
        global max_value
        global opt_vec_Lp
        P_bo_or=x[0]
        P_bo=np.zeros([n,T])
        for v in range(n):
            for t in range(T):
                P_bo[v,t]=P_bo_or[v+t*n]
        Pr_bo=np.zeros([n,T])
        for v in range(n):
            for t in range(T):
                Pr_bo[v,t]=1-(1/(1+np.exp(-(P_bo[v,t]+beta*base_price[v])/(gamma*base_price[v]))))
        # set problem
        problem = pulp.LpProblem('tsp_mip', pulp.LpMaximize)
        z = {(u,v,t):pulp.LpVariable(name='z_{}_{}_{}'.format(u,v,t), lowBound=0, upBound=1, cat='Continuous') for u in range(m) for v in range(n) for t in range (T)}
        problem += pulp.lpSum([(W[u,v] + P_bo[v,t])* z[u,v,t] for u in range(m) for v in range(n) for t in range(T)])
        for v in range(n):
            for t in range(T):
                problem += pulp.lpSum([z[u, v ,t] for u in range(m)]) <= Pr_bo[v,t]*r_vt[v,t]
        for u in range(m):
            problem += pulp.lpSum([z[u,v,t] for v in range(n) for t in range(T)]) <= b_u[u]

        status = problem.solve(PULP_CBC_CMD(msg=0))
        objective_value=pulp.value(problem.objective)
        if objective_value>max_value:
            max_value=objective_value
            opt_vec_Lp=np.ones([m,n,T])
            tmp=0
            for u in range(m):
                for v in range(n):
                    for t in range(T):
                        opt_vec_Lp[u,v,t]=pulp.value(z[u,v,t])
        return -pulp.value(problem.objective)

    ##Search for x_vt
    start_time=time.time()
    myBopt = GPyOpt.methods.BayesianOptimization(f=f_approx, domain=bounds)
    myBopt.run_optimization(max_iter=10**8,max_time=bo_run_time)
    P_bo_or=myBopt.x_opt
    P_bo=np.zeros([n,T])
    for v in range(n):
        for t in range(T):
            P_bo[v,t]=P_bo_or[v+t*n]
    Pr_bo=(1-(1/(1+np.exp(-(P_bo+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1))))))

    opt_vec_usum=np.zeros([n,T])
    for j in range(n):
        for t in range(T):
            opt_vec_usum[j,t]=sum(opt_vec_Lp[:,j,t])

    #calculate E, which is the expected total rewards we get from task u with remaining capacity b_u at time t
    #Alaei's matching strategy uses this E.
    E=np.zeros([m,b_max+1,T+1])
    for i in range(m):
        for t in range(T):
            current_t=T-1-t
            for r in range(int(b_u[i])+1):
                tmp=0
                if r==0:
                    E[i,r,current_t]=0
                else:
                    for j in range(n):
                        tmp+=opt_vec_Lp[i,j,current_t]*max(W[i,j]+P_bo[j,current_t]+E[i,r-1,current_t+1],E[i,r,current_t+1])
                    E[i,r,current_t]=tmp+(1-opt_vec_usum[j,current_t])*E[i,r,current_t+1]

    # Calculate results
    bo_time=time.time()-start_time
    [bo_value,add_value,remove_average_num]=approximated_objective_value_calculate_alaei_matching_strategy_tdef(m,T,r_vt,Pr_bo,opt_vec_Lp,W,P_bo,E,b_u)

    bo_value_list.append(bo_value)
    bo_time_list.append(bo_time)

    #------------
    #RS-A method
    #------------
    start_time=time.time()
    max_value=0
    opt_vec_Lp=np.ones([m,n,T])
    while True:
        P_rand_approx=-np.random.rand(n,T)
        Pr_rand_approx=1-(1/(1+np.exp(-(P_rand_approx+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1)))))
        #Calculate the objective value of the search point
        problem = pulp.LpProblem('tsp_mip', pulp.LpMaximize)
        z = {(u,v,t):pulp.LpVariable(name='z_{}_{}_{}'.format(u,v,t), lowBound=0, upBound=1, cat='Continuous') for u in range(m) for v in range(n) for t in range (T)}
        problem += pulp.lpSum([(W[u,v] + P_rand_approx[v,t])* z[u,v,t] for u in range(m) for v in range(n) for t in range(T)])
        for v in range(n):
            for t in range(T):
                problem += pulp.lpSum([z[u, v ,t] for u in range(m)]) <= Pr_rand_approx[v,t]*r_vt[v,t]
        for u in range(m):
            problem += pulp.lpSum([z[u,v,t] for v in range(n) for t in range(T)]) <= b_u[u]

        status = problem.solve(PULP_CBC_CMD(msg=0))

        #If a larger objective value is obtained, update the solution.
        if pulp.value(problem.objective)> max_value:
            max_value=pulp.value(problem.objective)
            tmp=0
            for u in range(m):
                for v in range(n):
                    for t in range(T):
                        opt_vec_Lp[u,v,t]=pulp.value(z[u,v,t])
            Max_P_rand_approx=P_rand_approx
            Max_Pr_rand_approx=Pr_rand_approx
        #Exit when run time is over.
        if time.time()-start_time>rs_run_time:
            break

    opt_vec_usum=np.zeros([n,T])
    for j in range(n):
        for t in range(T):
            opt_vec_usum[j,t]=sum(opt_vec_Lp[:,j,t])

    #calculate E, which is the expected total rewards we get from task u with remaining capacity b_u at time t
    #Alaei's matching strategy uses this E.
    E=np.zeros([m,b_max+1,T+1])
    for i in range(m):
        for t in range(T):
            current_t=T-1-t
            for r in range(int(b_u[i])+1):
                tmp=0
                if r==0:
                    E[i,r,current_t]=0
                else:
                    for j in range(n):
                        tmp+=opt_vec_Lp[i,j,current_t]*max(W[i,j]+P_bo[j,current_t]+E[i,r-1,current_t+1],E[i,r,current_t+1])
                    E[i,r,current_t]=tmp+(1-opt_vec_usum[j,current_t])*E[i,r,current_t+1]

    # Calculate results
    rs_approx_time=time.time()-start_time
    [rs_approx_value,add_value,remove_average_num]=approximated_objective_value_calculate_alaei_matching_strategy_tdef(m,T,r_vt,Max_Pr_rand_approx,opt_vec_Lp,W,Max_P_rand_approx,E,b_u)

    rs_approx_value_list.append(rs_approx_value)
    rs_approx_time_list.append(rs_approx_time)

OMCRA_p_value=stats.ttest_rel(proposed_value_list, OMCRA_value_list)[1]
OMCRA_p_time=stats.ttest_rel(proposed_time_list, OMCRA_time_list)[1]
bo_p_value=stats.ttest_rel(proposed_value_list,bo_value_list)[1]
bo_p_time=stats.ttest_rel(proposed_time_list,bo_time_list)[1]
rs_p_value=stats.ttest_rel(proposed_value_list,rs_approx_value_list)[1]
rs_p_time=stats.ttest_rel(proposed_time_list,rs_approx_time_list)[1]

#Outputs results
with open('results/result_m=%d_n=%d_T=%d_bu=%dto%d_BOruntime=%d_RSruntime=%d_simulations=%d.csv' %(m,n,T,rand_low,rand_up,bo_run_time,rs_run_time,num_simulation), mode='w') as f_tmp:
    writer = csv.writer(f_tmp)
    writer.writerow(['method','objective value','computation time (seconds)','p_value','p_time','std_value','std_time'])
    writer.writerow(['Proposed',statistics.mean(proposed_value_list),statistics.mean(proposed_time_list),0,0,statistics.stdev(proposed_value_list),statistics.stdev(proposed_time_list)])
    writer.writerow(['OMCRA',statistics.mean(OMCRA_value_list),statistics.mean(OMCRA_time_list),OMCRA_p_value,OMCRA_p_time,statistics.stdev(OMCRA_value_list),statistics.stdev(OMCRA_time_list)])
    writer.writerow(['BO-A',statistics.mean(bo_value_list),statistics.mean(bo_time_list),bo_p_value,bo_p_time,statistics.stdev(bo_value_list),statistics.stdev(bo_time_list)])
    writer.writerow(['RS-A',statistics.mean(rs_approx_value_list),statistics.mean(rs_approx_time_list),rs_p_value,rs_p_time,statistics.stdev(rs_approx_value_list),statistics.stdev(rs_approx_time_list)])
