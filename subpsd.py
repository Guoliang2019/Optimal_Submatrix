#!/usr/bin/env python
# coding: utf-8

# # Computing Optimal Submatrix of an Invalid Correlation Matrix
# 
# 
# ### Goal: Given invalid correlation matrix C,find the optimal submatrix that has the largest similarity to the corresponding principal submatrix.
# 
# ### Key Mathematical Operations:
# 
# * $C \in R^{n \times n}$ : fake correlation matrix
# * [$J(C)$](https://www.researchgate.net/publication/280050954_The_most_simple_methodology_to_create_a_valid_correlation_matrix_for_risk_management_and_option_pricing_purposes): function computing the optimal valid correlation matrix representing $C$ 
# * $\hat{C} = J(C),\in R^{n×n}$
# * $similarity ∶= 1 - \frac{2}{n(n-1)} \times \sum_{i<j}|\frac{C_{ij}-\hat{C}_{ij}}{C_{ij}}|$
# * $change(i) := \sum_{j}|C_{ij}-\hat{C}_{ij}|$
# * $i^* = max_{_{i=1,2,3,...,n}}change(i)$
# * $j^* = i^*$
# * $C_{sub\setminus i^*,j^*} := \{ C_{ij} \}_{i\neq i^*,j\neq j^*},\in R^{ n-1 \times n-1}$
# * $\hat{C}_{sub\setminus i^*,j^*} := J(C_{sub\setminus i^*,j^*}), \in R^{ n-1 \times n-1}$
# 
# ### Algorithm
# $C_{optimal}=C$ <br />
# $\hat{C}_{optimal}=\hat{C}$ <br />
# $similarity_{optimal}=similarity$ <br />
# $for \quad m = n-1,n-2,...,5,4,3:$ <br />
# $\quad \quad C=C_{sub\setminus i^*,j^*}$ <br />
# $\quad \quad \hat{C}=\hat{C}_{sub\setminus i^*,j^*}$ <br />
# $\quad \quad if \quad similarity > similarity_{optimal}:$ <br />
# $\quad \quad \quad \quad C_{optimal}=C$ <br />
# $\quad \quad \quad \quad \hat{C}_{optimal}=\hat{C}$ <br />
# $\quad \quad \quad \quad similarity_{optimal}=similarity$ <br />
# $\quad \quad \quad \quad if \quad similarity_{optimal} = 1:$ <br />
# $\quad \quad \quad \quad \quad \quad stop$

# ## Python Scripting

# In[1]:


import numpy as np
import numpy.linalg as LA
from statsmodels.stats.moment_helpers import cov2corr


# ## $J(C): find\_valid\_corr$

# In[2]:


def find_valid_corr(C,search_size=1000,phi_size=100000,max_attempt=500,step_unit=1e-17,disp=False):
    '''
    Compute the optimal valid correlation matrix representing matrix C
    
    Parameters:
    -----------
    C: numpy array, symmetric square matrix whose diagonal values equal to 1 with any other element from -1 to 1
    search_size: int, number of valid correlation matrices for C in the search space
    phi_size: int, number of standard normal draws
    max_attempt: int, maximum attempts to construct a positive definite seed matrix by slightly changing eigenvalues 
    step_unit: float, at each attempt, increase non-positive eigenvalues by one step_unit 
    disp: boolean, if true print information on seed matrix, eigenvalues, and upper matrix after Cholesky decomposition
    
    Return:
    -------
    optimal_C_hat: numpy array, optimal valid correlation matrix representing C
    '''
    if C.shape[0] != C.shape[1]:
        raise ValueError('Matrix is not square matrix')
    if np.sqrt(np.sum((C.T - C)**2)) > (C.shape[0]**2)*1e-5:
        raise ValueError('Matrix is not symmetric')
    
    val_,vec = LA.eigh(C)
    if (val_>=-1e-08).all(): # positive is defined as >= -1e-8 due to numerical precision
        print('C is already a PSD matrix')
        return C
    
    chol_done = False
    attempt = 0
    while (not chol_done) and attempt <= max_attempt:
        try:
            #print('{}th attempt'.format(attempt))
            val = val_.copy()
            val[val<0] = attempt*step_unit # set negative eigenvalues into 0
                               # 0 here represented by 1e-16, this is to make psd matrix a little more positive
                               # to allow Cholesky Decomposition to work
            B_star = vec*(np.sqrt(val)) # take square root for every eigen values
            B = B_star/np.linalg.norm(B_star,axis=1,keepdims=True) # normalize row vectors
            seed_C = np.matmul(B,B.T)
            
            U = LA.cholesky(seed_C) # return upper matrix
            chol_done = True
        except:
            if attempt == max_attempt:
                raise RuntimeError('Failed to find Seed Matrix')
            attempt += 1
    if disp:
        print('seed correlation matrix:')
        print(seed_C)
        print('eigen values of seed matrix:')
        print(LA.eigh(seed_C)[0])
        print('chol_done:',chol_done,'at {}th attempt'.format(attempt))
        print('Upper Matrix:')
        print(U)
    
    optimal_C_hat = None
    optimal_error = np.inf 
    for _ in range(search_size):
        phi = np.random.randn(phi_size,C.shape[0])
        R = phi.dot(U)
        C_hat = cov2corr(np.cov(R,rowvar=False))
        error = np.sum((C_hat - C)**2)
        if error < optimal_error and (LA.eigh(C_hat)[0]>=-1e-8).all(): # positive is defined as >= -1e-5 due to computing reason
            optimal_C_hat = C_hat
            optimal_error = error
        else:
            continue
    print('element-wise error:',optimal_error)
    print('distance error:',error_measure(C,optimal_C_hat,'d'))
    print('similarity:',error_measure(C,optimal_C_hat,'s'))
    return optimal_C_hat
    #return optimal_error


# ## $ Similarity: error\_measure$

# In[3]:


def error_measure(C,C_hat,method='d'):
    '''
    Measure the difference/similarity between Matrix C and its simulated counterparty C_hat
    
    Parameters:
    -----------
    C: numpy array, symmetric square matrix whose diagonal values equal to 1 with any other element from -1 to 1
    C_hat: numpy array, symmetric square matrix whose diagonal values equal to 1 with any other element from -1 to 1
    method: str, distance/d to compute the square root of sum of square of element-wise difference
                 similarity/s to compute the 1 - sum of element-wise change of the upper matrix devided by number of elements 
    Return:
    -------
    float, distance or similarity
    '''
    if C.shape != C_hat.shape:
        raise ValueError('Simulated Matrix does not have the same dimension the original Matrix')
    if C.shape[0] != C.shape[1]:
        raise ValueError('Matrices are not square matrix')
    if np.sqrt(np.sum((C.T - C)**2)) > (C.shape[0]**2)*1e-6:
        raise ValueError('Matrix C is not symmetric')
    if np.sqrt(np.sum((C_hat.T - C_hat)**2)) > (C_hat.shape[0]**2)*1e-6:
        raise ValueError('Matrix C_hat is not symmetric')
    
    if method in ('distance','d'):
        return np.sqrt(np.sum((C - C_hat)**2))
    elif method in ('similarity','s'):
        n = C.shape[0]
        return 1-np.sum(np.abs(np.triu((C - C_hat),k=1)/C))/(n*(n-1)/2)
    else:
        raise NameError('method can only be distance(d)/similarity(s)')


# ## $ Algorithm: find\_optimal\_subcorr$

# In[4]:


def find_optimal_subcorr(C,stopping_dim=3):
    '''
    Compute the optimal valid submatrix that has the largest similarity to the corresponding principal submatrix of a given matrix
    
    Parameters:
    -----------
    C: numpy array, symmetric square matrix whose diagonal values equal to 1 with any other element from -1 to 1
    stopping_dim: int, the lowest dimension of interest to search on
    
    Return:
    -------
    opt_C: numpy array, the principal submatrix 
    opt_C_hat: numpy array, optimal valid submatrix that has the largest similarity to the corresponding principal submatrix
    '''
    cur_C = C.copy()
    print('Dimension: '+str(C.shape[0]))
    cur_C_hat = find_valid_corr(cur_C)
    opt_C = cur_C
    opt_C_hat = cur_C_hat
    opt_sim = error_measure(cur_C,cur_C_hat,'s')
    for m in range(C.shape[0]-1,stopping_dim-1,-1):
        print('Dimension: '+str(m))
        change_vec = np.sum(np.abs(cur_C - cur_C_hat),axis=1)
        ind_max = change_vec.argmax()
        remain_ind = [i for i in range(cur_C.shape[0]) if i != ind_max]
        cur_C = cur_C[np.ix_(remain_ind,remain_ind)]
        cur_C_hat = find_valid_corr(cur_C)
        sim = error_measure(cur_C,cur_C_hat,'s')
        if sim > opt_sim:
            opt_C = cur_C
            opt_C_hat = cur_C_hat
            opt_sim = sim
            if opt_sim > 1-1e-10:
                break
    print('optimal similarity {} at dimension {}'.format(opt_sim,opt_C.shape[0]))
    return opt_C, opt_C_hat


# In[ ]:




