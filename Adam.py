

### Initialize
def initialize_he_Adam(layers_dims):

    
    V = {}
    S = {}
    parameters = {}
    L = len(layers_dims) - 1 
     
    for l in range(1, L + 1):
        
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2 / layers_dims[l-1])
        V['dW'+str(l)] = np.zeros_like(parameters["W" + str(l)])
        S['dW'+str(l)] = np.zeros_like(parameters["W" + str(l)])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        V["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        S["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        
    
    return parameters,V,S
    
    
#Parameters Update 
def update_with_Adam(params,grads,types,lr,V,S,beta1,beta2,t):
  L = len(types)
  V_C = {}
  S_C = {}


  for i in range(1,L+1):


    I = str(i)
    V['dW'+I] = beta1*V['dW'+I] + (1-beta1)*grads['dW'+I]
    V['db'+I] = beta1*V['db'+I] + (1-beta1)*grads['db'+I]

    V_C['dW'+I] = V['dW'+I]/(1-np.power(beta1,t))
    V_C['db'+I] = V['db'+I]/(1-np.power(beta1,t))


    S['dW'+I] = beta2*S['dW'+I] + (1-beta2)*np.power(grads['dW'+I],2)
    S['db'+I] = beta2*S['db'+I] + (1-beta2)*np.power(grads['db'+I],2)

    S_C['dW'+I] = S['dW'+I]/(1-np.power(beta2,t))
    S_C['db'+I] = S['db'+I]/(1-np.power(beta2,t))
    
    
    params['W'+I] = params['W'+I] - lr*(V_C['dW'+I]/np.sqrt(S_C['dW'+I]+1e-8)+1e-8)
    params['b'+I] = params['b'+I] - lr*(V_C['db'+I]/np.sqrt(S_C['db'+I]+1e-8)+1e-8)

  return params,V,S
  
  
  
# Training  
def batch_train_Adam(X,Y,P,types,iter,lr,drop_layer,batch_size,V,S,beta1,beta2):
  params = P
  m = X.shape[1]
  permutation = list(np.random.permutation(m))
  shuffled_X = X[:, permutation]
  shuffled_Y = Y[:, permutation].reshape((1,m))
  
  n_batches = m//batch_size
  t = 0




  
  
  for i in range(iter):
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    for k in range(0,n_batches):
      X_batch = X[:,k*batch_size:(k+1)*batch_size]
      Y_batch = Y[:,k*batch_size:(k+1)*batch_size]


      out = forward(X_batch,params,types,drop_layer)
      grads = backward(X_batch,Y_batch,out, params,types)
      t= t+1
      params,V,S = update_with_Adam(params,grads,types,lr,V,S,beta1,beta2,t)
      

    if i%100==0:
      C = compute_cost(out['A'+str(len(types))],Y_batch)
      
      print('iteration :' +str(i))
      
      print(C)
  return params
  
### Usage  
#params,V,S = initialize_he_Adam([2,10,5,1])
#types = ['tanh','tanh','sigmoid']
#new_params = batch_train_Adam(X=train_X,Y=train_Y,P=params,types=types,iter=1000,lr=.008,drop_layer='none',batch_size=64,V=V,S=S,beta1=0.9,beta2=0.999)
  

  
    
    
    
    
