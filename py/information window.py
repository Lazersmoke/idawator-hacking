import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.io import wavfile
from scipy.signal import hilbert







# The data set is the MusicVAE prior
# The model is:
# Have a big chunk of just data, parameters that the model needs to learn.
# That's part of the input, and then the music is the rest of the input
# (dataBank, lambdas, input) => (output)
# For lambda=0, it should be the identity (in a non definitional way)
# For lambda=eps, it should be a small change of some kind
# D^{ij}_k \lambda_i \text{input}_j

# |b_i> (1 + \lambda_i) <b_i|
def applyLambdasInBasis(lambdas,basis,x):
  afterLambdas = np.einsum("ik,ij,j",np.diag(lambdas + 1),basis,x)
  return np.einsum("ij,i",basis,afterLambdas)

def relu(x):
  return np.maximum(0,x)


testVec = np.random.rand(10)
#print(testVec)
#print(applyLambdasInBasis(np.zeros(3),np.random.rand(3,10),testVec))

def applyLambdasInNonLinBasis(lambdas,egress,nonLinIngress,x):
  return np.einsum("ij,jk,k",egress,np.diag(lambdas + 1),applyNonlinear(nonLinIngress,x))

# Overall Pipeline is:
# inputs -> faithful vector representation -> non linear map to encoded -> lambdas -> egress

# Want to learn the function [faithful rep ==> (important \oplus trash)]


def reconstructionLoss(enc,dec,x):
  important = runEncoder(enc,x)
  recons = runDecoder(dec,important)
  return np.linalg.norm(x - recons)

zSize = 5
netDepth = 2

def deserializeModel(params,xSize,depth):
  funcSize = xSize * depth * (xSize + 1)
  enc = deserializeNonLin(params[:funcSize],xSize,depth)
  dec = deserializeNonLin(params[funcSize:2 * funcSize],xSize,depth)
  return (enc,dec)

def identityNonLin(xSize,depth):
  i = np.insert(np.eye(xSize),0,np.zeros(xSize),axis=1)
  return np.reshape(np.tile(i,(depth,1,1)),(depth,xSize,xSize + 1))

def serializeNonLin(nonLin):
  return nonLin.flatten()

def deserializeNonLin(serializedNonLin,xSize,depth):
  return np.reshape(serializedNonLin,(depth,xSize,xSize+1))

def serialL(params,trainingSamps,xSize,depth):
  (enc,dec) = deserializeModel(params,xSize,depth)
  totalLoss = 0
  for x in trainingSamps:
    important = runEncoder(enc,x)
    recons = runDecoder(dec,important)
    totalLoss += 8000 * np.linalg.norm(x - recons)
  return np.log(totalLoss)

def minimizeCallback(params,trainingSamps,xSize,depth):
  (enc,dec) = deserializeModel(params,xSize,depth)
  totalLoss = 0
  for x in trainingSamps:
    important = runEncoder(enc,x)
    recons = runDecoder(dec,important)
    #if(i == 0):
      #print("Input defect:",x - trainingBias)
      #print("Important:",important)
      #print("Reconstruction defect:",recons - trainingBias)
    totalLoss += np.linalg.norm(x - recons)
  print("Params:",params)
  print("Loss:",np.log(totalLoss))
  print()

def c2(f,*args):
  return (lambda p: f(p,args))

squeezeSize = 5
def runEncoder(enc,x):
  importantOplusTrash = applyNonlinear(enc,x)
  return importantOplusTrash[:squeezeSize]

def runDecoder(dec,important):
  importantOplusZero = np.pad(important,(0,dec.shape[1]-important.shape[0]))
  return applyNonlinear(dec,importantOplusZero)



# Use a three index object as a non-linear map:
# A^i_j (k) ~~~ A^i_j (n) \circ relu \circ A^i_j (n-1) \circ \dots \circ relu \circ A^i_j (1)
# where A^i_j (a) is the affine map (A^i_j (a) \cdot) \circ append(1,-) for biases
# Shape A = [number of submaps,input, input + 1]
def applyNonlinear(a,x):
  biasX = np.append(1,x)
  if(a.shape[0] < 2):
    return np.einsum("ij,j",a[0,:,:],biasX)
  thisMap = a[0,:,:]
  restMap = a[1:,:,:]
  return applyNonlinear(restMap,relu(np.einsum("ij,j",thisMap,biasX)))

#sampNonLin = 2 * np.random.rand(5,6,8) - 1
#sampInput = np.ones(5)
#print(applyNonlinear(sampNonLin,sampInput))
#print(applyNonlinear(sampNonLin,2 * sampInput))

def doTestNonLin():
  numTrainingSamps = 100
  trainingBias = 15 * np.random.normal(size=zSize)
  #print("Training bias was:",trainingBias)
  trainingStdev = 6
  trainingSamps = trainingBias + np.random.normal(size=(numTrainingSamps,zSize),scale=trainingStdev)
  #print("Stdev loss:",np.log(trainingStdev))


  initVal = identityNonLin(zSize,2 * netDepth)
  minResult = minimize(lambda p: serialL(p,trainingSamps,zSize,netDepth),serializeNonLin(initVal),callback=lambda p: minimizeCallback(p,trainingSamps,zSize,netDepth),options={'disp':True,'maxiter':203})
  print(minResult)
  #print(applyNonlinear(np.reshape(identityNonLin,(zSize,zSize+1,1)),np.ones(zSize)))

  optEnc = deserializeNonLin(minResult.x[:int(minResult.x.size/2)],zSize,netDepth)
  optDec = deserializeNonLin(minResult.x[int(minResult.x.size/2):],zSize,netDepth)

  newExample = trainingBias + np.random.normal(size=zSize)
  print("Start:",newExample)
  encVal = runEncoder(optEnc,newExample)
  print("Encoded as:",encVal)
  decVal = runDecoder(optDec,newExample)
  print("Decoded as:",decVal)
  print("Training bias was:",trainingBias)
  print("Reconstruction Loss:",np.linalg.norm(newExample - decVal))

doTestNonLin()


def doSaxStuff():
  fs, saxdata = wavfile.read("saxsamp.wav")

  leftSaxData = saxdata[:,0]

  def compactGaussian(length=50):
    x = np.linspace(-1,1,length)
    return np.exp(1-1/(1-(np.square(x)-0.00001)))

  def informationWindow(y):
    g = compactGaussian(length=np.size(y))
    noise = 2 * np.random.rand(np.size(y)) - 1
    return g * y + (1-g) * noise

  x = np.linspace(-1,1,50)
  g = compactGaussian(np.size(x))
  plt.plot(x,g)
  plt.show(block=True)

  signal = np.sin(10 * x)
  plt.figure(figsize=(18,10))
  plt.subplot(211)
  plt.plot(x,signal)
  for i in range(0,10):
    plt.scatter(x,informationWindow(signal))

  plt.subplot(212)

  windowedSax = informationWindow(leftSaxData)
  plt.plot(windowedSax)
  plt.plot(leftSaxData)

  wavfile.write("windowedSax.wav",fs,windowedSax)
  plt.show(block=False)
  input("...")

