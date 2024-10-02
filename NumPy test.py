import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

##Numpy test
#Multi-line comment
'''
def sum_trad():
    start = time.time()
    x = range(1000000)
    y = range(1000000)
    z = []
    for i in range(len(x)):
        z.append(x[i] + y[i])
    return time.time() - start

def sum_numpy():
    start = time.time()
    x = np.arange(1000000)
    y = np.arange(1000000)
    z = x + y
    return time.time() - start
print ('time sum: ', sum_trad(), 'time numpy: ', sum_numpy())
'''

#Array creation
'''

arr = np.array([2,5,6,9],float)

print(arr,type(arr))

arr = np.array([1,2,3], float)
arr.tolist()
print(arr)  
print(list(arr))

arr= np.array([1,2,3],float)
arr1=arr
arr2 = arr.copy()
arr[0]=0
print(arr)
print(arr1)
print(arr2)
'''

#arr = np.array([10,20,33], float)
#print(arr)

#arr.fill(1)
#print(list(arr))
#print(np.random.normal(0,1,5))
#prin=t(np.random.random(5))
#print(np.identity(3, dtype=float))
#print(np.eye(3, k=-1, dtype=float))
#print(np.ones((3,3), dtype=float))
#print(np.zeros(6, dtype=int))

#arr=np.array([[13,32,31], [64,25,76]], float)
#print(np.zeros_like(arr))
#print(arr)

#print(np.ones_like(arr))
#print(arr)

#arr1 = np.array([1,3,2])
#arr2 = np.array([4,5,6])
#print(np.vstack([arr1,arr2]))

#print(np.random.rand(2,3))

#Multivariate normal distribution
#print(np.random.multivariate_normal([10,0], [[3,1],[1,4]], size=[5,])) 

#arr = np.array([2., 6., 5., 5.])
#print(arr[:3])
#print(arr[3])
#arr[0]=5.0
#print(arr)

#Unique values of the array
#print(np.unique(arr)) 
#Sort the array
#print(np.sort(arr)) 
#Shuffle the array
#np.random.shuffle(arr) #Affects the original array
#print(arr)

#Compares two arrays
#statement=np.array_equal(arr, np.array([1,3,2]))
#print(statement)

matrix = np.array([[4.,5.,6.], [2,3,6]], float)
#print(matrix)
#print(matrix[0,0])
#print(matrix[0,2])

#arr = np.array([[4.,5.,6.], [2.,3.,6.]], float) #2x3 array
#print(arr[1:2,2:3]) #2nd row, 3rd column

#print(arr[1,:]) #2nd row, all columns
#print(arr[:,2]) #all rows, 3rd column
#print(arr[-1:,-2:]) #last row, last 2 columns

#arr = np.array([[10,29,23],[24,25,46]],float)#2x3 array
#print(arr)
#print(arr.flatten()) #Flatten the array
#print(arr.shape) #Shape of the array
#print(arr.dtype) #Data type of the array

#int_arr = matrix.astype(np.int32) #Convert the array-type to int32
#print(int_arr.dtype)

#arr=np.array([[1,2,3],[4,5,6]],float)
#print(len(arr)) #Number of rows

#print(2 in arr) #Check if 2 is in the array
#print(0 in arr) #Check if 0 is in the array

#arr = np.array(range(8),float)
#print(arr)
#print(arr.reshape((4,2))) #Reshape the array to 4x2
#print(arr.shape) #Shape of the array

#arr=np.array(range(6),float).reshape((2,3)) #Create array and reshape to 2x3
#print(arr)
#print(arr.transpose()) #Transpose the array

#matrix = np.arange(15).reshape((3,5)) #Create array and reshape to 3x5
#print(matrix)
#print(matrix.T) #Transpose the array
#print(matrix.transpose()) #Same as above

#arr=np.array([14,32,13],float)
#print(arr)
#print(arr[:,np.newaxis]) #Add a new axis to the array
#print(arr[:,np.newaxis].shape) #Shape of the new array
#print(arr[np.newaxis,:] ) #Add a new axis to the array
#print(arr[np.newaxis,:].shape) #Shape of the new array 

#arr1 = np.array([10,22],float)
#arr2 = np.array([31,43,54,61],float)
#arr3 = np.array([71,82,29],float)
#print(np.concatenate((arr1,arr2,arr3))) #Concatenate the arrays

#arr1 = np.array([[11,12],[32,42]],float)
#arr2 = np.array([[54,26],[27,28]],float)
#print(np.concatenate((arr1,arr2))) #Concatenate the arrays
#print(np.concatenate((arr1,arr2),axis=0)) #Concatenate the arrays along the rows
#print(np.concatenate((arr1,arr2),axis=1)) #Concatenate the arrays


#arr = np.array([10,20,30],float)
#str = arr.tobytes() #Convert the array to a binarystring
#print(str)
#print(np.frombuffer(str)) #Convert the binary string back to an array

#arr1 = np.array([1,2,3],float)
#arr2 = np.array([1,2,3],float)  
#print(arr1+arr2) #Add the arrays
#print(arr1-arr2) #Subtract the arrays
#print(arr1*arr2) #Multiply the arrays
#print(arr1/arr2) #Divide the arrays
#print(arr1%arr2) #Modulus of the arrays
#print(arr1**arr2) #Raise the arrays to the power of the second array

#arr1 = np.array([1,2,3],float)
#arr2 = np.array([1,2],float)
#print(arr1+arr2) #Can't add arrays of different shapes

#arr1 = np.array([[1,2],[3,4], [5,6]],float)
#arr2 = np.array([1,2],float)
#print(arr1)
#print(arr2)
#print(arr1+arr2) #Broadcasting, add the array to each row of the matrix
#For explicit broadcasting use np.newaxis
#print(arr1+arr2[np.newaxis,:]) #Add the array to each row of the matrix

#arr=np.array([[1,2],[5,9]],float)
#print(arr>=7) #Check if each element is greater than or equal to 7
#print(arr[arr>=7]) #Return the elements that are greater than or equal to 7
#print(arr[np.logical_and(arr>=5, arr<11)]) #Return the elements that are greater than 5 and less than 10

#arr1 = np.array([1,4,5,9],float)
#arr2 = np.array([0,1,1,3,1,1,1],int)
#print(arr1[arr2]) #Select elements from arr1 based on the indices in arr2
#print(arr1.take(arr2)) #Same as above
#print(arr1[[0,1,1,3,1]])

#arr1 = np.array([[1,2],[5,13]],float)
#arr2 = np.array([1,0,0,1],int)
#arr3 = np.array([1,1,0,1],int)
#print(arr1)
#print(arr1[arr2,arr3]) #Select elements from arr1 based on the indices in arr2 and arr3

#arr1 = np.array([7,6,6,9],float)
#arr2 = np.array([1,0,1,3,3,1],int)
#print(arr1.take(arr2)) #Select elements from arr1 based on the indices in arr2

#arr1 = np.array([2,1,6,2,1,9], float)
#arr2 = np.array([3,10,2],float)
#arr1.put([1,4],arr2) #Put the values in arr2 into the indices in arr1
#print(arr1)

#arr1=np.array([[11,22],[23,14]],float)
#arr2=np.array([[25,30],[13,33]],float)
#print(arr1*arr2) #Element-wise multiplication

#X = np.arange(15).reshape((3,5))
#print(X)
#print(X.T)
#print(np.dot(X .T,X))#X^T*X

#arr1 = np.array([12,43,10],float)
#arr2 = np.array([21,42,14],float)
#print(np.outer(arr1,arr2)) #Outer product of the arrays
#print(np.inner(arr1,arr2)) #Inner product of the arrays
#print(np.cross(arr1,arr2)) #Cross product of the arrays


#matrix = np.array([[74,22,10],[92,31,17],[21,22,12]],float)
#print(matrix)
#print(np.linalg.det(matrix)) #Determinant of the matrix
#inv_matrix = np.linalg.inv(matrix) #Inverse of the matrix
#print(inv_matrix)
#print(np.dot(inv_matrix,matrix)) #Matrix multiplication of the matrix and its inverse

#vals, vecs = np.linalg.eig(matrix) #Eigenvalues and eigenvectors of the matrix
#print(vals)
#print(vecs)

#arr=np.random.rand(8,4)
#print(arr.mean()) #Mean of the array
#print(np.mean(arr)) #Mean of the array using numpy
#print(arr.sum()) #Sum of the array

##Pandas test (Done in console, remove all "I/O")
#Series
#obj=pd.Series([3,5,-2,1])
#print(obj)

#plt.plot([10,5,2,4],color='green',label='line1',linewidth=5)
#plt.ylabel('y',fontsize=40)
#plt.xlabel('x',fontsize=40)
#plt.axis([0,3,0,15])
#plt.show()

#fig = plt.figure(figsize=(10,10))
#ax=fig.add_subplot(111)
#ax.set_xlabel('x',fontsize=40)
#ax.set_ylabel('y',fontsize=40)
#fig.suptitle('figure',fontsize=40)
#ax.plot([10,5,2,4],color='green',label='line1',linewidth=5)
#fig.savefig('figure.png')

#fig=plt.figure(figsize=(10,10))
#ax=fig.add_subplot(111)
#r=np.arange(0,10,0.3)
#p1,=ax.plot(r,r,'r--',label='line 1',linewidth=10)
#p2,=ax.plot(r,r**0.5,'bs',label='line 2',linewidth=10)
#p3,=ax.plot(r,np.sin(r),'g^',label='line 3',markersize=10)
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles, labels,fontsize=40)
#ax.set_xlabel('x',fontsize=40)
#ax.set_ylabel('y',fontsize=40)
#fig.suptitle('figure 1',fontsize=40)
#fig.savefig('figure_multipleLines.png')

#colors = ['b','c','y','m','r']
#fig=plt.figure(figsize=(10,10))
#ax=fig.add_subplot(111)
#ax.scatter(np.random.random(10),np.random.random(10),marker='x', color=colors[0])
#p1=ax.scatter(np.random.random(10),np.random.random(10),marker='x', color=colors[1],s=50)
#p2=ax.scatter(np.random.random(10),np.random.random(10),marker='o', color=colors[2],s=50)
#p3=ax.scatter(np.random.random(10),np.random.random(10),marker='o', color=colors[3],s=50)
#ax.legend((p1,p2,p3),('point 1','point 2','point 3'),fontsize=20)
#ax.set_xlabel('x',fontsize=40)
#ax.set_ylabel('y',fontsize=40)
#fig.savefig('figure_scatterplot.png')