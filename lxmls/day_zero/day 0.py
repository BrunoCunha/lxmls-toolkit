#LxMLS day 0 exercices

print "Hello world!"

print 2**3

countries = ["Portugal", "Spain", "United Kingdom"]

len("asd")
"asd"[1]

countries[:2]


i = 2
while i <10:
    print i
    i += 2
    
hour = 16
if hour <12:
    print "Good morning"
elif hour >= 12 and hour <=20:
    print "good afternoon"
else:
    print "Good evening"
    

def greet(hour):
    if hour <0 or hour > 24:
        print "invalid hour: it should be between 0 and 24"
    elif hour <12:
        print "Good morning"
    elif hour >= 12 and hour <=20:
        print "good afternoon"
    else:
        print "Good evening"
        

def greet_w_exception(hour):
    if hour <0 or hour > 24:
        raise ValueError("invalid hour: it should be between 0 and 24")
    elif hour <12:
        print "Good morning"
    elif hour >= 12 and hour <=20:
        print "good afternoon"
    else:
        print "Good evening"
        
        
        
import numpy as np

np.random.normal

import my_tools
my_tools.my_print("asd")
reload(my_tools)


#0.4.4 matplotlib
import matplotlib.pyplot as plt

X = np.linspace(-4,4,1000)

plt.plot(X, X**2+np.cos(X**2))
plt.savefig("simple.pdf")


Y = X**2
plt.plot(X,Y,"r")

Ints = np.arange(-4,5)
plt.plot(Ints, Ints**2, "bo")

plt.xlim(-4, 5, 4.5)
plt.ylim(-1, 17)
plt.show()


#0.4.5 Numpy

A = np.array([
        [1,2,3],
        [2,3,4],
        [4,5,6]        
        ])
    
A[0,:]    
A[0]    

A[:,0]
A[1:,0]


X = np.linspace(0, 4 * np.pi, 1000)
C = np.cos(X)
S = np.sin(X)

plt.plot(X, C)
plt.plot(X, S)

A = np.arange(100)
#these two make the same:
print np.mean(A)
print A.mean()

C = np.cos(A)
print C.ptp()


#compute f(x) = x^2 integral:

i = np.arange(0.0, 1000.0)

upper = (i/1000.0)**2
bottom = 1000

np.sum(upper/bottom) #shall be ~close to 1.0/3.0


#0.5 essential Linear algebra

import numpy as np
m = 3
n = 2 
a = np.zeros([m,n])
print a

print a.shape
print a.dtype.name

a = np.zeros([m,n], dtype=int)
print a.dtype.name

a = np.array([[2,3], [3,4]])
print a


#0.5.2
a = np.array([[2,3], [3,4]])
b = np.array([[1,1], [1,1]])


#Matrix multiplication:
d = np.dot(a,b)
print d

#a elements * b elements:
print a * b

#Outer product 
np.outer(a,b)

#identity matrix - np.eye
I = np.eye(2)
x = np.array([2.3, 3.4])

print I
print np.dot(I,x)

#Transpose matrix
a = np.array([[1,2], [3,4]])
print a.T

import numpy as np
import galton as galton
galton_data = galton.load()

print "mean heigth = ", np.mean(galton_data)
print "STD = ", np.std(galton_data)

print "mean heigth fathers = ", np.mean(galton_data[:,0])
print "mean heigth Sons = ", np.mean(galton_data[:,1])

import matplotlib.pyplot as plt
plt.title("Histogram of all heights")
plt.hist(np.ravel(galton_data))


plt.title("Father vs Son")    
plt.scatter(galton_data[:,1], galton_data[:,0])


pure = galton_data
noise = np.random.normal(0,1, len(galton_data))
signal = galton_data
signal[:,0] = pure[:,0] + noise 
signal[:,1] = pure[:,1] + noise 
plt.scatter(signal[:,1], signal[:,0])



plt.hist(galton_data, bins=25, stacked = True, orientation= "horizontal")       
         
print "Continue here"
