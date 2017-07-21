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


#0.7 onwards:


#error of ∂∂we (pag. 29)

import numpy as np
import matplotlib.pyplot as plt
X = np.linspace(-8, 8, 1000)
Y = (X+2)**2 - 16*np.exp(-((X-2)**2))


# derivative of the function f
def get_Y_dev(x):
    return (2*x+4)-16*(-2*x + 4)*np.exp(-((x-2)**2))


def gradient_descent(start_x,func,grad):
    # Precision of the solution
    prec = 0.0001
    #Use a fixed small step size
    step_size = 0.1
    #max iterations
    max_iter = 100
    x_new = start_x
    res = []
    for i in xrange(max_iter):
        x_old = x_new
        #Use beta egual to -1 for gradient descent
        x_new = x_old - step_size * grad(x_new)
        f_x_new = func(x_new)
        f_x_old = func(x_old)
        res.append([x_new,f_x_new])
        if(abs(f_x_new - f_x_old) < prec):
            print "change in function values too small, leaving"
            return np.array(res)
    print "exceeded maximum number of iterations, leaving"
    return np.array(res)

def grad_desc(start_x, eps, prec):
    '''
    runs the gradient descent algorithm and returns the list of estimates
    
    example of use grad_desc(start_x=3.9, eps=0.01, prec=0.00001)
    '''
    x_new = start_x
    x_old = start_x + prec * 2
    res = [x_new]
    while abs(x_old-x_new) > prec:
        x_old = x_new
        x_new = x_old - eps * get_Y_dev(x_new)
        res.append(x_new)
    return np.array(res)


def get_error(w):
    sum = 0
    for i in xrange(0, len(X)):
        temp = (X[i].T.dot(w) - get_Y_dev[i])
        sum += temp**2
    return sum




np.array([0.1,0.01,0.001])



print "Continue here"
	