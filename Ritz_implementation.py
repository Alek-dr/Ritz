import matplotlib.pyplot as plt
from numpy import linspace
from sklearn.metrics import mean_squared_error
from sympy import Symbol, Integral, Derivative, init_printing, symbols, solve, diff

def get_u(fi,n,const):
    #Basis functions sum
    y = 0
    for i in range(n):
        y += const[i]*fi.subs(k,i+1)
    return y

def ritz(a,b,n,f,fi,fi0):
    u = Symbol("u")
    const = [Symbol("C"+str(i)) for i in range(1,n+1)]
    u_ = get_u(fi,n,const)
    u_+=fi0
    f = f.subs({u:u_})
    J = Integral(f,(x,a,b))
    F = J.doit()
    #Partial derrivatives
    derr = [diff(F,c) for c in const]
    #Get С
    sol = solve(derr,const)
    #Put С into u
    u = u_.subs(sol)
    return u

init_printing()
J,k,x,u = symbols("J,k,x,u")

#Integrand
f = Derivative(u,x)**2 + (x**2)*(u**2) + (2*x+2)*u

#Limits of integration
a = -1
b = 1
#u(a) = u_a, u(b) = u_b
u_a = 0
u_b = 2

#basis functions
fi = x**(k-1)*(x-a)*(x-b)

fi0 = ((u_b - u_a)/(b-a))*(x-a)+u_a

u2 = ritz(a,b,2,f,fi,fi0)
u3 = ritz(a,b,3,f,fi,fi0)

x_ = linspace(a,b,10)

y_2 = [u2.subs(x,v).evalf() for v in x_]
y_3 = [u3.subs(x,v).evalf() for v in x_]

line1, = plt.plot(x_,y_2, 'b-', markersize=2, label="N=2")
line2, = plt.plot(x_,y_2, 'ro', markersize=4, label="N=3")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.grid(True)
plt.show()

print("MSE {}".format(mean_squared_error(y_3,y_2)))