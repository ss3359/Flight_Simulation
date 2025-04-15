import random 
import numpy as np  
import pandas as pd 
import math 
import pygame
from sympy import symbols,diff,Matrix,cos,sin
import matplotlib.pyplot as plt


'''
Overview: This is code represnting a flight simulation in two dimensions
This will be using the equations for Lift, Drag, Weight, and Thrust. Then, we will
sum all of the forces using Newton's Second Law. Finally, we will use the Runge
Kutta method to update the velocity and the position. 
'''


class Plane: 
    #Gravity
    g=9.81 #m/s^2

    #Density Of Air At Crusing Altitude
    rho=0.413 # kg/m^3

    #Angular Rates
    p=0
    q=0
    r=0

    #Velocities
    u=0 #Forward Velocity
    v=0 #Side Velocity 
    w=0 #Vertical Velocity

    #Thrust Velocities And Thrust Angle
    m_dot=700 #kg/s
    Vexit=500 #m/s
    Vinlet=250 #m/s
    theta_t=0 #degrees (Thurst angle, thrust is purely forward)
    

    #Time Step
    h=0.005
    def __init__(self,x,y,z,vx,vy,vz,theta,phi,psi,m,S,Cl,Cd,alpha,dtheta,dphi,dpsi):
        self.x=x
        self.y=y
        self.z=z
        self.vx=vx
        self.vy=vy
        self.vz=vz
        self.theta=theta
        self.phi=phi
        self.psi=psi
        self.m=m
        self.S=S
        self.Cl=Cl
        self.Cd=Cd
        self.alpha=alpha
        self.dtheta=theta
        self.dphi=dphi
        self.dpsi=dpsi

    def Lift(self,x,y,z):
        vel_length=min(math.sqrt(x**2+y**2+z**2),500)
        # vel_unit_vector=np.array([x/vel_length,y/vel_length,z/vel_length])
        L = (0.5)*self.rho*vel_length**2*self.S*self.Cl
        Lx=L*math.cos(self.theta)
        Ly=0
        Lz=L*math.sin(self.theta)
        Lift=np.array([Lx,Ly,Lz])
        return Lift
    
    def Thrust(self):
       thrust_dir=np.array([math.cos(self.theta)*math.cos(self.phi), math.sin(self.phi), 
                                   math.sin(self.theta)*math.cos(self.phi)])

       T=self.m_dot*(self.Vexit-self.Vinlet)
       Thrust_vector=T*thrust_dir
       
       return Thrust_vector
    
    def Drag(self,x,y,z):
        vel_length=min(math.sqrt(x**2+y**2+z**2),500)
        if(vel_length==0):
            return np.array([0,0,0])
        drag_dir=np.array([-x/vel_length,-y/vel_length,-z/vel_length])
        D=(0.5)*self.rho*vel_length**2*self.S*self.Cd
        Drag=D*drag_dir

        return np.array([Drag[0],Drag[1],Drag[2]])
    
    def Weight(self):
       
        self.m=max(self.m-(5)*self.h,1000)

        Wx=-(self.m)*(self.g)*math.sin(self.theta)
        Wy=(self.m)*(self.g)*math.sin(self.phi)*math.cos(self.theta)
        Wz=(self.m)*(self.g)*math.cos(self.phi)*math.cos(self.theta)
      
        Weight=np.array([Wx,Wy,Wz])
        return Weight   

    #Rate of Changes of Roll, Pitch, and Yaw angles respectively: 
    def Dphi(self,J=0,K=0,L=0):
        p=math.radians(self.phi)
        q=math.radians(self.theta)
        r=math.radians(self.psi)
        return (p+J)+(q+K)*(math.sin((p+J))*math.tan((q+K)))+r*(math.cos((p+J))*math.tan((q+K)))
        
    def Dtheta(self,J=0,K=0,L=0):
        p=math.radians(self.phi)
        q=math.radians(self.theta)
        r=math.radians(self.psi)

        return (q+K)*(math.cos((p+J)))-(r+L)*(math.sin((p+J)))
    def Dpsi(self,J=0,K=0,L=0):
        p=math.radians(self.phi)
        q=math.radians(self.theta)
        r=math.radians(self.psi)
        return (q+K)*(math.sin((p+J))/math.cos((q+K)))+(r+L)*(math.cos((p+J))/math.cos((q+K)))
        
    def DPos(self,J=0,K=0,L=0):
        dx=(self.vx)+J
        dy=(self.vy)+K
        dz=(self.vz)+L

        return dx,dy,dz
        # This function will use Newtons Second Law Of Motion F=ma.
        # Fx=m*dVx => dVx = (1/m)*Fx
        # Fy=m*dVy => dVy = (1/m)*Fy
        # Fz=m*dVz => dVz = (1/m)*Fz

        #To update the position, we can use 
        #x = x+v*dx 
        #y = y+v*dy
        #z = z+v*dz 
        #
        # We will use the Runge Kutta Method to compute the new velocity
        # and the poistion of the airplane 

    def GetVectors(self,vx,vy,vz,J=0,K=0,L=0):
        Lift=self.Lift(vx+J,vy+K,vz+L)
        Thrust=self.Thrust()
        Drag=self.Drag(vx+J,vy+K,vz+L)
        Weight=self.Weight()

        return Lift,Thrust,Drag,Weight


    def UpdateVelocityAndPosition(self,dt):
           
        L,T,D,W=self.GetVectors(self.vx,self.vy,self.vz)
        dx1,dy1,dz1=self.DPos()

        j1=dt*(1/self.m)*(L[0]+T[0]+D[0]+W[0])
        a1=dt*dx1

        k1=dt*(1/self.m)*(L[1]+T[1]+D[1]+W[1])
        b1=dt*dy1

        l1=dt*(1/self.m)*(L[2]+T[2]+D[2]+W[2])
        c1=dt*dz1

        L,T,D,W=self.GetVectors(self.vx,self.vy,self.vz,j1/2,k1/2,l1/2)
        dx2,dy2,dz2=self.DPos(a1/2,b1/2,c1/2)


        j2=dt*(1/self.m)*(L[0]+T[0]+D[0]+W[0])
        a2=dt*dx2
        k2=dt*(1/self.m)*(L[1]+T[1]+D[1]+W[1])
        b2=dt*dy2
        l2=dt*(1/self.m)*(L[2]+T[2]+D[2]+W[2])
        c2=dt*dz2


        L,T,D,W=self.GetVectors(self.vx,self.vy,self.vz,j2/2,k2/2,l2/2)
        dx3,dy3,dz3=self.DPos(a2/2,b2/2,c2/2)


        j3=dt*(1/self.m)*(L[0]+T[0]+D[0]+W[0])
        a3=dt*dx3

        k3=dt*(1/self.m)*(L[1]+T[1]+D[1]+W[1])
        b3=dt*dy3

        l3=dt*(1/self.m)*(L[2]+T[2]+D[2]+W[2])
        c3=dt*dz3

        L,T,D,W=self.GetVectors(self.vx,self.vy,self.vz,j3,k3,l3)
        dx4,dy4,dz4=self.DPos(a3,b3,c3)

        j4=dt*(1/self.m)*(L[0]+T[0]+D[0]+W[0])
        a4=dt*dx4

        k4=dt*(1/self.m)*(L[1]+T[1]+D[1]+W[1])
        b4=dt*dy4

        l4=dt*(1/self.m)*(L[2]+T[2]+D[2]+W[2])
        c4=dt*dz4

        #Velocity Update
        self.vx+=(1/6)*(j1+(2*j2)+(2*j3)+j4)
        self.vy+=(1/6)*(k1+(2*k2)+(2*k3)+k4)
        self.vz+=(1/6)*(l1+(2*l2)+(2*l3)+l4) 
        V=np.array([self.vx,self.vy,self.vz])

        #Position Update 
        self.x+=(1/6)*(a1+(2*a2)+(2*a3)+a4)
        self.y+=(1/6)*(b1+(2*b2)+(2*b3)+b4)
        self.z+=(1/6)*(c1+(2*c2)+(2*c3)+c4)

        P=np.array([self.x,self.y,self.z])


        return P,V           

    def UpdateAngles(self,dt):

        #iter=10
        # dt=0.01 

        j1=dt*self.Dphi()
        k1=dt*self.Dtheta()
        l1=dt*self.Dpsi()

        j2=dt*self.Dphi(j1/2,k1/2,l1/2)
        k2=dt*self.Dtheta(j1/2,k1/2,l1/2)
        l2=dt*self.Dpsi(j1/2,k1/2,l1/2)

        j3=dt*self.Dphi(j2/2,k2/2,l2/2)
        k3=dt*self.Dtheta(j2/2,k2/2,l2/2)
        l3=dt*self.Dpsi(j2/2,k2/2,l2/2)

        j4=dt*self.Dphi(j3,k3,l3)
        k4=dt*self.Dtheta(j3,k3,l3)
        l4=dt*self.Dpsi(j3,k3,l3)

        self.phi+=(1/6)*(j1+(2*j2)+(2*j3)+j4)
        self.theta+=(1/6)*(k1+(2*k2)+(2*k3)+k4)
        self.psi+=(1/6)*(l1+(2*l2)+(2*l3)+l4)

        return np.array([self.phi,self.theta,self.psi])






def Flight(): 

     #Phases of Flight are as follows: 
    # 1->Takeoff 
    # 2->Climb 
    # 3->Cruising 
    # 4-> Descent
    # 5-> Landing

    PHASES=[1,2,3,4,5]

    #Constants/Constraints 

    S= 837.57 # Surface Area Of Wing m^2
    M=4500 #kg, mass of the airplane
    dt=0.01 #time step
    x=0 #meters
    y=0 #meters
    z=0 #10000 #meters (crusing altitude)
    vx=0 #230 #(m/s) Mach 0.8 at crusing altitude
    vy=0
    vz=0 # Level Flight

    phi= 0 # roll angle wings are level (degrees)
    theta= 0# # 2 pitch angle in degrees(slight nose-up to maintiain lift) 
    psi= 0 #1 #yaw angle in degrres (this will vary depending on the flight) 
    Cd= 0.05 # 0.02 Drag Coefficient at Crusing Altitude 
    Cl=0.1 #0.4 Lift Coefficent at Crusing Altitude 
    dtheta=0 # degrees/sec pitch rate (steady)
    dphi=0  #degrees/sec, roll rate (constant altitude)
    dpsi=0 #degrees/sec, yaw rate (small adjustments only)
    alpha= 0 #5 #angle of attack of airplane (degrees)
    # iter= 0 #number of iterations. 

    P=Plane(x,y,z,vx,vy,vz,theta,phi,psi,M,S,Cl,Cd,alpha,dtheta,dphi,dpsi)

    time =0
    max_time=3000
    phase=1
    time_log=[]
    altitude_log=[]
    while(time<=max_time):
        #Takeoff
        if(phase==1 and P.z<=50):
            P.theta=math.radians(10)
            P.Cd=0.8
            P.Cl=0.1
        elif phase==1: 
            phase=2 
        
        #Climb To Crusing Altitude 
        if(phase==2 and P.z<1000): 
            P.theta=math.radians(5)
        elif(phase==2): 
            phase=3 

        #Cruising Altitude
        if(phase==3 and P.z< 7000):
            P.theta=math.radians(0)
            P.Cl=0.4
            P.Cd=0.05
        elif(phase==3):
            phase=4

        #Descent
        if(phase==4 and P.z>50): 
            P.theta=math.radians(-5)
            P.Cl=0.2
            P.Cd=0.1
        elif(phase==4): 
            phase=5

        #Landing
        if(phase==5):
            P.theta=math.radians(-2) #Final Approach
            if(P.z<=0): 
                P.z=0
                print("Landing Success!")
                break

        Pos,Vel=P.UpdateVelocityAndPosition(dt)
        Angles=P.UpdateAngles(dt)
        print(f"Time: {round(time,2)}s | Phase: {phase} | Altitude: {round(P.z,2)} m ")
        time_log.append(time)
        altitude_log.append(P.z)

        time+=dt

    return time_log, altitude_log


def main(): 

    time_log, altitude_log=Flight()
    plt.plot(time_log,altitude_log)
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.title("Flight Path")
    plt.show()    
main()
