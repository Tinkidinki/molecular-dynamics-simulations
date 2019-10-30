import numpy as np
import random
import matplotlib.pyplot as plt
import shutil
import os
import glob

#Select functionality as "euler", "verlet" or "periodic_verlet"
INTEGRATION_TYPE = "periodic_verlet"

#Change virial coefficients:
V_1 = 0.5
V_2 = 0.4


class Simulator():
    # Please give volume in m^3 and step_size in seconds
    def __init__(self,N, epsilon, sigma, V, num_steps, step_size):
        self.N = N 
        self.e = epsilon
        self.s = sigma
        self.V = V 
        self.num_steps = num_steps
        self.step_size = step_size
        self.len = self.V**(1/3)
        self.m = 6.624 * 10**-26  # Mass of argon

    def get_rand_pos(self):
        return [random.uniform(0, self.len) for i in range(3)]

    def get_rand_momentum(self):
        return [random.uniform(0,500)*self.m for i in range(3)]

    
    def initialise_system(self):
        self.position_vector = np.array([self.get_rand_pos() for i in range(self.N)])
        self.momentum_vector = np.array([self.get_rand_momentum() for i in range(self.N)])
        self.force = np.array([[0,0,0] for i in range(self.N)])
        self.kinetic_energy = [0 for i in range(self.num_steps)]
        self.potential_energy = [0 for i in range(self.num_steps)]
        self.total_momentum = [0 for i in range(self.num_steps)]

       
    
    def write_to_file(self,t, folder):
        '''write current positions and momentums''' 
        position_string = ""
        momentum_string = ""
        for i in range(self.N):
            position_string +=str(self.position_vector[i][0]) + " " + str(self.position_vector[i][1]) + " " + str(self.position_vector[i][2]) + "\n"
            momentum_string +=str(self.momentum_vector[i][0]) + " " + str(self.momentum_vector[i][1]) + " " + str(self.momentum_vector[i][2]) + "\n"

        with open(folder+"/position.txt", "a+") as f:
            f.write("Timestep: "+str(t)+"\n")
            f.write(position_string)

        with open(folder+"/momentum.txt", "a+") as f:
            f.write("Timestep: "+str(t)+"\n")
            f.write(momentum_string)
            


    # Check value of force
    def calc_force(self,pos1, pos2):
        diff = pos1 - pos2

        if INTEGRATION_TYPE == "periodic_verlet":
            for i in range(len(diff)):
                if diff[i] >= self.len: diff[i] %= self.len

        r = np.linalg.norm(diff)
        force = 8*self.e*(6*self.e**12/r**14 - 3*self.s**6/r**8)* diff
        return force
    
    def total_force(self,i):
        total_force = np.array([0,0,0])
        for j in range(self.N):
            if j==i:
                continue
            total_force = total_force +  self.calc_force(self.position_vector[i], self.position_vector[j])

        return total_force
    
    def total_potential_energy(self):
        pe = 0
        for i in range(self.N):
            for j in range(i+1, self.N):
                r = np.linalg.norm(self.position_vector[i] - self.position_vector[j])
                pe+= 4*self.e*((self.s/r)**12 - (self.s/r)**6) 
    
        return pe          

    def total_kinetic_energy(self):
        ke = 0
        for i in range(self.N):
            ke += (np.linalg.norm(self.momentum_vector[i]))**2/(2*self.m)
  
        return ke

    def make_force_vector(self):
        self.force = np.array([self.total_force(i) for i in range(self.N)])
    

    def plot_values(self, folder):
        plt.plot(self.kinetic_energy, label="K.E.")
        plt.plot(self.potential_energy, label = "P.E")
        # plt.plot(self.kinetic_energy + self.potential_energy, label = "Total Energy")
        plt.savefig(folder+'/energy.png')
        plt.show()
        plt.plot(self.total_momentum, label = "Total Momentum")
        plt.savefig(folder+'/momentum.png')
        plt.show()

    def temp(self, avg_ke): return 2/3*(6.023*10**23)/8.314*avg_ke

    def write_energies(self, folder):
        with open(folder+"/energy.txt", "a+") as f:
            for i in range(self.num_steps):
                f.write(str(self.kinetic_energy[i])+ " ")
                f.write(str(self.potential_energy[i])+ "\n")

        with open(folder + "/temp_pressure.txt", "a+") as f:
            for i in range(self.num_steps):
                temp = self.temp(self.kinetic_energy[i]/self.N)
                Z = 1 + V_1*(self.N/self.V) + V_2*(self.N/self.V)**2
                pressure = 8.314*temp*Z
                f.write(str(temp) + "    " + str(pressure) + "\n")

        with open(folder + "/pressure.txt", "a+") as f:
            for i in range(self.num_steps):
                
                f.write(str())

    
    def do_calculations(self):

    #------------------------------------------EULER INTEGRATION-------------------------------------------------
   
        if INTEGRATION_TYPE == "euler": 

            files = glob.glob('Euler/*')
            for f in files:
                os.remove(f)

        
            self.initialise_system()

            for t in range(self.num_steps):
                self.write_to_file(t, "Euler")

                self.make_force_vector()
                self.momentum_vector += self.force*self.step_size
                self.position_vector += self.momentum_vector*self.step_size/self.m 
                    

                self.potential_energy[t] = self.total_potential_energy()
            
                self.kinetic_energy[t] = self.total_kinetic_energy()
                
                total_momentum_v = np.sum(self.momentum_vector, axis = 0)
                self.total_momentum[t] = np.linalg.norm(total_momentum_v)
        
        
            self.plot_values("Euler")
            self.write_energies("Euler")

    #--------------------------------------------------------------------------------------------------

    #---------------------------VERLET INTEGRATION-----------------------------------------------------

        elif INTEGRATION_TYPE == "verlet":
        
            files = glob.glob('Verlet/*')
            for f in files:
                os.remove(f)

            self.initialise_system()

            for t in range(self.num_steps):
                self.write_to_file(t, "Verlet")

                self.momentum_vector += self.force*self.step_size/2
                self.make_force_vector()
                self.momentum_vector += self.force*self.step_size/2
                self.position_vector += self.momentum_vector*self.step_size/self.m + self.force/(2*self.m)*(self.step_size)**2

                self.potential_energy[t] = self.total_potential_energy()
            
                self.kinetic_energy[t] = self.total_kinetic_energy()

                total_momentum_v = np.sum(self.momentum_vector, axis = 0)
                self.total_momentum[t] = np.linalg.norm(total_momentum_v)

            self.plot_values("Verlet")
            self.write_energies("Verlet")

    #--------------------------------------PERIODIC VERLET INTEGRATION----------------------------------------------

        elif INTEGRATION_TYPE == "periodic_verlet":
            
            files = glob.glob('PeriodicVerlet/*')
            for f in files:
                os.remove(f)

            self.initialise_system()

            for t in range(self.num_steps):
                self.write_to_file(t, "PeriodicVerlet")

                self.momentum_vector += self.force*self.step_size/2
                self.make_force_vector()
                self.momentum_vector += self.force*self.step_size/2
                self.position_vector += self.momentum_vector*self.step_size/self.m + self.force/(2*self.m)*(self.step_size)**2

                
                for i in range(self.N):
                    if (self.position_vector[i][0] > self.len): self.position_vector[i][0] %= self.len
                    if (self.position_vector[i][1] > self.len): self.position_vector[i][1] %= self.len
                    if (self.position_vector[i][2] > self.len): self.position_vector[i][2] %= self.len
                    

                self.potential_energy[t] = self.total_potential_energy()
            
                self.kinetic_energy[t] = self.total_kinetic_energy()

                total_momentum_v = np.sum(self.momentum_vector, axis = 0)
                self.total_momentum[t] = np.linalg.norm(total_momentum_v)

            self.plot_values("PeriodicVerlet")
            self.write_energies("PeriodicVerlet")

    #-------------------------------------------------------------------------------------------------------------------------

    
            

sigma = 3.4 * 10**-10
epsilon = 1.65 * 10**-21
volume = 1

# Give the parameters you'd like below:
# All units are SI units
# (Number of particles, epsilon, sigma, volume, number of timesteps, time interval per timestep)
system = Simulator(20,epsilon, sigma, volume,10,1)
system.do_calculations()









