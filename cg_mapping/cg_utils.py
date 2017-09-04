import mdtraj
import math
import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import pdb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import Counter

class State(object):
    """ Container to store basic thermodynamic information

    Parameters
    ---------
    k_b : float
        Boltzmann constant
    T : float
        Temperature
        """

    def __init__(self, k_b=8.314e-3, T=305):
        self._k_b = k_b
        self._T = T
    
    @property
    def k_b(self):
        return self._k_b

    @property
    def T(self):
        return self._T

    @k_b.setter
    def k_b(self, k_b):
        self._k_b = k_b

    @T.setter
    def T(self, T):
        self._T = T

    def __str__(self):
        return("k_B = {} \nT = {}".format(self.k_b, self.T))
    

    def gaussian(self, x,  x0, w, A):
        """ Generic gaussian function
    
        Parameters
        ----------
        x0 : float
            Mean around which gaussian is centered
        w : float
            width of gaussian
        A : float
            Total area of gaussian
    
        Returns
        -------
        Probability
    
        Notes
        -----
        Equation form follows Milano, Goudeau, and Muller-Plathe (2005)
        """
        return (A/(w*np.sqrt(np.pi/2))) * np.exp(-2*(x-x0)**2/(w**2))
    
    def gaussian_to_energy(self, x, constant, a, x0):
        """ Compute energy distribution from gaussian distribution
    
        Parameters
        ----------
        constant : float
        a : coefficient on harmonic term
        x0 : reference value
    
        Returns
        -------
    
        Notes
        -----
        Equation form follows Milano, Goudeau, and Muller-Plathe (2005)
        a is only half the value of the force constant, and needs to be 
        doubled 
        """
        return constant + 0.5*a*(x-x0)**2
    
    
    def harmonic_energy(self, x_val, x0, force_constant):
        """ Generic harmonic energy evaluation
    
        Parameters
        ----------
        x_val : float
            Independent variable for which we are calculating energy 
        x0 : float
            harmonic reference
        force_constant : float
            harmonic force constant
    
        Returns
        -------
        energy_eval
        """
        return 0.5*force_constant*((x_val-x0)**2)
    
    
    def harmonic_force(self, x_val, x0, force_constant):
        """ Generic harmonic force evaluation
    
        Parameters
        ----------
        x_val : float
            Independent variable for which we are calculating energy 
        x0 : float
            harmonic reference
        force_constant : float
            harmonic force constant
    
        Returns
        -------
        force_eval
        """
        return force_constant*(x_val-x0)
    
    def fit_to_gaussian(self, independent_vars, dependent_vars, energy_fit=False):
        """ Fit values to gaussian distribution
    
        Parameters
        ----------
        independent_vars : list()
            Likely the bond lengths or angles
        dependent_vars: list()
    
        Returns
        -------
        harmonic_parameters : dict()
            force_constant, x0

        Notes
        -----
            """
    
        # Fit probabilities to gaussian probabilities
        if not energy_fit:
            params, covar = curve_fit(self.gaussian, independent_vars, 
                    dependent_vars)#, method='dogbox', bounds = [(0,0,-np.inf), 
                        #(np.inf, np.inf, np.inf)])
            x0 = params[0]
            w = params[1]
            A = params[2]

            # Extracts sprint constant
            force_constant = self._k_b * self._T/w**2
            #bonded_parameters={'force_constant': force_constant, 'x0': x0}
        # Fit gaussian energies
        else:
            params, covar = curve_fit(self.gaussian_to_energy, independent_vars,
                    dependent_vars, method='dogbox', bounds=[(-np.inf,0,0), 
                        (np.inf, np.inf, np.inf)])
            constant = params[0]
            force_constant = params[1]
            x0 = params[2]
    
        bonded_parameters={'force_constant': force_constant, 'x0': x0}
        return bonded_parameters
    
    def compute_bond_parameters(self, traj, atomtype_i, atomtype_j):
        """
        Calculate bonded parameters from a trajectory
        Compute the probability distribution of bond lengths
        Curve fit to gaussian distribution
        
        Parameters
        ---------
        traj : mdtraj Trajectory
        atomtype_i : str
            First atomtype 
        atomtype_j : str
            Second atomtype
    
        Returns
        -------
        force_constant : float
            Harmonic bond constant
        x0 : float 
          Bond reference
    
        """
    
        topol = traj.topology
        target_pair = (atomtype_i, atomtype_j)
        bonded_pairs = []
        if len([(i,j) for i,j in topol.bonds]) == 0:
            sys.exit("No bonds detected, check your input files")
    
        for (i, j) in topol.bonds:
            if set((i.name, j.name)) == set(target_pair):
                bonded_pairs.append((i.index, j.index))

        if len(bonded_pairs) == 0:
            #print("No {}-{} bonds detected".format(atomtype_i, atomtype_j))
            return None

        print(bonded_pairs[0:10])
        # Compute distance between bonded pairs
        bond_distances = np.asarray(mdtraj.compute_distances(traj,bonded_pairs))
    
        fig,ax =  plt.subplots(1,1)
        # 51 bins, 50 probabilities
        all_probabilities, bins, patches = ax.hist(bond_distances.flatten(), 50, normed=1)
        ax.set_xlabel("Distance (nm)")
        ax.set_ylabel("Probability")
        plt.savefig("{}-{}_bond_distribution.jpg".format(atomtype_i, atomtype_j))
        plt.close()
    
    
        # Need to compute energies from the probabilities
        # For each probability, compute energy and assign it appropriately
        all_energies = []
        all_distances = []
    
        for index, probability in enumerate(all_probabilities):
            first_bin = bins[index]
            second_bin = bins[index+1]
            bin_width = second_bin - first_bin
            if probability - 1e-6 <=0:
                probability = 1e-6
            energy = -self._k_b * self._T * np.log(probability)
            distance = np.mean((bins[index], bins[index+1]))
            all_energies.append(energy)
            all_distances.append(distance)
        # Shift energies to positive numbers
        min_shift = min(all_energies)
        all_energies = [energy - min_shift for energy in all_energies]
        min_index = np.argmin(all_energies)
        converged = False
        i = 2
        while not converged:
            try:
                # Slice data to be center the fit around the minima
                sliced_distances = all_distances[min_index-i: min_index+i]
                sliced_energies = all_energies[min_index-i: min_index+i]
                sliced_probabilities = all_probabilities[min_index-i: min_index+i]

                #bonded_parameters = self.fit_to_gaussian(sliced_distances, sliced_energies)
                bonded_parameters = self.fit_to_gaussian(sliced_distances, sliced_probabilities)
                converged=True

            except RuntimeError:
                i +=1
                if min_index + i >= 50 or min_index -i <= 0:
                    #bonded_parameters = self.fit_to_gaussian(all_distances, all_energies)
                    bonded_parameters = self.fit_to_gaussian(all_distances, all_probabilities)
                    converged=True
            

        predicted_energies = self.harmonic_energy(all_distances, **bonded_parameters)
        fig, axarray = plt.subplots(2,1,sharex=True)
        axarray[0].plot(all_distances, predicted_energies, c='darkgray', label="Predicted")
        axarray[1].plot(all_distances, all_energies, c='black', label="Target")
        axarray[0].legend()
        axarray[1].legend()
        axarray[1].set_xlabel("Distance (nm)")
        axarray[0].set_ylabel("Energy (kJ/mol)")
        axarray[1].set_ylabel("Energy (kJ/mol)")
        plt.savefig("{}-{}_bond_energies.jpg".format(atomtype_i, atomtype_j))
        plt.close()
        return bonded_parameters
    
    
    
    
    def compute_angle_parameters(self, traj, atomtype_i, atomtype_j, atomtype_k):
        """
        Calculate angle parameters from a trajectory
        Compute the probability distribution of angles
        Curve fit to gaussian distribution
        
        Parameters
        ---------
        traj : mdtraj Trajectory
        atomtype_i : str
            First atomtype 
        atomtype_j : str
            Second atomtype
        atomtype_k : str
            Third atomtype
    
        Returns
        -------
        force_constant : float
            Harmonic bond constant
        x0 : float 
          Bond reference
    
        Notes
        -----
        Considers both angles i-j-k and k-j-i
    
        """
    
        topol = traj.topology
        #target_triplet = set((atomtype_i, atomtype_j, atomtype_k))
        target_triplet = [atomtype_i, atomtype_j, atomtype_k]
        target_counter = Counter(target_triplet)
        all_triplets = []
        participating_bonds = []
        if len([(i,j) for i,j in topol.bonds]) == 0:
            sys.exit("No bonds detected, check your input files")
    
        # Iterate through topology bonds
        # Find all participating bonds that could fit in the triplet
        # If (1,2) and (2,4) then (1,2,4) is a triplet
        for (i, j) in topol.bonds:
            # If i.name and j.name aren't in the target_triplet, ignore it
            pair = Counter([i.name,j.name])
            diff = target_counter - pair
            if len([i for i in diff.elements()]) == 1 :
                participating_bonds.append([i,j])



        # Iterate through all combinations of participating bonds
        # If they share the same atom (atomtype_j), then this is a hit
        # The set of the 2 pairs must also have length 3 (one atom in common)
        # The names must all match up to the target triplet names
        for pair1, pair2 in itertools.combinations(participating_bonds,2):
            triplet_set = set((*pair1, *pair2))
            temp_triplet = [atomtype_i, atomtype_j, atomtype_k]
            all_pair_names = [i.name for i in pair1] + [j.name for j in pair2]
            for i in all_pair_names:
                try:
                    temp_triplet.remove(i)
                except ValueError:
                    pass
            sym_diff = sorted([a.index for a in set((pair1)).symmetric_difference((pair2))])
            intersection = [a for a in set((pair1)).intersection(set((pair2)))]
            if len(triplet_set) == 3 and len(intersection) == 1 and atomtype_j in intersection[0].name and len(temp_triplet)==0:
                all_triplets.append([sym_diff[0], intersection[0].index,
                    sym_diff[1]])
    
        if len(all_triplets) == 0:
            return None
        print(all_triplets[0:10])
    
    
        # Compute angle between triplets
        all_angles_rad = np.asarray(mdtraj.compute_angles(traj, all_triplets))
    
    
        fig,ax =  plt.subplots(1,1)
        # 51 bins, 50 probabilities
        vals, bins, patches = ax.hist([value for value in all_angles_rad.flatten() if not math.isnan(value)], 50, normed=1)
        ax.set_xlabel("Angle (rad)")
        ax.set_ylabel("Probability")
        plt.savefig("{}-{}-{}_angle_distribution.jpg".format(atomtype_i, atomtype_j, 
            atomtype_k))
        plt.close()
    
    
        # Need to compute energies from the probabilities
        # For each probability, compute energy and assign it appropriately
        all_energies = []
        all_angles = []
        all_probabilities = []
    
        for index, probability in enumerate(vals):
            first_bin = bins[index]
            second_bin = bins[index+1]
            bin_width = second_bin - first_bin
            if probability - 1e-6 <=0:
                probability = 1e-6
            angle = np.mean((bins[index], bins[index+1]))
            scaled_probability = probability/np.sin(angle)
            energy = -self._k_b * self._T * np.log(scaled_probability)
            all_energies.append(energy)
            all_angles.append(angle)
            all_probabilities.append(scaled_probability)

        fig, ax = plt.subplots(1,1)
        ax.plot(all_angles, all_probabilities, label="Scaled probabilities")
        ax.set_xlabel("Angle (rad)")
        ax.set_ylabel("Probability")
        ax.legend()
        plt.savefig("{}-{}-{}-scaled_probabilities.jpg".format(atomtype_i, atomtype_j, atomtype_k))
        plt.close()

        # Shift energies to positive numbers
        min_shift = min(all_energies)
        all_energies = [energy - min_shift for energy in all_energies]
        min_index = np.argmin(all_energies)
        
        converged = False
        i = 2
        while not converged:
            try:
                #bonded_parameters = self.fit_to_gaussian(all_angles[min_index-i:min_index+i], all_energies[min_index-i:min_index+i])
                bonded_parameters = self.fit_to_gaussian(all_angles[min_index-i:min_index+i], all_probabilities[min_index-i:min_index+i])
                if bonded_parameters['force_constant'] > 0.01:
                    converged = True
                else:
                    converged = False
                    i += 1
            except RuntimeError:

                try:
                    bonded_parameters = self.fit_to_gaussian(all_angles[min_index-i:min_index+i], all_energies[min_index-i:min_index+i], energy_fit=True)
                    if bonded_parameters['force_constant'] > 0.01:
                        converged = True
                    else: 
                        converged = False
                        i += 1
                except RuntimeError:
                    i += 1
                    if min_index + i >= 50 or min_index -i <=0:
                        # Before fitting, may need to reflect energies about a particular angle
                        mirror_angles = np.zeros_like(all_angles)
                        mirror_energies = np.zeros_like(all_energies)
                        mirror_probabilities = np.zeros_like(all_probabilities)
                        for i, val in enumerate(all_angles):
                            mirror_energies[i] = all_energies[-i-1]
                            mirror_probabilities[i] = all_energies[-i-1]
                            mirror_angles[i] = 2*all_angles[-1] - all_angles[-i-1]
                    
                        all_angles.extend(mirror_angles)
                        all_energies.extend(mirror_energies)
                        all_probabilities.extend(mirror_probabilities)

                        #bonded_parameters = self.fit_to_gaussian(all_angles, all_energies)
                        try:
                            bonded_parameters = self.fit_to_gaussian(all_angles, all_probabilities)
                            converged=True
                        except RuntimeError:
                            bonded_parameters = self.fit_to_gaussian(all_angles, all_energies, energy_fit=True)
                        converged = True

        predicted_energies = self.harmonic_energy(all_angles, **bonded_parameters)
        fig, axarray = plt.subplots(2,1,sharex=True)
        axarray[0].plot(all_angles, predicted_energies, c='darkgray', label="Predicted")
        axarray[1].plot(all_angles, all_energies, c='black', label="Target")
        axarray[0].legend()
        axarray[1].legend()
        axarray[1].set_xlabel("Angle (rad)")
        axarray[0].set_ylabel("Energy (kJ/mol)")
        axarray[1].set_ylabel("Energy (kJ/mol)")
        plt.savefig("{}-{}-{}_angle_energies.jpg".format(atomtype_i, atomtype_j, atomtype_k))
        plt.close()

   
        return bonded_parameters
    
    
    
            
        
    def compute_rdf(self, traj, atomtype_i, atomtype_j, output):
        """
        Compute RDF between pair of atoms, save to text
    
        Parameters
        ---------
        traj : mdtraj Trajectory
        atomtype_i : str
            First atomtype 
        atomtype_j : str
            Second atomtype
        output : str
            Filename
    
            """
    
        pairs = traj.topology.select_pairs(selection1='name {}'.format(atomtype_i),
                selection2='name {}'.format(atomtype_j))
        (first, second) = mdtraj.compute_rdf(traj, pairs, [0, 2], bin_width=0.01 )
        #np.savetxt('{}-{}-{}.txt'.format(i, j, options.output), np.column_stack([first,second]))
        np.savetxt('{}.txt'.format(output), np.column_stack([first,second]))
    
