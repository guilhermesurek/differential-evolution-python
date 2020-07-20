import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import openpyxl

class DEError(Exception):
    '''Base class for throw exception error in this module.'''
    pass

class InitializationError(DEError):
    '''Exception for initialization errors.'''
    pass

class DEManager():

    def __init__(self, run, de_class, kw_args, excel_results='results.xlsx'):
        self.__de_class = de_class
        self.__run = run
        self.__kw_args = kw_args
        self._df_res_columns = ['ID', 'Population_Size', 'Mutation_Factor', 'Crossover_Prob', 'Max_Iter', 'Max_FES', 'Min_Error', 'Min_Error_Length', 'Run',
                                 'Iterations', 'FES', 'Eval', 'XBest']
        self._df_res = pd.DataFrame(columns=self._df_res_columns)
        self._res_path = excel_results

    def run(self):
        final_results = []
        # For each configuration
        for i in range(len(self.__kw_args)):
            results = []
            # Run {self.run} times
            for j in range(self.__run):
                #Instanciate DE
                DE = eval(self.__de_class)(**self.__kw_args[i])
                DE.fit()                                # Runs DE instance
                run_results = DE.get_results()          # Get results
                results.append(run_results)             # Save results
                # Save result in df
                res_to_df = [i, self.__kw_args[i].get('population_size'), self.__kw_args[i].get('mutation_factor'), self.__kw_args[i].get('cross_prob'), self.__kw_args[i].get('max_it'), self.__kw_args[i].get('max_fes'),
                             self.__kw_args[i].get('min_error'), self.__kw_args[i].get('min_error_length'), j]
                res_to_df.extend(run_results[:-1])
                res_to_df.append('; '.join([str(x) for x in run_results[-1]]))
                res_to_df = pd.Series(res_to_df, index=self._df_res_columns)
                self._df_res = self._df_res.append(res_to_df, ignore_index=True)
            # Report Result
            try:
                msg = f"Population Size: {self.__kw_args[i]['population_size']} | Mutation Factor: {self.__kw_args[i]['mutation_factor']} | Crossover Prob: {self.__kw_args[i]['cross_prob']} | Max Iter.: {self.__kw_args[i]['max_it']}"
                print(msg)
            except:
                print("Fail to read keyword arguments.")
            # Getting final results
            rmin = np.min([res[2] for res in results])
            rmax = np.max([res[2] for res in results])
            rmean = np.mean([res[2] for res in results])
            rmedian = np.median([res[2] for res in results])
            rstd = np.std([res[2] for res in results])
            print(f"Best: {rmin} | Worst: {rmax} | Mean: {rmean} | Meadian: {rmedian} | Std: {rstd}")
            print(80*"-")
            fes_mean = np.mean([res[1] for res in results])
            fes_std = np.std([res[1] for res in results])
            # Append Final Results
            final_results.append([rmin, rmax, rmean, rmedian, rstd, fes_mean, fes_std])
        # Final Report
        print(f"  ID    |  Pop Size  |  Mutation Factor  |  Crossover Prob  |  Max Iter  |  Runs  ")
        for k in range(len(final_results)):
            print("  {0:03d}   |    {1:05d}   |      {2:6.3f}       |      {3:6.3f}      |    {4:05d}   |  {5:04d}  ".format(k, self.__kw_args[k]['population_size'], self.__kw_args[k]['mutation_factor'], self.__kw_args[k]['cross_prob'], self.__kw_args[k]['max_it'], self.__run))
        print(84*"-")
        print(f"  ID    |        Best        |       Worst      |        Mean      |      Meadian     |         Std     ")
        for k in range(len(final_results)):
            print("  {0:03d}   |  {1:16.6f}  | {2:16.6f} | {3:16.6f} | {4:16.6f} | {5:16.6f}".format(k, final_results[k][0], final_results[k][1], final_results[k][2], final_results[k][3], final_results[k][4]))
        print(84*"-")
        print(f"  ID    | Function Evaluations ")
        for k in range(len(final_results)):
            print("  {0:03d}   |  {1:9.0f}+{2:.0f}".format(k, final_results[k][5], final_results[k][6]))
        # Save df to excel
        self._df_res.to_excel(self._res_path)

class DifferentialEvolution():
    '''
        Differential Evolution is a stochastic population-based algorithm for continuous function optimization.
        INPUT
            - boundary: For Cartesian running, must be informed the boundaries of the problem
            - population_size: The size of the population to search
            - mutation_factor: The factor of matutation
            - cross_prob: The probability of crossover
            - max_it: The maximum number of iterations
            - max_fes: The maximum number of function evaluations or fitness evaluations
            - min_error: The minimum error for consecutives best points
            - verbose: Print iterantion information
    '''

    class Individual():
        ''' 
            An Individual is a single instance, an element of the population.
        '''

        def __init__(self, element):
            # A single element in cartesian is a point in the R^X space
            self.point = element
            self.fit = 0.0
        
        def individual_fitness(self):
            self.fit = float(self.__cartesian_eval(self.point))
            return self.fit
        
        def __cartesian_eval(self, point):
            return np.sin(np.sqrt(point[0] ** 2 + point[1] ** 2))
    
    # Initialize the class
    def __init__(self, boundary=None, population_size=100, mutation_factor=0.01, cross_prob=0.9, init_seed=None,
                 max_it=100, max_fes=None, min_error=None, min_error_length=None, plot=False, verbose=False):
        # Population properties
        self.__population_size = population_size            # Population size
        self.__population = []                              # Current population
        self.__children = []                                # Current children
        self.__fitness = {}                                 # Current fitness
        self.__id_sorted = []                               # ID of population sorted
        self.__progress = []                                # Track progress
        if init_seed != None and type(init_seed) != int:
            raise InitializationError(f"Initial seed must be an integer.")
        if init_seed == None:
            self.__init_rand = random.Random()         # Initial population random initialization
        else:
            self.__init_rand = random.Random(init_seed)         # Initial population random initialization
        # Mutation properties
        if mutation_factor > 2 or mutation_factor < 0:
            raise InitializationError(f"Mutation facotr must be between 0 and 2.")
        self.__mutation_factor = mutation_factor            # Mutation Facotr
        self.__mutation_rand = random.Random()              # Mutation random initialization
        # Crossover properties
        if cross_prob > 1 or cross_prob < 0:
            raise InitializationError(f"Crossover probability must be between 0 and 1.")
        self.__cross_prob = cross_prob                      # Crossover Probability
        self.__cross_rand = random.Random()                 # Crossover random initialization
        self.__number_parents = int((1-cross_prob)*population_size) # Number of Parents for mating
        # General properties
        self.__verbose = verbose                            # Print information
        self.__plot = plot                                  # Plot progress
        if max_it == None or max_it <= 0:
            raise InitializationError(f"Maximum number of iterations must be a integer number greater than 0.")
        self.__min_error = min_error                        # Minimum error for termination criteria
        if min_error != None and min_error_length == None:
            raise InitializationError(f"When min_error is set, you must set min_error_length too. Its apply the minimum error to the last min_error_length best evaluations.")
        self.__min_error_length = min_error_length          # Apply minimum error to the last {min_error_length} best evaluations
        if min_error_length != None:
            self.__error_list = [None] *  min_error_length  # Save the last {min_error_length} errors
        self.__max_it = max_it                              # Maximum iterations / generations
        self.__max_fes = max_fes                            # Maximum function evaluations / fitness evaluations
        self.__it = 0                                       # Current number of iterations / generations
        self.__fes = 0                                      # Current number of function evaluations / fitness evaluations
        self.__termination_criteria = False                 # Save termination criteria evaluation
        # Search space, for Cartesian boundaries must be informed, for Permutation a vector of points must be informed
        if boundary == None:
            raise InitializationError(f"Boundary for the search must be informed.")
        if type(boundary) != list:
            raise InitializationError(f"Boundary must be a list not a {type(boundary)}.")
        if len(boundary[0]) != 2:
            raise InitializationError(f"Boundary must be in the format x lines 2 columns (x, 2).")
        for bound in boundary:
            if len(bound) > 1:
                for xbound in bound:
                    if (type(xbound) != float and type(xbound) != int):
                        raise InitializationError(f"Boundaries must be int or float type.")
            else:
                if (type(bound) != float and type(bound) != int):
                    raise InitializationError(f"Boundaries must be in int or float.")
        self.__boundary = boundary                      # Boundaries for Cartesian type

    def __eval_min_error(self):
        '''
            Calculate the error between two consecutives evaluations e verify if the last {self.__min_error_length} erros are less than {self.__min_error}
        '''
        # Verify if there is at least two values to calculate error
        if len(self.__progress) > 1:
            # Save the last {self.__min_error_length} consecutives errors (transfer 3 to 4, 2 to 3, 1 to 2, 0 to 1)
            for i in range(self.__min_error_length-1):
                self.__error_list[self.__min_error_length-i-1] = self.__error_list[self.__min_error_length-i-2]
            # Calculate the last error value
            self.__error_list[0] = abs(self.__progress[-1]-self.__progress[-2])
            # If all the vector is filled, verify if the termination condition is satisfied
            if self.__error_list[-1] != None:
                # Verify if all last {self.__min_error_length} erros are less than {self.__min_error}
                if sum([error <= self.__min_error for error in self.__error_list]) == self.__min_error_length:
                    return True
        return False

    def __eval_termination_criteria(self):
        '''
            Check all tree termination criterias: Minimum error between consecutive evaluations; Maximum number of iterations; Maximum number of function evaluations.
        '''
        # Validate termination criteria
        # Validate minimum error criteria
        if self.__min_error != None:
            if self.__eval_min_error():
                if self.__verbose:
                    print(f"[Termination Criteria] Minimum Error Reached: All last {self.__min_error_length} consecutives best evaluation erros was less than {self.__min_error}.")
                self.__termination_criteria = True
        # Validate number of iterations
        if self.__it >= self.__max_it:
            if self.__verbose:
                print(f"[Termination Criteria] Maximum Iterations Reached: Reached {self.__max_it} iterations.")
            self.__termination_criteria = True
        # Validate number of function evaluations
        if self.__max_fes != None:
            if self.__fes >= self.__max_fes:
                if self.__verbose:
                    print(f"[Termination Criteria] Maximum Function Evaluations Reached: Reached {self.__fes} FES over a limit of {self.__max_fes}.")
                self.__termination_criteria = True                       

    def __update_fes(self):
        '''
            Update number of function evaluations.
        '''
        ##Update fes
        self.__fes = self.__fes + self.__population_size

    # A function to execute the mutation properly
    def __mutate(self, ind):
        ''' 
            Apply to mutation with the mutation probability.
        '''
        # Shuffle the population
        pool = self.__cross_rand.sample(self.__population, len(self.__population))
        new_ind = ind + np.array(self.__mutation_factor) * (np.array(pool[0]) - np.array(pool[1]))
        new_ind = np.array(new_ind >= np.array(self.__boundary).T[0]) * new_ind + (np.array(new_ind >= np.array(self.__boundary).T[0])-1)*-1 * pool[0]
        new_ind = np.array(new_ind <= np.array(self.__boundary).T[1]) * new_ind + (np.array(new_ind <= np.array(self.__boundary).T[1])-1)*-1 * pool[0]
        return list(new_ind)

    def __select_mutation(self):
        ''' 
            Loop over all population and apply mutation with the {self.__mutation_factor} probability.
        '''
        for i in range(0, self.__population_size):
            self.__children[i] = self.__mutate(self.__population[i])

    def __crossover(self, parent1, parent2):
        ''' 
            Apply to crossover with the two parents inputed. For Permutation, cross gene parts. For Cartesian, apply median.
        '''
        child = []
        ranEl = self.__cross_rand.randint(0, len(parent1))
        # For each dimension, calculate median of parent 1 and parent 2
        for i in range(len(parent1)):
            if self.__cross_rand.random()<= self.__cross_prob or i == ranEl:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    def __select_crossover(self):
        ''' 
            Select top {self.__number_parents} best parents and keep them. To all the rest apply
            crossover.
        '''
        children = []
        # Select pool
        pool = self.__children
        # Apply crossover to the pool. first x last, second x last but one, third x last but two and so on.
        for i in range(0, len(self.__children)):
            child = self.__crossover(pool[i], pool[len(self.__children)-i-1])
            children.append(child)
        self.__children = children

    # Sort population by evaluation
    def __select_population(self):
        ''' 
            Run the individual fitness function for each child in children. Select between parent (population) and child. Sort the result.
        '''
        # Loop over children, evaluate fitness of child and select parent/child, saving the fitness of each individual
        for i in range(0, self.__population_size):
            # Evaluate the child
            fit_chi = self.Individual(self.__children[i]).individual_fitness()
            try:
                # If child best perform parent, change
                if fit_chi <= self.__fitness[i]:
                    self.__population[i] = self.__children[i]
                    self.__fitness[i] = fit_chi
                # Else, keep things and try again later
            except KeyError:
                # population fitness not evaluated yet, first run
                self.__population[i] = self.__children[i]
                self.__fitness[i] = fit_chi
        # Update fes
        self.__update_fes()
        # Sort the fitness result and save (ID, Fitness)
        self.__id_sorted = sorted(self.__fitness.items(), key = operator.itemgetter(1))#, reverse = True)
        # Update iteration number
        self.__it = self.__it + 1
        # Update progress
        self.__progress.append(self.__id_sorted[0][1])
        # Evaluate Termination Criteria
        self.__eval_termination_criteria()
        # Verbose Info
        self.__verbose_progress()

    def __verbose_progress(self):
        '''
            Verbose information about optimization progress.
        '''
        # Check verbose
        if self.__verbose:
            msg = f"PopSize {self.__population_size} | Iter. {self.__it} of {self.__max_it}"
            if self.__max_fes != None:
                msg = msg + f" | FES {self.__fes} of {self.__max_fes}"
            else:
                msg = msg + f" | FES {self.__fes}"
            if self.__min_error != None:
                if len(self.__progress) > 1:
                    msg = msg + " | Error {0:e} of {1:e}".format(self.__error_list[0], self.__min_error)
            msg = msg + " | Value {0:.6f}".format(self.__progress[-1])
            print(msg)
        else:
            if self.__termination_criteria:
                msg = f"PopSize {self.__population_size} | Iter. {self.__it} of {self.__max_it}"
                if self.__max_fes != None:
                    msg = msg + f" | FES {self.__fes} of {self.__max_fes}"
                else:
                    msg = msg + f" | FES {self.__fes}"
                if self.__min_error != None:
                    if len(self.__progress) > 1:
                        msg = msg + " | Error {0:e} of {1:e}".format(self.__error_list[0], self.__min_error)
                msg = msg + " | Value {0:.6f}".format(self.__progress[-1])
                print(msg)
            #print(f"Population {self.__population}")

    # A function to initialize the population
    def __initialize_population(self):
        ''' 
            Initialize population. For Permutation run population_size permutation samples of the points without repetition.
            For Cartesian select population_size random points in the search space.
        '''
        for i in range(0, self.__population_size):
            aux = []
            for axis in self.__boundary:
                aux.append(self.__init_rand.uniform(axis[0],axis[1]))
            self.__population.append(np.array(aux))
        # Initialize fitness dict
        self.__fitness = {}
        # Initialize Children
        self.__children = self.__population
        # Sort population by evaluation
        self.__select_population()
    
    def __next_gen(self):
        '''
            Run Differential Evolution to get population next geration. Apply crossover, apply mutation and select results.
        '''
        # Select and apply crossover
        self.__select_crossover()
        # Apply mutation
        self.__select_mutation()
        # Resort population
        self.__select_population()

    def plot_progress(self):
        '''
            Plot progress.
        '''
        # Plot progress
        plt.plot(self.__progress)
        plt.ylabel('Value')
        plt.xlabel('Generation')
        plt.show()
        return self.__progress
    
    def get_results(self):
        '''
            Get algorithm results.
        '''
        # Get Results: Iterations | FES | Eval | XBest 
        return [self.__it, self.__fes, self.__progress[-1], self.__population[self.__id_sorted[0][0]]]

    def fit(self):
        '''
            Main function to run Differential Evolution
        '''
        # Initialize Population
        self.__initialize_population()
        
        while not self.__termination_criteria:
            # Run next generation
            self.__next_gen()
        
        # Plot Progress
        if self.__plot:
            self.plot_progress()
