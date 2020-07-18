##How To setup your our DE
# Basically your need to overwrite the eval function, the function that calculates the value for the individual
class MyDE(DifferentialEvolution):
    class Individual(DifferentialEvolution.Individual):
        def __cartesian_eval(self, point):
            res = 0
            for i in range(len(point)):
                # Search Domain: [-4, 4]
                # Global Minimum: -78.332  x = [-2.9035]
                res = res + (point[i] ** 4 - 16 * point[i] ** 2 + 5 * point[i])
            return res/2

if __name__ == '__main__':
    # Prepare kw_args
    my_kw_args = {}
    my_kw_args[0] = {"boundary":[(-4,4),(-4,4)], "population_size":100, "mutation_factor":0.5, "cross_prob":0.8, "init_seed":None,
            "max_it":100, "min_error":1E-8, "min_error_length":10, "plot":True, "verbose":False}
    # Define runs
    my_runs = 5
    # Define your DE class
    my_class = 'MyDE'
    # Instanciate DEManager
    MyDEM = DEManager(run=my_runs, de_class=my_class, kw_args=my_kw_args)
    # Run DEManager
    print("DE Manager run\n" + 14*"-" + "\n")
    MyDEM.run()

    # Or just run a single run with the DE instance
    print("\nSignle run with the DE Instance\n" + 31*"-" + "\n")
    MyDEInst = MyDE(**my_kw_args[0])
    MyDEInst.fit()
    MyDEInst.get_results()
