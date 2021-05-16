# -*- coding: utf-8 -*-
"""
Created on Thu May 13 23:15:54 2021

@author: norah
"""
#參考網址：
#https://ithelp.ithome.com.tw/articles/10211706
#https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
import numpy as np
import random as rd
import time
import copy

class Location:
    def __init__(self,x):
        #self.name = name
        self.loc = x

    def distance_between(self, location2):
        assert isinstance(location2, Location)
        return X[self.loc-1][location2.loc-1]


def create_locations(i):
    locations =[]
    x = np.arange(1,i+1).tolist()
    for x in x:
        locations.append(Location(x))
    return locations


class Route:
    def __init__(self, path):
        # path is a list of Location obj
        self.path = path
        self.length = self._set_length()

    def _set_length(self):
        total_length = 0
        path_copy = self.path[:]
        from_here = path_copy.pop(0)
        init_node = copy.deepcopy(from_here)
        while path_copy:
            to_there = path_copy.pop(0)
            total_length += to_there.distance_between(from_here)
            from_here = copy.deepcopy(to_there)
        total_length += from_here.distance_between(init_node)
        return total_length


class GeneticAlgo:
    def __init__(self, locs, level=10, populations=100, variant=3, mutate_percent=0.1, elite_save_percent=0.1):
        self.locs = locs
        self.level = level
        self.variant = variant
        self.populations = populations
        self.mutates = int(populations * mutate_percent)
        self.elite = int(populations * elite_save_percent)

    def _find_path(self):
        # locs is a list containing all the Location obj
        locs_copy = self.locs[:]
        path = []
        while locs_copy:
            to_there = locs_copy.pop(locs_copy.index(rd.choice(locs_copy)))
            path.append(to_there)
            #print(path)
        return path

    def _init_routes(self):
        routes = []
        for _ in range(self.populations):
            path = self._find_path()
            routes.append(Route(path))
        return routes

    def _get_next_route(self, routes):
        routes.sort(key=lambda x: x.length, reverse=False)
        elites = routes[:self.elite][:]
        crossovers = self._crossover(elites)
        return crossovers[:] + elites

    def _crossover(self, elites):
        # Route is a class type
        normal_breeds = []
        mutate_ones = []
        for _ in range(self.populations - self.mutates):
            father, mother = rd.choices(elites[:4], k=2)
            index_start = rd.randrange(0, len(father.path) - self.variant - 1)
            # list of Location obj
            father_gene = father.path[index_start: index_start + self.variant]
            father_gene_names = [loc.loc for loc in father_gene]
            mother_gene = [gene for gene in mother.path if gene.loc not in father_gene_names]
            mother_gene_cut = rd.randrange(1, len(mother_gene))
            # create new route path
            next_route_path = mother_gene[:mother_gene_cut] + father_gene + mother_gene[mother_gene_cut:]
            next_route = Route(next_route_path)
            # add Route obj to normal_breeds
            normal_breeds.append(next_route)

            # for mutate purpose
            copy_father = copy.deepcopy(father)
            idx = range(len(copy_father.path))
            gene1, gene2 = rd.sample(idx, 2)
            copy_father.path[gene1], copy_father.path[gene2] = copy_father.path[gene2], copy_father.path[gene1]
            mutate_ones.append(copy_father)
        mutate_breeds = rd.choices(mutate_ones, k=self.mutates)
        return normal_breeds + mutate_breeds

    def evolution(self):
        routes = self._init_routes()
        for _ in range(self.level):
            routes = self._get_next_route(routes)
        routes.sort(key=lambda x: x.length)
        return routes[0].path, routes[0].length
    

input_size = 4
X = np.random.randint(1,31,size=(input_size,input_size))
X = np.triu(X,1)
X += X.T - np.diag(X.diagonal())
#X = [[0, 10, 15, 20], [10, 0, 35, 25],
#            [15, 35, 0, 30], [20, 25, 30, 0]]
t_ga = time.time()
my_locs= create_locations(input_size)
my_algo = GeneticAlgo(my_locs, level=400, populations=150, variant=2, mutate_percent=0.02, elite_save_percent=0.15)
best_route, best_route_length = my_algo.evolution()
runtime_ga = round(time.time() - t_ga, 3)
best_route.append(best_route[0])
print([loc.loc for loc in best_route], best_route_length)
#print("time:",runtime_ga)