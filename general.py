from time import time
import pandas as pd

class Chronometer:
    """
    Keep track of computation time (linear)
    """
    def __init__(self, name :str=''):
        self.track = []
        self.name = name
    def start(self, name :str=''):
        if name != '':
            self.name = name
        self.time = time()
    def end(self):
        self.track.append(time() - self.time)
        self.time = self.track[-1]
    def estimate(self): return sum(self.track)/len(self.track)
    def display(self):
        self.end()
        print("{} § {} min {} sec. <=> {} s".format(self.name, int(self.time//60), int(self.time%60), self.time))
    def display_str(self):
        self.end()
        return "{} § {} min {} sec. <=> {} s".format(self.name, int(self.time//60), int(self.time%60), self.time)
    def sleep(self, n :int):
        time.sleep(n)
    def clean(self):
        self.track = []
    def times_loop(self, iterator_, func):
        l = len(iterator_)
        for i in iterator_:
            self.start()
            res = func(i)
            self.end()
            l-=1
            estimated_time = self.estimate()*l
            print("{} § time left {} min {} sec. <=> {} s".format(self.name, int(estimated_time//60), int(estimated_time%60), estimated_time))
            yield res
            
def print_columns_in_cell(list_):
    """
    Explore lists in a glance on Jupyter Notebook
    """
    to_print = list_.copy()
    while to_print!=[]:
        line = ""
        size = 0
        while to_print!=[] and size+len(to_print[0])+4 < 116:
            l = to_print.pop(0)
            line += "'{}', ".format(l)
            size += len(l)+4
        print(line)

def logic_reduction(list_, operator='or'):
    """
    Reduce binary array lists as one array following chosen operator
    """    
    res = list_[0]
    if operator=='or':
        for to_sum in list_[1:]:
            res = res|to_sum
    else operator=='and':
        for to_sum in list_[1:]:
            res = res&to_sum
    return res

def weighted_mean(x, weights, center=None, norm=abs): 
    """
    Weighted mean with weighted variance option
    """    
    if center is None:
        return sum(x*weights)/weights.sum()
    else:
        return (weights*norm(x-center)).sum()/weights.sum()

def replace_in_string(string_, dict_):
    """
    Replace all string_'s identical components by repertoried strings in dict_
    """    
    for d in dict_:
        while d in string_:
            string_ = string_.replace(d, str(dict_[d]))
    return string_