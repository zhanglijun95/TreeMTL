from statistics import mean
import numpy as np

class Layout():
    # Main Structure:
    ## Store the current state, the number of cuts applied, and the lowest available cut position
    ## Task metrics are calculated in metric_inference function from known results
    
    def __init__(self, T, B, S):
        super().__init__()
        self.T = T
        self.B = B
        self.state = S
        if not self.check_valid():
            print("T, B, and S are not compatiable!")
            return
        
        self.num_cut = 0
        self.lowest_avail_cut = 0
        
        # new to fast-MTL
        self.conv_iter = [] # list of tuple, tuple for range
        self.virtual_loss = [] # list of float, float for loss
        
        # new to IndFeatCorr
        self.prob = 0
        
        # Should be updated in the metric_inference function
        self.metric_list = [0.] * self.T
        self.score = 0.
        self.flops = 0.
        
    def check_valid(self):
        valid = True
        if len(self.state) != self.B:
            valid = False
            
        tasks = set()
        for task_set in self.state[0]:
            tasks.update(task_set)
        if len(tasks) != self.T:
            valid = False
        return valid
    
    def set_score(self):
        self.score = mean(self.metric_list)
        
    def set_score_weighted(self, weights):
        if len(weights) != (self.T) or sum(weights)!=1:
            print('Wrong weights!', flush=True)
            return
        self.score = np.dot(np.array(self.metric_list),np.array(weights))
    
    def __str__(self):
        return str(self.state)
    def __repr__(self):
        return str(self.state)
    # For redundancy
    def __eq__(self, other):
        equal = True
        for b in range(self.B):
            task_set_list1 = self.state[b]
            task_set_list2 = other.state[b]
            for task_set in task_set_list1:
                if task_set not in task_set_list2:
                    equal = False
                    break
            for task_set in task_set_list2:
                if task_set not in task_set_list1:
                    equal = False
                    break
            if not equal:
                break
        return equal
    def __getitem__(self, index):
        return self.state[index]
    # For sort
    def __lt__(self, other):
        return self.score < other.score