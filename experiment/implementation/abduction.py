from pysat.examples.hitman import Hitman
import pulp as plp
import random
from timeit import default_timer as timer

class MultilayerRELUEXplanation:
    def __init__(self, weight_bias, y_max, problem_name="multilayer_relu_explanation"):
        self.K = len(weight_bias)//2
        self.OUTPUT_SIZE = weight_bias[-1].shape[0]
        self.INPUT_SIZE = weight_bias[0].shape[0]
        self.X = {(i, 0): plp.LpVariable(cat=plp.LpContinuous, name="x_%d_%d" % (i, 0)) for i in range(self.INPUT_SIZE)}
        self.S = {}
        self.Z = {}
        self.rule_3 = {}
        self.rule_4 = {}
        self.rule_5 = {}
        for k in range(self.K):
            current_weight = weight_bias[2*k]
            current_bias = weight_bias[2*k + 1]
            M, N = current_weight.shape

            self.X.update({(n, k+1): plp.LpVariable(cat=plp.LpContinuous, name="x_%d_%d" % (n, k+1), lowBound=0) for n in range(N)})
            self.S.update({(n, k): plp.LpVariable(cat=plp.LpContinuous, name="s_%d_%d" % (n,k), lowBound=0) for n in range(N)})
            self.Z.update({(n, k): plp.LpVariable(cat=plp.LpBinary, name="z_%d_%d" % (n,k), lowBound=0) for n in range(N)})

            self.rule_3.update({(n,k): plp.lpSum(current_weight[m,n] * self.X[(m, k)] for m in range(M)) + current_bias[n] == self.X[(n,k+1)] - self.S[(n,k)] for n in range(N)})
            self.rule_4.update({(n,k): self.X[(n,k+1)] <= (1 - self.Z[(n,k)]) * y_max for n in range(N)})
            self.rule_5.update({(n,k) : self.S[(n,k)] <=  self.Z[(n,k)] * y_max for n in range(N)})
        # model
        self.base_model = plp.LpProblem(problem_name)
        for rule in (self.rule_3, self.rule_4, self.rule_5):
            for i in rule:
                self.base_model += rule[i]
        self.base_model.writeLP(problem_name+"-base_model.lp")
                
    def __repr__(self):
        return repr(self.base_model)
    
    def entail_util(self, assignment_x, not_e, i, j, debug=False):
        model = self.base_model.deepcopy()
        model += not_e
        for constraint in assignment_x:
            model += constraint
        c = model.solve()
        if debug:
            print("assigment:", assignment_x)
            print(model)
            for i in range(self.INPUT_SIZE):
                print("x%s=" % i, self.X[(i,0)].value())
            for i in range(self.OUTPUT_SIZE):
                print("y%s=" % i, self.X[(i, self.K)].value())
        return c == -1 #or self.X[(i,self.K-1)].value() == self.X[(j,self.K-1)].value()
    
    def entail(self, assignment_x, not_e_map, debug=False):
        for key in not_e_map:
            if self.entail_util(assignment_x, not_e_map.get(key), *key, debug):
                return True
        return False  
        
    def get_subset_minimal_with_randomized(self, x, prediction, count=1, debug=False):
        not_e_map = {(i,prediction):self.X[(i, self.K)] >= self.X[(prediction, self.K)] for i in range(self.OUTPUT_SIZE)  if i != prediction} # ~E
        assignment_x = {i:self.X[(i, 0)] == x[i] for i in range(self.INPUT_SIZE)}
        
        result = []
        for num in range(count):
            temp_list = list(assignment_x.keys())[:] 
            temp_dict = assignment_x.copy()
            random.shuffle(temp_list)
            for i in temp_list:
                temp = temp_dict[i]
                del temp_dict[i]

                if not self.entail(temp_dict.values(), not_e_map, debug):
                    temp_dict[i] = temp
            if len(result) == 0 or len(result) > len(temp_dict):
                result = temp_dict
                
        return result 
    
    def get_subset_minimal(self, x, prediction, debug=False):
        not_e_map = {(i,prediction):self.X[(i, self.K)] >= self.X[(prediction, self.K)] for i in range(self.OUTPUT_SIZE)  if i != prediction} # ~E
        assignment_x = {i:self.X[(i, 0)] == x[i] for i in range(self.INPUT_SIZE)}
        
        for i in list(assignment_x.keys())[:]:
            temp = assignment_x[i]
            del assignment_x[i]
           
            if not self.entail(assignment_x.values(), not_e_map, debug):
                assignment_x[i] = temp
                
        return assignment_x 
    
    def get_cardinality_minimal(self, x, prediction, debug=False, timeout=18):
        not_e_map = {(i,prediction):self.X[(i, self.K)] >= self.X[(prediction, self.K)] for i in range(self.OUTPUT_SIZE)  if i != prediction} # ~E
        assignment_x = {i:self.X[(i, 0)] == x[i] for i in range(self.INPUT_SIZE)}
        cube = set(assignment_x.keys())
        to_hit = []
        start = timer()
        while timer()  - start < timeout:
            with Hitman(bootstrap_with=to_hit, htype='sorted') as hitman:
                h = hitman.get()
                h = h if h else []
            
            if self.entail([assignment_x[key] for key in h], not_e_map, debug):
                return {key:assignment_x[key] for key in h}
            else:
                to_hit.append(cube-set(h))
            
        
        