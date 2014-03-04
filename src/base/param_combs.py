



import collections as coll


def BuildCombinations(param_vals_ll):

    if len(param_vals_ll) == 0:
        return [[]]
    else:
        for i, param_vals_l in enumerate(param_vals_ll):
            param_combs = list()
            
            for param_val in param_vals_l:
                pvals_l_residual = BuildCombinations(param_vals_ll[i+1:])
                param_combs.extend( [ [param_val] + pval for pval in pvals_l_residual ] )
            
            return param_combs


class ParamGridIter(object):

	def __init__(self, params_vals_d):

		self.params_lst = params_vals_d.keys()
		self.params_vals_combs = BuildCombinations( params_vals_d.values() )
		self.combs_cnt = -1


	def __iter__(self):

		return self


	def next(self):
		
		if self.combs_cnt == len(self.params_vals_combs):
			raise StopIteration
		
		self.combs_cnt += 1
	
	
		params_comb_ordered_dict = coll.OrderedDict( zip(self.params_lst, self.params_vals_combs[self.combs_cnt]) )

	

		return params_comb_ordered_dict

	





