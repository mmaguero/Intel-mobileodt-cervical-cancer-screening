import pandas as pd
from math import log

sol = pd.read_csv('../data/solution_stg1_release.csv')
aprox = pd.read_csv('../submission/RF_20-06-2017_11-15.csv')

aprox['Type_1'] = aprox.apply(lambda x : max(min(x['Type_1'], 1 - 1e-15),1e-15), axis = 1)
aprox['Type_2'] = aprox.apply(lambda x : max(min(x['Type_2'], 1 - 1e-15),1e-15), axis = 1)
aprox['Type_3'] = aprox.apply(lambda x : max(min(x['Type_3'], 1 - 1e-15),1e-15), axis = 1)

merged = pd.merge(sol, aprox, on = 'image_name')
print(merged)

result = merged.apply(lambda x: x['Type_1_x']*log(x['Type_1_y']) + x['Type_2_x']*log(x['Type_2_y']) + x['Type_3_x']*log(x['Type_3_y']), axis = 1)

#print(result)
print(-sum(result)/512)
