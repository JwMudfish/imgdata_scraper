import pandas as pd
import os

def file_count(save_path):
    dir_name = os.listdir(save_path)
    
    result = []
    for name in dir_name:
        files = len(os.listdir(f'{save_path}/{name}'))
        result.append([name, files])
    
    df = pd.DataFrame(result, columns = ['label', 'cnt'])
    return df

#print(file_count('./cls_seed_images'))

save_path = './cls_seed_images'
a = [[name, len(os.listdir(f'{save_path}/{name}'))] for name in os.listdir(save_path)]
df = pd.DataFrame(a, columns = ['label', 'cnt'])
print(df)