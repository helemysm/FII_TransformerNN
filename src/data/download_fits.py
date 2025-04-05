import subprocess
import os
import numpy as np
import glob as glob
import pickle
import subprocess
from tqdm import tqdm
import pandas as pd
import json

import re


path_origin ='/user/yhelem/storage/exo_images/dataset_spoc/dataset_url/'
path_dest = '/user/yhelem/storage/exo_images/sparse_attention_model/sparse_model_timeseries/dataset_fits/'
output_prefix = '/user/yhelem/storage/exo_images/dataset_fits_candidate_exofop/'

df = pd.read_csv('/user/yhelem/storage/exo_images/TOIS.csv')

df = df[df['TFOPWG Disposition'].isin(['CP', 'KP'])].copy() 

def get_sectors(sectors_str):
    row_sectors = np.array(sectors_str.split(','), dtype=int)
    return row_sectors[row_sectors < 27]  


df['Filtered Sectors'] = df['Sectors'].apply(get_sectors)

tic_ids_with_sectors = df[['TIC ID', 'Filtered Sectors']]

sh_files = np.sort(glob.glob("/user/yhelem/storage/exo_images/spoc_sh_01_26/*.sh"))

found_tic_ids = []

for sh_file in sh_files:
    with open(sh_file, 'r') as file:
        lines = file.readlines()
        
        sector = int(sh_file.split('_')[8].split('s')[1])#int(re.search(r's00(\d)', sh_file).group(1))  
        
        print(sector)
        for line in lines:
            match = re.search(r'hlsp_tess-spoc_tess_phot_0*([1-9][0-9]*)', line)
            
            if match:
                tic_id = match.group(1) 
               
                
                if tic_id in tic_ids_with_sectors['TIC ID'].astype(str).values:
                    
                    print('TIC ID found ', tic_id, tic_ids_with_sectors.loc[tic_ids_with_sectors['TIC ID']==int(tic_id)]['Filtered Sectors'].values[0])
                        
                    sectors_tic = tic_ids_with_sectors.loc[tic_ids_with_sectors['TIC ID']==int(tic_id)]['Filtered Sectors'].values[0]
                    print('==========>>>>>>>> sectors found : ', sectors_tic)
                    print('==========>>>>>>>> sector for check : ', sector)
                    
                    if sector in sectors_tic:
                        
                        command = line
                        file_name_tic = str(tic_id)+'_'+str(sector)+'_pc.fits'

                        new_output_path = os.path.join(output_prefix, str(tic_id), file_name_tic)

                        print(new_output_path)

                        command_parts = command.split("--output")
                        modified_command = f"{command_parts[0]} --output '{new_output_path}' {command_parts[1].split(' ')[2]}"
                        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                        print(modified_command)
                        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                        # Execute 
                        process = subprocess.Popen(modified_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        stdout, stderr = process.communicate()

                        found_tic_ids.append(tic_id)

print("TIC IDs encontrados:", found_tic_ids)

