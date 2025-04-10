import os.path
import nibabel
import SimpleITK as sitk
import six
import pandas as pd
from functions import *
import numpy as np

#Define Directories
data = pd.read_table("/mnt/cat/jdeseo/SOOP/Data/participants.tsv")
data = data.dropna(subset=["etiology"])


#Remove IDs were registration did not work or lesion size too small
PAT_IDS = data["participant_id"]

SEQUENCE = "dwi"
IMG_DIR = "/mnt/cat/jdeseo/SOOP/Data"
MASKS_DIR = "/mnt/cat/jdeseo/SOOP/Data/masks_per_lesion_6_DWI/"
MNI_DIR = "/mnt/cat/jdeseo/SOOP/Data"
VASC_MASC_DIR = "/mnt/cat/jdeseo/segmentation_test/mni_vascular_territories.nii"
SAVE_PATH = "/mnt/cat/jdeseo/SOOP/Data/Features/"

#Dictonary for decoding vascular territory map
territories = {2: {"territory": "ACA", "side":"right"}, 4: {"territory": "MCA", "side":"right"},
               6: {"territory": "PCA", "side":"right"}, 8: {"territory": "Pons/Medulla", "side":"right"},
               10: {"territory": "Cerebellum", "side":"right"}, 12: {"territory": "ACA", "side":"left"}, 
               14: {"territory": "MCA", "side":"left"}, 16: {"territory": "PCA", "side":"left"},
               18: {"territory": "Pons/Medulla", "side":"left"}, 20: {"territory": "Cerebellum", "side":"left"},
               0: {"territory": "Background", "side":""}}

#Loading vascular territory atlas
vasc_mask, aff, _ = load_mri(VASC_MASC_DIR)

#Initialize dictionarries for data storage
data = {"ID": [], "Total Volume (summ)": [], "Total volume (direct)": [],"Number of lesions":[], "Main Territory": [], "Main Side": [], 
"ACAright" : [], "MCAright":[], "PCAright": [], "Pons/Medullaright": [], "Cerebellumright" : [], "ACAleft" : [], "MCAleft":[], 
"PCAleft": [], "Pons/Medullaleft":[], "Cerebellumleft": [], "Billateral": [], "Totalleft": [], "Totalright":[], "Portion largest lesion": [],
"MNI Volume":[], "Main Territory (lesions)": [], "ACAright (lesions)" : [], "MCAright (lesions)":[], "PCAright (lesions)": [], 
"Pons/Medullaright (lesions)": [], "Cerebellumright (lesions)" : [], "ACAleft (lesions)" : [], "MCAleft (lesions)":[], 
"PCAleft (lesions)": [], "Pons/Medullaleft (lesions)":[], "Cerebellumleft (lesions)": [], "ACAright (count)" : [], "MCAright (count)":[], 
"PCAright (count)": [], "Pons/Medullaright (count)": [], "Cerebellumright (count)" : [], "ACAleft (count)" : [], "MCAleft (count)":[], 
"PCAleft (count)": [], "Pons/Medullaleft (count)":[], "Cerebellumleft (count)": []}

lesion_data = {"ID": [], "territory_com": [], "territory_voxels": [], "ACAright" : [], "MCAright":[], "PCAright": [], 
         "Pons/Medullaright": [], "Cerebellumright" : [], "ACAleft" : [], "MCAleft":[], "PCAleft": [], "Pons/Medullaleft":[], 
         "Cerebellumleft": []}

territory_none = []
no_mask = []
no_lesion = []
#Cycle through patients
for pat in PAT_IDS:   
    #Load relevant images
    print(pat)
    img_dir, mask_d = get_image_directory(IMG_DIR, pat, SEQUENCE)
    try:
        img, affine, or_voxel_volume = load_mri(mask_d)
    except FileNotFoundError:
        no_mask.append(pat)
        continue
    mni_mask_dir = get_mni_dir(MNI_DIR, pat, SEQUENCE)
    mni_lesion_mask, aff, voxel_volume = load_mri(mni_mask_dir)
    
    #Get territory labeling for lesion mask
    vasc_territories = register_vascular_territories(vasc_mask, mni_lesion_mask, aff, pat, MASKS_DIR) 
    
    #Get number of affected voxels per territory
    values, counts = np.unique(vasc_territories, return_counts = True)
    affected_territories = dict(zip(values,counts))   
    vol = get_3D_volume(mni_lesion_mask, voxel_volume)
    del affected_territories[0.0]
    try:
        mt = max(affected_territories, key = lambda x: affected_territories[x])
    except ValueError:
        no_lesion.append(pat)
        continue
    main_territory = territories[int(mt)]["territory"] + territories[int(mt)]["side"]
    data["ID"].append(pat)
    data["MNI Volume"].append(vol)
    #initialize counters
    total = 0
    total_left = 0
    total_right = 0
    
    #Cycle through vascular territories
    for n in range(2,22,2):
        fn = float(n)
        ter = territories[n]["territory"] + territories[n]["side"]
        try:
            volume = affected_territories[fn] * voxel_volume
            data[ter].append(volume)
            total += volume
            if n > 10:
                total_left += volume
            if n <= 10:
                total_right += volume
        except KeyError:
            data[ter].append(0)
    
    #Add new data to dictionary
    data["Total Volume (summ)"].append(total)
    data["Main Territory"].append(main_territory)
    data["Totalleft"].append(total_left)
    data["Totalright"].append(total_right)
    if total_left > 0 and total_right > 0:
        data["Billateral"].append("yes")
    else:
        data["Billateral"].append("no")
    if total_left > total_right:
        data["Main Side"].append("left")
    elif total_right > total_left:  
        data["Main Side"].append("right")
    
    #Define vascular territory per lesion
    data_lesions = territory_per_lesion_voxel(mni_lesion_mask, pat, territories, vasc_mask, conn = 6, voxel_volume = voxel_volume)
    #Calculate portion of largest lesion of total volume (native space)
    N, lv, lwisev, stats = get_connected_components(img, pat, MASKS_DIR, affine, cutoff = 10, connectivity = 6, 
                                 voxel_volume = or_voxel_volume)
    data["Total volume (direct)"].append(lv)
    data["Number of lesions"].append(N)
    portion_largest = np.amax(lwisev)/lv
    data["Portion largest lesion"].append(portion_largest)
    
    lesion_volume_per_territory = {"ACAright" : 0, "MCAright":0, "PCAright": 0, 
         "Pons/Medullaright": 0, "Cerebellumright" : 0, "ACAleft" : 0, "MCAleft":0, "PCAleft": 0, "Pons/Medullaleft":0, 
         "Cerebellumleft": 0}
    lesion_count_per_territory = {"ACAright" : 0, "MCAright":0, "PCAright": 0, 
         "Pons/Medullaright": 0, "Cerebellumright" : 0, "ACAleft" : 0, "MCAleft":0, "PCAleft": 0, "Pons/Medullaleft":0, 
         "Cerebellumleft": 0}
    for i, lesion in enumerate(data_lesions["territory_voxels"]):
        if lesion == "none":
            territory_none.append(pat)
        else:
            lesion_volume_per_territory[lesion] += (data_lesions["lesion volume"][i])
            lesion_count_per_territory[lesion] += 1
    main_territory_lesions = max(lesion_volume_per_territory, key = lambda x: lesion_volume_per_territory[x])
    data["Main Territory (lesions)"].append(main_territory_lesions)
    for terr in lesion_volume_per_territory:
        data[terr + " (lesions)"].append(lesion_volume_per_territory[terr])
        data[terr + " (count)"].append(lesion_count_per_territory[terr])                                       
    df_lesions = pd.DataFrame(data_lesions)
    df_lesions.to_csv(SAVE_PATH + "PerLesion/" + "PerLesionParameters" + pat + ".csv", index=False)

#Write dataframe and save data
df = pd.DataFrame(data)
df["Total_right (lesion)"] = df["ACAright (lesions)"] + df["MCAright (lesions)"] + df["PCAright (lesions)"] + df["Cerebellumright (lesions)"] + df["Pons/Medullaright (lesions)"]
df["Total_left (lesion)"] = df["ACAleft (lesions)"] + df["MCAleft (lesions)"] + df["PCAleft (lesions)"] + df["Cerebellumleft (lesions)"] + df["Pons/Medullaleft (lesions)"]
df["ratio_sides"] = abs((df["Total_right (lesion)"] - df["Total_left (lesion)"])/(df["Total_right (lesion)"] + df["Total_left (lesion)"]))
df['Billateral (lesions)'] = ['yes' if (a < 0.9) else 'no' for a in  df["ratio_sides"]]

df["Total_anterior (lesion)"] = df["ACAright (lesions)"] + df["MCAright (lesions)"] + df["ACAleft (lesions)"] + df["MCAleft (lesions)"]
df["Total_posterior (lesion)"] =  df["PCAleft (lesions)"] + df["Cerebellumleft (lesions)"] + df["Pons/Medullaleft (lesions)"] + df["PCAright (lesions)"] + df["Cerebellumright (lesions)"] + df["Pons/Medullaright (lesions)"]
df["ratio_ant_post"] = abs((df["Total_anterior (lesion)"] - df["Total_posterior (lesion)"])/(df["Total_anterior (lesion)"] + df["Total_posterior (lesion)"]))
df['ant_and_post_lesions'] = ['yes' if (a < 0.9) else 'no' for a in  df["ratio_ant_post"]]
df.to_csv(SAVE_PATH + "PerPatient/" + "PerPatientParameters.csv", index=False)

missing_masks = pd.DataFrame(no_mask)
missing_masks.to_csv(SAVE_PATH + "no_masks.csv", index=False)

