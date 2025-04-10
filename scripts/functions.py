import cc3d
import numpy as np
import nibabel
import os
import pandas as pd
from scipy import ndimage

def get_3D_volume(msk, voxel_size):
    return (np.sum(msk) * voxel_size)

def get_lesion_metrics(msk, comp, voxel_size):
    lesionwise_volumes = []
    lesion_volume = get_3D_volume(msk, voxel_size)
    for _, labeled_msk_cluster in cc3d.each(comp, binary=True, in_place=True):
        lesionwise_vol = get_3D_volume(labeled_msk_cluster, voxel_size)
        lesionwise_volumes.append(lesionwise_vol)

    return lesion_volume, lesionwise_volumes

def load_mri(img_dir):
    img = nibabel.load(img_dir)
    affine = img.affine
    img_data = img.get_fdata()
    vox_size = np.prod(img.header.get_zooms()) / 1000
    return img_data, affine, vox_size

def get_connected_components(mask,  pat_num, dir, affine, cutoff = 10, connectivity = 6, voxel_volume = 0.001):
    components, N = cc3d.connected_components(mask, connectivity=connectivity, return_N = True)
    stats = cc3d.statistics(components)
    path = os.path.join(dir, "masks"+ pat_num)
    l_volume, lwise_volume = get_lesion_metrics(mask, components, voxel_volume)
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(1, np.max(components)+1):
        mask = components * (components == i)/i
        vol = np.sum(mask)
        if vol > cutoff:
            out = nibabel.Nifti1Image(mask, affine)
            save_path = os.path.join(path, pat_num + "_lesion" + str(i) + ".nii.gz")
            nibabel.save(out, save_path)
    return N, l_volume, lwise_volume, stats

def territory_per_lesion_voxel(mni_mask, pat_num, territory_dict, territoy_map, voxel_volume, conn = 26):
    lesion_data = {"ID": [], "total volume": [], "lesion volume": [], "territory_com": [], "territory_voxels": []}
    total_volume = get_3D_volume(mni_mask, voxel_volume)
    components, N = cc3d.connected_components(mni_mask, connectivity = conn, return_N = True)
    for label, lesion in cc3d.each(components, binary = False, in_place = False):
        volume = get_3D_volume(lesion, voxel_volume)/np.max(lesion)
        lesion_data["ID"].append(pat_num)
        center_mass = ndimage.measurements.center_of_mass(lesion)
        vt = territoy_map[round(center_mass[0]), round(center_mass[1]), round(center_mass[2])]
        territory_com = territory_dict[int(vt)]["territory"] + territory_dict[int(vt)]["side"]
        vasc_map = register_vascular_territories(territoy_map, lesion)
        values, counts = np.unique(vasc_map, return_counts = True)
        affected_territories = dict(zip(values,counts))   
        del affected_territories[0.0]
        if bool(affected_territories):
            tv = max(affected_territories, key = lambda x: affected_territories[x])
            territory_voxels = territory_dict[int(tv)]["territory"] + territory_dict[int(tv)]["side"] 
        else:
            territory_voxels = "none"

        lesion_data["total volume"].append(total_volume)
        lesion_data["lesion volume"].append(volume)
        lesion_data["territory_com"].append(territory_com)
        lesion_data["territory_voxels"].append(territory_voxels)
    return lesion_data
    



def get_image_directory(img_dir, patient, sequence):
    if sequence == "flair":
        mri_dir = os.path.join(img_dir, "FLAIR_bc", patient + "_FLAIR_bc.nii.gz")
        mask_dir = os.path.join(img_dir, "lesion_maps_flair", patient, "flair", patient + "_space-TRACE_desc-lesionAcute_mask.nii.gz")
        return mri_dir, mask_dir
    mri_dir = os.path.join(img_dir, "DWI_bc", patient + "_DWI_bc.nii.gz")
    mask_dir = os.path.join(img_dir, "derivatives/lesion_masks", patient, "dwi", patient + "_space-TRACE_desc-lesionAcute_mask.nii.gz")
    return mri_dir, mask_dir

def get_mni_dir(img_dir, patient, sequence):
    mask = os.path.join(img_dir, "output_mni", patient, "masks", patient + "_space-TRACE_desc-lesionAcute_mask.nii.gz")
    return mask

def register_vascular_territories(vasc_mask, lesion_mask, affine=0, pat_num=0, path=0, save = False):
    vm = vasc_mask.copy()
    vm[lesion_mask == 0] = 0
    if save:
        out = nibabel.Nifti1Image(vasc_mask, affine)
        save_path = os.path.join(path, "pat" + pat_num + "vascular" +".nii")
        nibabel.save(out, save_path)
    return vm







