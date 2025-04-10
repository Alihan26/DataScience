import SimpleITK as sitk
import pandas as pd
import os
from pathlib import Path
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np

def registration_qc(image_paths, labels, output_path, lesion_msk_path, brain_mask_path=None):
    # Load images
    images = [nib.load(img_path).get_fdata() for img_path in image_paths]

    # Load brain mask if provided
    if brain_mask_path is not None:
        brain = nib.load(brain_mask_path).get_fdata()
    else:
        brain = 1.0 * (images[0] > 0.1)  # Use the second image as mask reference

    # Set background to NaN for transparency
    brain[brain == 0] = np.nan

    # Load lesion mask
    lesion_msk = nib.load(lesion_msk_path).get_fdata()

    # Determine the slice with the largest number of positive pixels for each view
    lesion_sums_axial = np.sum(lesion_msk > 0, axis=(0, 1))
    lesion_sums_sagittal = np.sum(lesion_msk > 0, axis=(1, 2))
    lesion_sums_coronal = np.sum(lesion_msk > 0, axis=(0, 2))

    best_slice_axial = np.argmax(lesion_sums_axial) if np.any(lesion_sums_axial > 0) else lesion_msk.shape[-1] // 2
    best_slice_sagittal = np.argmax(lesion_sums_sagittal) if np.any(lesion_sums_sagittal > 0) else lesion_msk.shape[0] // 2
    best_slice_coronal = np.argmax(lesion_sums_coronal) if np.any(lesion_sums_coronal > 0) else lesion_msk.shape[1] // 2

    lesion_msk[lesion_msk == 0] = np.nan
    num_images = len(images)
    num_views = 3  # Axial, sagittal, coronal
    plt.figure(figsize=(5 * num_images, 5 * num_views * 2), dpi=80, facecolor='black')  # Adjusted height for views
    plt.subplots_adjust(left=0.0001,
                        bottom=0.001,
                        right=0.9999,
                        top=0.98,
                        wspace=0.,
                        hspace=0.)

    # Plot each view
    for view_idx, (best_slice, axis_label) in enumerate(zip(
        [best_slice_axial, best_slice_sagittal, best_slice_coronal],
        ["Axial", "Sagittal", "Coronal"]
    )):
        for row in range(2):  # Two rows for each view
            for i, img in enumerate(images):
                plt.subplot(num_views * 2, num_images, (view_idx * 2 + row) * num_images + i + 1)
                img[np.isnan(brain)] = np.nan

                if axis_label == "Axial":
                    img_slice = img[:, :, best_slice]
                    lesion_slice = lesion_msk[:, :, best_slice]
                    brain_slice = brain[:, :, best_slice]
                elif axis_label == "Sagittal":
                    img_slice = img[best_slice, :, :]
                    lesion_slice = lesion_msk[best_slice, :, :]
                    brain_slice = brain[best_slice, :, :]
                elif axis_label == "Coronal":
                    img_slice = img[:, best_slice, :]
                    lesion_slice = lesion_msk[:, best_slice, :]
                    brain_slice = brain[:, best_slice, :]
        
                plt.imshow(np.rot90(img_slice), 'gray')
                if row == 1:  # Only add lesion overlay for the first row
                    plt.imshow(np.rot90(lesion_slice), 'hsv', interpolation='none', alpha=0.5)
                plt.axis('off')

                # Add titles only for the first row of axial images
                if view_idx == 0 and row == 0:
                    plt.title(labels[i], color='white', fontsize=14)

        # Add view label to the left side of the plot
        plt.text(-0.1, 0.5 - view_idx / 2, axis_label, color="white", fontsize=16,
                 rotation=90, transform=plt.gcf().transFigure)

    # Show and save the figure
    plt.savefig(output_path)




remove_ids = []
table = pd.read_table("/mnt/cat/jdeseo/SOOP/Data/participants.tsv")
table = table.dropna(subset="etiology")

ids = list(table["participant_id"])
existing_dirs = os.listdir("/mnt/cat/jdeseo/SOOP/Data/derivatives/lesion_masks")
cleaned_ids = ids
for id in ids:
    if id not in existing_dirs:
        cleaned_ids.remove(id)


for id in cleaned_ids:
    print(id)
    try:
        fixed_image = sitk.ReadImage('/mnt/cat/jdeseo/SOOP/Data/skullstriped_DWI/' + id + "_dwi_ss.nii.gz")
        moving_image = sitk.ReadImage('/mnt/cat/jdeseo/SOOP/Data/skullstriped_FLAIR/' + id + "_FLAIR_ss.nii.gz")
        save_dir = '/mnt/cat/jdeseo/SOOP/Data/Registered_FLAIR/' + id + '_FLAIR_reg.nii.gz'
        masks = os.listdir("/mnt/cat/jdeseo/SOOP/Data/derivatives/lesion_masks/" + id + "/dwi/")
        param_file = "/mnt/cat/jdeseo/SOOP/Data/Transform_MNI/TransformParameters.0.txt"

        elastix = sitk.ElastixImageFilter()
        elastix.SetFixedImage(fixed_image)
        elastix.SetMovingImage(moving_image)
        elastix.SetOutputDirectory(os.path.dirname("/mnt/cat/jdeseo/SOOP/Data/Transform/"))
        elastix.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
        elastix.LogToConsoleOff()
        elastix.LogToFileOff()
        
        elastix.Execute()
        reg_image = elastix.GetResultImage()
        sitk.WriteImage(reg_image, save_dir)

        fixed_image = sitk.ReadImage("/mnt/cat/jdeseo/SOOP/Data/caa_flair_in_mni_template_smooth_brain_intres.nii")
        moving_image = reg_image
        

        elastix = sitk.ElastixImageFilter()
        elastix.SetFixedImage(fixed_image)
        elastix.SetMovingImage(moving_image)
        elastix.SetOutputDirectory(os.path.dirname("/mnt/cat/jdeseo/SOOP/Data/Transform_MNI/"))
        elastix.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
        elastix.LogToConsoleOff()
        elastix.LogToFileOff()
        
        elastix.Execute()
        mni_flair = elastix.GetResultImage()

        #Save FLAIR MNI
        save_flair = "/mnt/cat/jdeseo/SOOP/Data/output_mni/" + id + "/flair/"
        Path(save_flair).mkdir(parents=True, exist_ok=True)     
        save_path = save_flair + id + "_FLAIR_MNI.nii.gz"
        sitk.WriteImage(mni_flair, save_path)

        #Execute Transformation DWI
        moving_image = sitk.ReadImage('/mnt/cat/jdeseo/SOOP/Data/skullstriped_DWI/' + id + "_dwi_ss.nii.gz")
        transform_param_map = sitk.ReadParameterFile(param_file)
        transformix = sitk.TransformixImageFilter()
        transformix.SetMovingImage(moving_image)
        transformix.SetTransformParameterMap(transform_param_map)
        transformix.LogToConsoleOff()
        transformix.LogToFileOff()
        transformix.Execute()
        mni_dwi = transformix.GetResultImage()

        save_dwi = "/mnt/cat/jdeseo/SOOP/Data/output_mni/" + id + "/dwi/"
        Path(save_dwi).mkdir(parents=True, exist_ok=True)     
        save_path = save_dwi + id + "_DWI_MNI.nii.gz"
        sitk.WriteImage(mni_dwi, save_path)

        #Execute Transformation lesion mask
        transform_param_map = sitk.ReadParameterFile(param_file)
        transform_param_map['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']
        transformix = sitk.TransformixImageFilter()
        for mask in masks:
            lesion_mask = sitk.ReadImage("/mnt/cat/jdeseo/SOOP/Data/derivatives/lesion_masks/" + id + "/dwi/" + mask)
            print(mask)
            transformix.SetMovingImage(lesion_mask)
            transformix.SetTransformParameterMap(transform_param_map)
            transformix.LogToConsoleOff()
            transformix.LogToFileOff()
        # Execute the transformation lesion mask
            transformix.Execute()
            mask_image = transformix.GetResultImage()
            save_mask = "/mnt/cat/jdeseo/SOOP/Data/output_mni/" + id + "/masks/"
            Path(save_mask).mkdir(parents=True, exist_ok=True)
            Path(save_dwi).mkdir(parents=True, exist_ok=True)
            save_path = save_mask + mask
            sitk.WriteImage(mask_image, save_path)

            registration_qc(image_paths=["/mnt/cat/jdeseo/SOOP/Data/output_mni/"  + id + "/flair/" + id + "_FLAIR_MNI.nii.gz"], 
                        labels=["FLAIR"],
                        output_path="/mnt/cat/jdeseo/SOOP/Data/QC_MNI/" + mask[:-7] + "_FLAIR.jpeg", 
                        lesion_msk_path="/mnt/cat/jdeseo/SOOP/Data/output_mni/" + id + "/masks/" + mask)
        
            registration_qc(image_paths=["/mnt/cat/jdeseo/SOOP/Data/output_mni/" + id + "/dwi/" + id +"_DWI_MNI.nii.gz"], 
                        labels=["DWI"],
                        output_path="/mnt/cat/jdeseo/SOOP/Data/QC_MNI/" + mask[:-7] + "_DWI.jpeg", 
                        lesion_msk_path="/mnt/cat/jdeseo/SOOP/Data/output_mni/" + id + "/masks/" + mask)
    except:
        remove_ids.append(id)

missing_data = {"missing_data" : remove_ids}
df = pd.DataFrame(missing_data)
df.to_csv("/mnt/cat/jdeseo/SOOP/Data/missing_data.csv")
    
    
  
