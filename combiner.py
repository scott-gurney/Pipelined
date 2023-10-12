import numpy as np
import nibabel as ni

def merge_nii_files (sfile, ns):
    # This will load the first image for header information
    img = ni.load(sfile % (5, ns[0]))
    dshape = list(img.shape)
    dshape.append(len(ns))
    data = np.empty(dshape, dtype=img.get_data_dtype())

    header = img.header
    equal_header_test = True

    # Now load all the rest of the images
    for n, i in enumerate(ns):
        img = ni.load(sfile % (5,i))
        equal_header_test = equal_header_test and img.header == header
        data[...,n] = np.array(img.dataobj)

    imgs = ni.Nifti1Image(data, img.affine, header=header)
    if not equal_header_test:
        print("WARNING: Not all headers were equal!")
    return(imgs)

nii_files = "C:\\Users\\User\Downloads\\20230727\REST-fMRI\\20230727_REST-fMRI_fUWWBIC23AMMMDMDMM030MED-0006-%0*d.nii"
images = merge_nii_files(nii_files, range(1,354))

ni.save(images, '20230727_REST-fMRI_fUWWBIC23AMMMDMDMM030MED-0006-00001-01.nii')