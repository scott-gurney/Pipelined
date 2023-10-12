addpath('spm12')
addpath('conn')

load('/gpfs/scratch/uce19kqu/ADNI_AD/REAL.mat')


conn_batch('Setup.done',true)
conn_batch('Denoising.done',true)
conn_batch('Analysis.name','RRC','Analysis.type','ROI-to-ROI','Analysis.weight', 'none','Analysis.done',true)