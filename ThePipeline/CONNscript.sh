#!/bin/bash
addpath('spm12')
addpath('conn')
load('/gpfs/uqe18wbu/ExperimentOne/conn_project01.mat')
conn_batch('Setup.done',true)
conn_batch('Denoising.done',true)
conn_batch('Analysis.name','RRC','Analysis.type','ROI-to-ROI','Analysis.weight', 'none','Analysis.done',true)
