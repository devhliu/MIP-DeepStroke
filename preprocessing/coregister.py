import nipype.interfaces.spm as spm

input = "/home/snarduzz/Data/33723/Neuro_Cerebrale_64Ch/diff_trace_tra_TRACEW_33723.nii"
output = "/home/snarduzz/Data/33723/Neuro_Cerebrale_64Ch/t2_tse_tra_33723.nii"

coreg = spm.Coregister()
coreg.inputs.target = output
coreg.inputs.source = input
coreg.run()