import numpy as np
import pandas as pd

# Load data sub 01
folder = 'data/sub-01/ses-01/eeg/sub-01_ses-01_task-rest_recording-eyetracking_physio.tsv/'
file = 'sub-01_ses-01_task-rest_recording-eyetracking_physio.tsv'
data = pd.read_csv(folder+file,  sep='\t', header=None)
columns = ['time', 'gazeH', 'gazeV', 'pupilsize', 'resX', 'resY', 'fixation', 'saccade', 'blink', 'tasktrigger', 'timetrigger', 'fmritrigger', 'interpolsamples']
data = data.rename({c: col for c, col in enumerate(columns)}, axis=1)

# Downsampling
data_ds = data.loc[data['fmritrigger'] == 1]
pupil_ds = data_ds.pupilsize.to_numpy()

# Save as 1D
np.savetxt(folder+'pupilds.1D', pupil_ds)