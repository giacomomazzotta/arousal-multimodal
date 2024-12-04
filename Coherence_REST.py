# EEGConnectivity
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.signal import coherence
import mne

# Load EEG data
# We'll need to perform a for loop to analyse all of the subjects' sessions or work on a supersubject
file_path = "/Users/giacomomazzotta/Desktop/BRAINHACK/sub-01/ses-01/eeg/sub-01_ses-01_task-rest_eeg_epoched.set"  # Replace with your file
myData = mne.read_epochs(file_path)

# Default Mode Network (DMN) channels
dmn_channels = ["Fpz", "Fp1", "Fp2", "Pz", "P3", "P4", "P7", "P8", "T7", "T8", "FT7", "FT8"]

# Sampling frequency
fs = myData.info['sfreq']  

# Get the EEG data for DMN
eeg_dmn = {ch: myData.get_data(picks=ch)[0] for ch in dmn_channels}

# Get the EEG data for SN
# eeg_sn = {ch: EEG1.get_data(picks=ch)[0] for ch in sn_channels}

tmin = 0.001  # 200 ms before the event
tmax = 1   # 500 ms after the event

# Create epochs based on events
epochs = mne.Epochs(eeg_dmn, events, event_id=selected_event_id, tmin=tmin, tmax=tmax, preload=True)


# Function to compute and plot coherence 
def plot_intra_network_coherence(eeg_dmn, dmn):
    coherence_results = []  # Store coherence results
    for i, (ch1, data1) in enumerate(eeg_dmn.items()):
        for j, (ch2, data2) in enumerate(eeg_dmn.items()):
            if i < j:  # Only compute each pair once
                f, Cxy = coherence(data1, data2, fs=myData.info['sfreq'], nperseg=myData.info['sfreq']*2)
                coherence_results.append((f, Cxy, ch1, ch2))  # Store f, Cxy, and channel names
                plt.figure(figsize=(8, 4))
                plt.semilogy(f, Cxy)
                plt.title(f'Coherence in {dmn}: {ch1} - {ch2}')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Coherence')
                plt.grid(True)
                plt.show()
    return coherence_results

# Intra-network coherence for DMN
coherence_results = plot_intra_network_coherence(eeg_dmn, "Default Mode Network")

# Function to filter coherence by frequency band
def filter_coherence_by_band(f, Cxy, band=(8, 13)):
    band_filter = (f >= band[0]) & (f <= band[1])
    mean_band_coherence = Cxy[band_filter].mean()  # Mean coherence in the band
    return mean_band_coherence

# Initialize a DataFrame to hold the mean coherence values for each pair of channels
mean_coherence_matrix = pd.DataFrame(index=dmn_channels, columns=dmn_channels)

# Example of filtering for the alpha band (8-13 Hz) in coherence computation
for f, Cxy, ch1, ch2 in coherence_results:
    mean_alpha_coherence = filter_coherence_by_band(f, Cxy, band=(8, 13))
    mean_coherence_matrix.loc[ch1, ch2] = mean_alpha_coherence
    mean_coherence_matrix.loc[ch2, ch1] = mean_alpha_coherence  

    
# Convert the DataFrame to a NumPy array (optional but useful for visualization)
mean_coherence_array = mean_coherence_matrix.to_numpy()
    
# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(mean_coherence_matrix.astype(float), annot=True, cmap='coolwarm', cbar_kws={'label': 'Mean Coherence (8-13 Hz)'},
            xticklabels=dmn_channels, yticklabels=dmn_channels)
plt.title('Mean Coherence (Alpha Band: 8-13 Hz) for DMN Channels')
plt.show()


##Plotting coherence for band filtered

def plot_intra_network_coherence_with_band(eeg_dmn, dmn, band=(8, 13)):
    coherence_results = []
    
    for i, (ch1, data1) in enumerate(eeg_dmn.items()):
        for j, (ch2, data2) in enumerate(eeg_dmn.items()):
            if i < j:  # Only compute each pair once
                # Compute coherence
                f, Cxy = coherence(data1, data2, fs=myData.info['sfreq'], nperseg=myData.info['sfreq'] * 2)
                
                # Plot full coherence
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.semilogy(f, Cxy)
                plt.title(f'Coherence in {dmn}: {ch1} - {ch2}')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Coherence')
                plt.grid(True)
                
                # Filter coherence by band
                band_filter = (f >= band[0]) & (f <= band[1])
                f_band = f[band_filter]
                Cxy_band = Cxy[band_filter]
                
                # Plot band-limited coherence
                plt.subplot(1, 2, 2)
                plt.semilogy(f_band, Cxy_band, color='orange')
                plt.title(f'{band[0]}-{band[1]} Hz Coherence: {ch1} - {ch2}')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Coherence (Filtered)')
                plt.grid(True)
                
                # Append results for correlation matrix
                coherence_results.append({
                    'channel_1': ch1,
                    'channel_2': ch2,
                    'mean_coherence': Cxy_band.mean()
                })
                
                plt.tight_layout()
                plt.show()
    
    return coherence_results

# Call the function
coherence_results = plot_intra_network_coherence_with_band(eeg_dmn, "Default Mode Network", band=(8, 13))

    
    ########################## Da sistemare
    
# # Function to perform permutation test for coherence
# def permutation_test_coherence(data1, data2, fs, npermutations=1000):
  
#     # Calculate observed coherence
#     observed_f, observed_Cxy = coherence(data1, data2, fs=fs, nperseg=fs*2)

#     # Null distribution for coherence
#     null_distribution = []
#     for _ in range(npermutations):
#         # Shuffle the second signal
#         permuted_data2 = np.random.permutation(data2)
#         _, permuted_Cxy = coherence(data1, permuted_data2, fs=fs, nperseg=fs*2)
#         null_distribution.append(permuted_Cxy)

#     # Convert the null distribution to an array for comparison
#     null_distribution = np.array(null_distribution)

#     # Calculate p-values
#     p_values = (np.sum(null_distribution >= observed_Cxy, axis=0) + 1) / (npermutations + 1)

#     return observed_f, observed_Cxy, p_values

# # Example: Apply the test to two EEG signals
# fs = 250  # Replace with your EEG's sampling frequency

# observed_f, observed_Cxy, p_values = permutation_test_coherence(data1, data2, fs=fs, npermutations=1000)

# # Plot results
# plt.figure(figsize=(10, 5))
# plt.semilogy(observed_f, observed_Cxy, label="Observed Coherence", color="blue")
# plt.semilogy(observed_f, p_values, label="p-values", linestyle="--", color="orange")
# plt.axhline(0.05, color="red", linestyle="--", label="Significance Threshold (p=0.05)")
# plt.title("Coherence and Significance")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Coherence / p-values")
# plt.legend()
# plt.grid(True)
# plt.show()
