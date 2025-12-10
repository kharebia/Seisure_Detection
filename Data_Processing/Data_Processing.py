import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# ---- Configuration ----
home = os.getcwd()
# point to a directory where data is located
data_dir = r"C:/Users/khare/OneDrive/Documents/Memristive_Hardware/Data"
subject = "Dog_1"
types = ["interictal", "ictal"]

# ---- Create path variables ----
in_path = os.path.join(data_dir, subject) # input path
root_dir = os.path.dirname(data_dir) # parent directory
out_path = os.path.join(root_dir, "Heatmaps", subject) # output path
os.makedirs(out_path, exist_ok=True) # create an output folder if it doesn't exist'

# ================================
#        MAIN PROCESSING LOOP
# ================================
for type_ in types:
    #count the number of files to process
    num_heatmaps = 0
    items = os.listdir(in_path)
    for f in items:
             if os.path.isfile(os.path.join(in_path, f)): # check if it's a file
                 if f"_{type_}_" in f: # check if its the type we want
                    num_heatmaps += 1
    num_heatmaps = num_heatmaps // 10
    for s in range(1, num_heatmaps + 1):

        # Move into subject’s data folder

        os.chdir(in_path)

        # Create an empty 256×250 matrix
        data = np.zeros((256, 250))

        # ---- Load & combine 10 files ----
        for k in range(1, 11):

            file_index = k + s * 10 - 10
            file_name = f"{subject}_{type_}_segment_{file_index}"

            mat = sio.loadmat(file_name + ".mat")
            seg_data = mat["data"] # load a data segment

            # Fill rows 0..255 using 16-row blocks
            for row in range(1, 17):
                data[(row * 16 - 16):(row * 16),(k * 25 - 25):(k * 25)] = seg_data[:, (row * 25 - 25):(row * 25)]

        # Normalize data to [0, 1] before plotting - make sure the average value is 0.5
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        data_norm_shifted = data_norm - np.mean(data_norm) + 0.5
        data_norm_shifted = np.clip(data_norm_shifted, 0, 1)

        # ================================
        #           SAVE HEATMAP
        #   exactly 256×250 pixels
        # ================================
        os.chdir(out_path) # go to the output folder
        fig = plt.figure(figsize=(250/100, 256/100), dpi=100)

        ax = plt.axes([0, 0, 1, 1])  # full figure
        ax.imshow(data_norm, cmap='jet', interpolation='nearest')
        ax.axis('off')
        output_name = f"{subject}_{type_}_heatmap_{s}.jpg"
        plt.savefig(output_name, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
    print(f"{subject} {type_} heatmaps saved: {num_heatmaps}.")
os.chdir(home) # go home
print("All heatmaps saved as 256×250 pixel JPEG images.")
print("Output path:\n", out_path)

