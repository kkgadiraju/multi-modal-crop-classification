import matplotlib.pyplot as plt
import os, glob
import pandas as pd

data_folder = "../../0_data/crop-clf-ts-final"
classes = ['0', '1', '2', '3']
categories = ['train', 'val', 'test']
for tr_val_test in categories:
    for cls in classes:
        curr_folder = os.path.join(data_folder, tr_val_test, cls, "*.csv")
        curr_files = glob.glob(curr_folder)
        for curr_file_path in curr_files:
            out_file_name = os.path.basename(curr_file_path).split('.')[0]
            curr_ts = pd.read_csv(curr_file_path)['NDVI'].values
            plt.plot(curr_ts)
            plt.title('Class: {}, File: {}'.format(cls, out_file_name))
            plt.savefig('../../7_plots/{}/{}/{}.png'.format(tr_val_test, cls, out_file_name))
            plt.clf()  
