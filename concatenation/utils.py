import pandas as pd

def write_measures(history, folder_name, filename):
    out_df = pd.DataFrame({'loss': history.history['loss'], 'acc': history.history['acc'], 'val_loss': history.history['val_loss'], 'val_acc': history.history['val_acc']})
    out_df.to_csv(os.path.join(folder_name,'{}.csv'.format(filename)))
