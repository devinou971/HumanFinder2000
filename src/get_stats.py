import os 
import pandas as pd

results_dir = 'runs/detect/fine_tuning_dataset4'

output_file = "results_"+ results_dir.split("/")[-1] + '.xlsx'

results = []

for subfolder in os.listdir(results_dir):
    print(subfolder)
    if "v7" in subfolder:
        continue
    csv_path = os.path.join(results_dir, subfolder, 'results.csv')
    ouputs = os.listdir(os.path.join(results_dir, subfolder, 'weights'))
    best_row = 1
    for ouput in ouputs:
        if ouput.startswith("best_"):
            best_row = int(ouput.split("_")[1].split(".")[0])

    if os.path.isfile(csv_path):

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.replace(' ', '')
        # best_row = df.loc[df['epoch'].argmax()]
        best_row = df.loc[df['epoch'] == best_row]

        name = subfolder.split("_")[0] + "_" + "_".join(subfolder.split("_")[2:-1]) 
        best_row_dict = {"model": name}
        for column, value in best_row.iloc[0].items():
            best_row_dict[column] = value
        results.append(best_row_dict)

df = pd.DataFrame(results) 
df.to_excel(output_file, index=False)

print(f"Summary written to {output_file}")