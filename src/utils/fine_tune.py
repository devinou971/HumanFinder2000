import sys
from time import sleep
from ultralytics import YOLO
import os

def fine_tune_yolo(model_config, pre):
    # Load model
    all_runs = os.listdir("runs/detect")
    model = None
    for run in all_runs:
        if run.startswith(pre+"_dataset2_fine_tuned"):
            model = YOLO(model_config).load(os.path.join("runs/detect", run, "weights", "best.pt"))
            break
    
    if model is None:
        raise Exception(f"Could not find model {pre}")
    
    model.train(data="yolo_extra_data_model.yaml", name=pre+"ft_dataset_extra_fine_tuned_model_300epochs", epochs=300, patience=0)
    del model

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <model_config> <pre>")
        sys.exit(1)

    model_config = sys.argv[1]
    pre = sys.argv[2]
    fine_tune_yolo(model_config, pre)