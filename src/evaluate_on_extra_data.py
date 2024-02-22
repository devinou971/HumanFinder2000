from ultralytics import YOLO
from tqdm import tqdm

# Load model

models = [
    # "training_dataset2/v5n_dataset2_trained_model_300epochs",
    # "training_dataset2/v5n6_dataset2_trained_model_300epochs",
    # "training_dataset2/v6n_dataset2_trained_model_300epochs",
    # "training_dataset2/v8n_dataset2_trained_model_300epochs",

    # "training_dataset2/v5s_dataset2_trained_model_300epochs",
    # "training_dataset2/v5s6_dataset2_trained_model_300epochs",
    # "training_dataset2/v6s_dataset2_trained_model_300epochs",
    # "training_dataset2/v8s_dataset2_trained_model_300epochs",

    # "training_dataset2/v5m_dataset2_trained_model_300epochs",
    # "training_dataset2/v5m6_dataset2_trained_model_300epochs",
    # "training_dataset2/v6m_dataset2_trained_model_300epochs",
    # "training_dataset2/v8m_dataset2_trained_model_300epochs",

    # "training_dataset2/v5l_dataset2_trained_model_300epochs",
    # "training_dataset2/v5l6_dataset2_trained_model_300epochs",
    # "training_dataset2/v6l_dataset2_trained_model_300epochs",
    # "training_dataset2/v8l_dataset2_trained_model_300epochs",

    # "fine_tuning_dataset2/v5n_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v5n6_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v6n_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v8n_dataset2_fine_tuned_model_300epochs",

    # "fine_tuning_dataset2/v5s_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v5s6_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v6s_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v8s_dataset2_fine_tuned_model_300epochs",

    "fine_tuning_dataset2/v5m_dataset2_fine_tuned_model_300epochs",
    "fine_tuning_dataset2/v5m6_dataset2_fine_tuned_model_300epochs",
    "fine_tuning_dataset2/v6m_dataset2_fine_tuned_model_300epochs",
    "fine_tuning_dataset2/v8m_dataset2_fine_tuned_model_300epochs",

    # "fine_tuning_dataset2/v5l_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v5l6_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v6l_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v8l_dataset2_fine_tuned_model_300epochs",

    # "fine_tuning_dataset2/v7_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v7d6_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v7e6_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v7w6_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v7x_dataset2_fine_tuned_model_300epochs",
]

if __name__ == "__main__" : 
    for model_path in  tqdm(models):
        path = "runs/detect/" + model_path + "/weights/best.pt"
        mode_name = model_path.split("/")[-1]
        model = YOLO(path)
        res = model.val(data="yolo_extra_data_model.yaml", split="train", save_json=True, save_hybrid=False, plots=True, name="evaluations/eval_on_extra_data_"+mode_name)
        
        mp, mr, map50, map5095 = res.box.mean_results()
        with open("runs/detect/evaluations/eval_on_extra_data_"+mode_name+"/res.txt", "w") as f:
            f.write(f"mAP50-95;map50;prediction;recall\n{map5095};{map50};{mp};{mr}")