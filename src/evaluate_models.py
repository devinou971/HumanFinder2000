from ultralytics import YOLO
from tqdm import tqdm
import torch
torch.manual_seed(0)

# Load model

datasets = {
    "dataset2": "yolo_dataset2.0_model.yaml",
    "dataset4": "yolo_dataset4_model.yaml"
}

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

    # "fine_tuning_dataset2/v5m_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v5m6_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v6m_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v8m_dataset2_fine_tuned_model_300epochs",

    # "fine_tuning_dataset2/v5l_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v5l6_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v6l_dataset2_fine_tuned_model_300epochs",
    # "fine_tuning_dataset2/v8l_dataset2_fine_tuned_model_300epochs",

    # "training_dataset4/v5n_dataset4_trained_model_300epochs",
    # "training_dataset4/v5n6_dataset4_trained_model_300epochs",
    # "training_dataset4/v6n_dataset4_trained_model_300epochs",
    # "training_dataset4/v8n_dataset4_trained_model_300epochs",

    # "training_dataset4/v5s_dataset4_trained_model_300epochs",
    # "training_dataset4/v5s6_dataset4_trained_model_300epochs",
    # "training_dataset4/v6s_dataset4_trained_model_300epochs",
    # "training_dataset4/v8s_dataset4_trained_model_300epochs",

    # "training_dataset4/v5m_dataset4_trained_model_300epochs",
    # "training_dataset4/v5m6_dataset4_trained_model_300epochs",
    # "training_dataset4/v6m_dataset4_trained_model_300epochs",
    # "training_dataset4/v8m_dataset4_trained_model_300epochs",

    # "training_dataset4/v5l_dataset4_trained_model_300epochs",
    # "training_dataset4/v5l6_dataset4_trained_model_300epochs",
    # "training_dataset4/v6l_dataset4_trained_model_300epochs",
    # "training_dataset4/v8l_dataset4_trained_model_300epochs",

    # "fine_tuning_dataset4/v5n_dataset4_fine_tuned_model_300epochs",
    # "fine_tuning_dataset4/v5n6_dataset4_fine_tuned_model_300epochs",
    "fine_tuning_dataset4/v6n_dataset4_fine_tuned_model_300epochs",
    "fine_tuning_dataset4/v8n_dataset4_fine_tuned_model_300epochs",

    "fine_tuning_dataset4/v5s_dataset4_fine_tuned_model_300epochs",
    "fine_tuning_dataset4/v5s6_dataset4_fine_tuned_model_300epochs",
    "fine_tuning_dataset4/v6s_dataset4_fine_tuned_model_300epochs",
    "fine_tuning_dataset4/v8s_dataset4_fine_tuned_model_300epochs",

    "fine_tuning_dataset4/v5m_dataset4_fine_tuned_model_300epochs",
    "fine_tuning_dataset4/v5m6_dataset4_fine_tuned_model_300epochs",
    "fine_tuning_dataset4/v6m_dataset4_fine_tuned_model_300epochs",
    "fine_tuning_dataset4/v8m_dataset4_fine_tuned_model_300epochs",

    "fine_tuning_dataset4/v5l_dataset4_fine_tuned_model_300epochs",
    "fine_tuning_dataset4/v5l6_dataset4_fine_tuned_model_300epochs",
    "fine_tuning_dataset4/v6l_dataset4_fine_tuned_model_300epochs",
    "fine_tuning_dataset4/v8l_dataset4_fine_tuned_model_300epochs",
]

if __name__ == "__main__" : 
    for model_path in  tqdm(models):
        path = "runs/detect/" + model_path + "/weights/best.pt"
        
        model_name = model_path.split("/")[-1]
        dataset = model_name.split("_")[1]
        
        model = YOLO(path)
        res = model.val(data=datasets[dataset], split="val", save_json=True, save_hybrid=False, plots=True, name="evaluations/eval_"+dataset+"/eval_"+model_name)
        with open("runs/detect/evaluations/eval_"+dataset+"/eval_"+model_name+"/res.txt", "w") as f:
            f.write(f"mAP50-95;map50;prediction\n{res.box.map};{res.box.map50};{res.box.p[0]}")

