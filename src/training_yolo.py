import os
from time import sleep
os.environ["WANDB_DISABLED"] = "true"


def start_training(model_config, dataset_config, pre_name, model_weights = None):
    os.system(f"python src/utils/training.py {model_config} {dataset_config} {pre_name} {model_weights if model_weights is not None else ''}")
    sleep(30)

if __name__ == '__main__':
    # Define configurations for fine-tuning
    fine_tuning_configs = [
        # -------------------------- DATASET 2
        ("yolov5n.yaml", "yolo_dataset2.0_model.yaml", "training_dataset2/v5n_dataset2_trained_model_300epochs"),
        ("yolov5n-p6.yaml", "yolo_dataset2.0_model.yaml","training_dataset2/v5n6_dataset2_trained_model_300epochs"),
        ("yolov6n.yaml", "yolo_dataset2.0_model.yaml","training_dataset2/v6n_dataset2_trained_model_300epochs"),
        ("yolov8n.yaml", "yolo_dataset2.0_model.yaml","training_dataset2/v8n_dataset2_trained_model_300epochs"),

        ("yolov5s.yaml", "yolo_dataset2.0_model.yaml","training_dataset2/v5s_dataset2_trained_model_300epochs"),
        ("yolov5s-p6.yaml", "yolo_dataset2.0_model.yaml","training_dataset2/v5s6_dataset2_trained_model_300epochs"),
        ("yolov6s.yaml", "yolo_dataset2.0_model.yaml","training_dataset2/v6s_dataset2_trained_model_300epochs"),
        ("yolov8s.yaml", "yolo_dataset2.0_model.yaml","training_dataset2/v8s_dataset2_trained_model_300epochs"),

        ("yolov5m.yaml", "yolo_dataset2.0_model.yaml","training_dataset2/v5m_dataset2_trained_model_300epochs"),
        ("yolov5m-p6.yaml", "yolo_dataset2.0_model.yaml","training_dataset2/v5m6_dataset2_trained_model_300epochs"),
        ("yolov6m.yaml", "yolo_dataset2.0_model.yaml","training_dataset2/v6m_dataset2_trained_model_300epochs"),
        ("yolov8m.yaml", "yolo_dataset2.0_model.yaml","training_dataset2/v8m_dataset2_trained_model_300epochs"),

        ("yolov5l.yaml", "yolo_dataset2.0_model.yaml","training_dataset2/v5l_dataset2_trained_model_300epochs"),
        ("yolov5l-p6.yaml", "yolo_dataset2.0_model.yaml","training_dataset2/v5l6_dataset2_trained_model_300epochs"),
        ("yolov6l.yaml", "yolo_dataset2.0_model.yaml","training_dataset2/v6l_dataset2_trained_model_300epochs"),
        ("yolov8l.yaml", "yolo_dataset2.0_model.yaml","training_dataset2/v8l_dataset2_trained_model_300epochs"),


        ("yolov5n.yaml",  "yolo_dataset2.0_model.yaml", "fine_tuning_dataset2/v5n_dataset2_fine_tuned_model_300epochs", "yolov5nu.pt"),
        ("yolov5n-p6.yaml", "yolo_dataset2.0_model.yaml","fine_tuning_dataset2/v5n6_dataset2_fine_tuned_model_300epochs", "yolov5n6u.pt"),
        ("yolov6n.yaml", "yolo_dataset2.0_model.yaml","fine_tuning_dataset2/v6n_dataset2_fine_tuned_model_300epochs",  "yolov6n.pt",),
        ("yolov8n.yaml", "yolo_dataset2.0_model.yaml","fine_tuning_dataset2/v8n_dataset2_fine_tuned_model_300epochs",  "yolov8n.pt",),

        ("yolov5s.yaml", "yolo_dataset2.0_model.yaml","fine_tuning_dataset2/v5s_dataset2_fine_tuned_model_300epochs", "yolov5su.pt"),
        ("yolov5s-p6.yaml", "yolo_dataset2.0_model.yaml","fine_tuning_dataset2/v5s6_dataset2_fine_tuned_model_300epochs", "yolov5s6u.pt"),
        ("yolov6s.yaml", "yolo_dataset2.0_model.yaml","fine_tuning_dataset2/v6s_dataset2_fine_tuned_model_300epochs", "yolov6s.pt"),
        ("yolov8s.yaml", "yolo_dataset2.0_model.yaml","fine_tuning_dataset2/v8s_dataset2_fine_tuned_model_300epochs", "yolov8s.pt"),

        ("yolov5m.yaml", "yolo_dataset2.0_model.yaml","fine_tuning_dataset2/v5m_dataset2_fine_tuned_model_300epochs", "yolov5mu.pt"),
        ("yolov5m-p6.yaml", "yolo_dataset2.0_model.yaml","fine_tuning_dataset2/v5m6_dataset2_fine_tuned_model_300epochs", "yolov5m6u.pt"),
        ("yolov6m.yaml", "yolo_dataset2.0_model.yaml","fine_tuning_dataset2/v6m_dataset2_fine_tuned_model_300epochs", "yolov6m.pt"),
        ("yolov8m.yaml", "yolo_dataset2.0_model.yaml","fine_tuning_dataset2/v8m_dataset2_fine_tuned_model_300epochs", "yolov8m.pt"),

        ("yolov5l.yaml", "yolo_dataset2.0_model.yaml","fine_tuning_dataset2/v5l_dataset2_fine_tuned_model_300epochs", "yolov5lu.pt"),
        ("yolov5l-p6.yaml", "yolo_dataset2.0_model.yaml","fine_tuning_dataset2/v5l6_dataset2_fine_tuned_model_300epochs", "yolov5l6u.pt"),
        ("yolov6l.yaml", "yolo_dataset2.0_model.yaml","fine_tuning_dataset2/v6l_dataset2_fine_tuned_model_300epochs", "yolov6l.pt"),
        ("yolov8l.yaml", "yolo_dataset2.0_model.yaml","fine_tuning_dataset2/v8l_dataset2_fine_tuned_model_300epochs", "yolov8l.pt"),

        # -------------------------- DATASET 4 

        ("yolov5n.yaml", "yolo_dataset4.0_model.yaml", "training_dataset4/v5n_dataset4_trained_model_300epochs"),
        ("yolov5n-p6.yaml", "yolo_dataset4.0_model.yaml","training_dataset4/v5n6_dataset4_trained_model_300epochs"),
        ("yolov6n.yaml", "yolo_dataset4.0_model.yaml","training_dataset4/v6n_dataset4_trained_model_300epochs"),
        ("yolov8n.yaml", "yolo_dataset4.0_model.yaml","training_dataset4/v8n_dataset4_trained_model_300epochs"),

        ("yolov5s.yaml", "yolo_dataset4.0_model.yaml","training_dataset4/v5s_dataset4_trained_model_300epochs"),
        ("yolov5s-p6.yaml", "yolo_dataset4.0_model.yaml","training_dataset4/v5s6_dataset4_trained_model_300epochs"),
        ("yolov6s.yaml", "yolo_dataset4.0_model.yaml","training_dataset4/v6s_dataset4_trained_model_300epochs"),
        ("yolov8s.yaml", "yolo_dataset4.0_model.yaml","training_dataset4/v8s_dataset4_trained_model_300epochs"),

        ("yolov5m.yaml", "yolo_dataset4.0_model.yaml","training_dataset4/v5m_dataset4_trained_model_300epochs"),
        ("yolov5m-p6.yaml", "yolo_dataset4.0_model.yaml","training_dataset4/v5m6_dataset4_trained_model_300epochs"),
        ("yolov6m.yaml", "yolo_dataset4.0_model.yaml","training_dataset4/v6m_dataset4_trained_model_300epochs"),
        ("yolov8m.yaml", "yolo_dataset4.0_model.yaml","training_dataset4/v8m_dataset4_trained_model_300epochs"),

        ("yolov5l.yaml", "yolo_dataset4.0_model.yaml","training_dataset4/v5l_dataset4_trained_model_300epochs"),
        ("yolov5l-p6.yaml", "yolo_dataset4.0_model.yaml","training_dataset4/v5l6_dataset4_trained_model_300epochs"),
        ("yolov6l.yaml", "yolo_dataset4.0_model.yaml","training_dataset4/v6l_dataset4_trained_model_300epochs"),
        ("yolov8l.yaml", "yolo_dataset4.0_model.yaml","training_dataset4/v8l_dataset4_trained_model_300epochs"),


        
        ("yolov5n.yaml",  "yolo_dataset4.0_model.yaml", "fine_tuning_dataset4/v5n_dataset4_fine_tuned_model_300epochs", "yolov5nu.pt"),
        ("yolov5n-p6.yaml", "yolo_dataset4.0_model.yaml","fine_tuning_dataset4/v5n6_dataset4_fine_tuned_model_300epochs", "yolov5n6u.pt"),
        ("yolov6n.yaml", "yolo_dataset4.0_model.yaml","fine_tuning_dataset4/v6n_dataset4_fine_tuned_model_300epochs",  "yolov6n.pt",),
        ("yolov8n.yaml", "yolo_dataset4.0_model.yaml","fine_tuning_dataset4/v8n_dataset4_fine_tuned_model_300epochs",  "yolov8n.pt",),

        ("yolov5s.yaml", "yolo_dataset4.0_model.yaml","fine_tuning_dataset4/v5s_dataset4_fine_tuned_model_300epochs", "yolov5su.pt"),
        ("yolov5s-p6.yaml", "yolo_dataset4.0_model.yaml","fine_tuning_dataset4/v5s6_dataset4_fine_tuned_model_300epochs", "yolov5s6u.pt"),
        ("yolov6s.yaml", "yolo_dataset4.0_model.yaml","fine_tuning_dataset4/v6s_dataset4_fine_tuned_model_300epochs", "yolov6s.pt"),
        ("yolov8s.yaml", "yolo_dataset4.0_model.yaml","fine_tuning_dataset4/v8s_dataset4_fine_tuned_model_300epochs", "yolov8s.pt"),

        ("yolov5m.yaml", "yolo_dataset4.0_model.yaml","fine_tuning_dataset4/v5m_dataset4_fine_tuned_model_300epochs", "yolov5mu.pt"),
        ("yolov5m-p6.yaml", "yolo_dataset4.0_model.yaml","fine_tuning_dataset4/v5m6_dataset4_fine_tuned_model_300epochs", "yolov5m6u.pt"),
        ("yolov6m.yaml", "yolo_dataset4.0_model.yaml","fine_tuning_dataset4/v6m_dataset4_fine_tuned_model_300epochs", "yolov6m.pt"),
        ("yolov8m.yaml", "yolo_dataset4.0_model.yaml","fine_tuning_dataset4/v8m_dataset4_fine_tuned_model_300epochs", "yolov8m.pt"),

        ("yolov5l.yaml", "yolo_dataset4.0_model.yaml","fine_tuning_dataset4/v5l_dataset4_fine_tuned_model_300epochs", "yolov5lu.pt"),
        ("yolov5l-p6.yaml", "yolo_dataset4.0_model.yaml","fine_tuning_dataset4/v5l6_dataset4_fine_tuned_model_300epochs", "yolov5l6u.pt"),
        ("yolov6l.yaml", "yolo_dataset4.0_model.yaml","fine_tuning_dataset4/v6l_dataset4_fine_tuned_model_300epochs", "yolov6l.pt"),
        ("yolov8l.yaml", "yolo_dataset4.0_model.yaml","fine_tuning_dataset4/v8l_dataset4_fine_tuned_model_300epochs", "yolov8l.pt"),
    ]

    # Start fine-tuning for each configuration
    for config in fine_tuning_configs:
        if not os.path.exists("runs/detect/"+config[2]):
            start_training(*config)


    
