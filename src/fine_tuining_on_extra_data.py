from ultralytics import YOLO
import os
from time import sleep
os.environ["WANDB_DISABLED"] = "true"


def start_fine_tuning(model_config, pre):
    os.system(f"python src\\utils\\fine_tune.py {model_config} {pre}")
    sleep(60)

if __name__ == '__main__':
    # Define configurations for fine-tuning
    fine_tuning_configs = [
        ("yolov5n.yaml", "v5n"),
        ("yolov5n-p6.yaml", "v5n6"),
        ("yolov6n.yaml", "v6n"),
        ("yolov8n.yaml", "v8n"),

        ("yolov5s.yaml", "v5s"),
        ("yolov5s-p6.yaml", "v5s6"),
        ("yolov6s.yaml", "v6s"),
        ("yolov8s.yaml", "v8s"),

        ("yolov5m.yaml", "v5m"),
        ("yolov5m-p6.yaml", "v5m6"),
        ("yolov6m.yaml", "v6m"),
        ("yolov8m.yaml", "v8m"),

        ("yolov5l.yaml", "v5l"),
        ("yolov5l-p6.yaml", "v5l6"),
        ("yolov6l.yaml", "v6l"),
        ("yolov8l.yaml", "v8l")

        # Add more configurations if needed
    ]

    # Start fine-tuning for each configuration
    for config in fine_tuning_configs:
        start_fine_tuning(*config)
