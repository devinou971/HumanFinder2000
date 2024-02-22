from ultralytics import YOLO


# Load model
models = [
    "fine_tuning_dataset2/v6m_dataset2_fine_tuned_model_300epochs",
    "fine_tuning_dataset2/v8m_dataset2_fine_tuned_model_300epochs",
    "fine_tuning_dataset4/v8s_dataset4_fine_tuned_model_300epochs",
]

if __name__ == '__main__':
    for model_name in models:
        model = YOLO("runs/detect/" + model_name + "/weights/best.pt") 
        model.predict("inputs/all_images", save=True, name="predictions_" + model_name.split("/")[-1])