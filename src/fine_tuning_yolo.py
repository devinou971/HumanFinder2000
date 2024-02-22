from ultralytics import YOLO
import sys
import os


# Load model
model = YOLO('yolov6n.yaml').load('yolov6n.pt')

dataset = "yolo_dataset2.0_model.yaml"
output_name = "v6n_dataset2_fine_tuned_model_300epochs"

if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    dataset = sys.argv[1]

if len(sys.argv) > 2:
    output_name = sys.argv[2]

print("dataset:",dataset)
print("output:", output_name)
if __name__ == '__main__':
    model.train(data=dataset, epochs=300, name=output_name)


