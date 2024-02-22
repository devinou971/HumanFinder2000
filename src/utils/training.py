import sys
from ultralytics import YOLO

def train_yolo(model_config, model_pretrained_weights, dataset_conf, output_name, resume=False):
    # Load model
    
    if resume:
        model = YOLO(model_config)
        model.train(data=dataset_conf, name=output_name, resume=True, device=0, patience=0)    
    else :
        model = YOLO(model_config)
        if model_pretrained_weights is not None:
            model = model.load(model_pretrained_weights)
        model.train(data=dataset_conf, name=output_name, epochs=300, patience=0, device=0)
    del model

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python script_name.py <model_config> <dataset_conf> <output_name> <model_pretrained_weights> <resuming>")
        sys.exit(1)

    model_config = sys.argv[1]
    dataset_conf = sys.argv[2]
    output_name = sys.argv[3]

    model_pretrained_weights = None
    if len(sys.argv) >= 5:
        model_pretrained_weights = sys.argv[4]

    resuming = False
    if len(sys.argv) >= 6:
        resuming = True if sys.argv[5] == "TRUE" else False

    train_yolo(model_config, model_pretrained_weights, dataset_conf, output_name, resuming)
