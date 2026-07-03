from pathlib import Path    
from ultralytics import YOLO


from ultralytics.utils.plotting import plot_results
import torch

#from ultralytics import settings; settings.update({"tensorboard": True})
#C:/Users/Arno/.conda/envs/ml1/python.exe -c "from ultralytics import settings; settings.update({'tensorboard': True}); print(settings)"
#C:/Users/Arno/.conda/envs/ml1/python.exe -m tensorboard.main --logdir runs

device = 0 if torch.cuda.is_available() else "cpu"
batch = 16 if torch.cuda.is_available() else 4


def main():
    script_dir = Path(__file__).resolve().parent
    #data_yaml = (script_dir / "../Datasets/drone-racing-dataset/data/gate_pose_dataset/gate-pose.yaml").resolve()
    data_yaml = (script_dir / "../Datasets/OCTOPUS/dataset_split/OCTOPUS_isaac.yaml").resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    
    model_path = "yolo26n-pose.pt"   # oder z.B. yolo11n-pose.pt, je nach installierter Version
    #model_path = "yolo11n-pose.pt"   
    model = YOLO(model_path)

    model.train(
        data=str(data_yaml),
        epochs=250,
        imgsz=640,
        val=True,          # Validation aktiv (default=True)
        plots=True,        # Plots speichern        
        batch=batch,
        fraction=1,
        device=device,
        #mosaic=1.0,
        #mixup=0.0,
        #copy_paste=0.0,
        close_mosaic=2,
        patience=50,
        project="runs/gate_pose",        
        name="debug_run_ISAAC_5k_001"
    )


    plot_results("runs/pose/runs/gate_pose/debug_run_ISAAC_5k_001/results.csv")
    model = YOLO("runs/pose/runs/gate_pose/debug_run_ISAAC_5k_001/weights/best.pt")
    metrics = model.val(plots=True)
    print(metrics)


if __name__ == "__main__":
    #print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
    main()
    