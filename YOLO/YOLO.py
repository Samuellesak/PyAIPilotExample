from pathlib import Path    
from ultralytics import YOLO


from ultralytics.utils.plotting import plot_results
import torch

#from ultralytics import settings; settings.update({"tensorboard": True})
#C:/Users/Arno/.conda/envs/ml1/python.exe -c "from ultralytics import settings; settings.update({'tensorboard': True}); print(settings)"
#C:/Users/Arno/.conda/envs/ml1/python.exe -m tensorboard.main --logdir runs

def _cuda_works():
    """Return True only if torchvision CUDA ops (NMS) are functional.
    torch.cuda.is_available() can be True while torchvision was built without
    a matching CUDA runtime, causing NotImplementedError at NMS."""
    if not torch.cuda.is_available():
        return False
    try:
        import torchvision.ops as _ops
        _b = torch.tensor([[0., 0., 1., 1.]], device='cuda')
        _s = torch.tensor([1.], device='cuda')
        _ops.nms(_b, _s, 0.5)
        return True
    except Exception as _e:
        print(f'[WARN] CUDA available but torchvision NMS failed ({_e}). '
              f'Falling back to CPU.\n'
              f'Fix: pip install torch torchvision --index-url '
              f'https://download.pytorch.org/whl/cu{torch.version.cuda.replace(".", "")}')
        return False

_use_cuda = _cuda_works()
device = 0 if _use_cuda else "cpu"
batch  = 16 if _use_cuda else 4


def main():
    script_dir = Path(__file__).resolve().parent
    data_yaml  = (script_dir / "Training/dataset.yaml").resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}\n"
            "Run 'python YOLO/Training/generate_labels.py' first.")

    model = YOLO(script_dir / "best.pt")   # fine-tune from existing gate-detection model

    run_name    = "sim_flights_v4"
    project_dir = script_dir / "runs" / "gate_pose"   # absolute → no Ultralytics prefix nesting
    results = model.train(
        data=str(data_yaml),
        epochs=200,
        imgsz=640,
        val=True,
        plots=True,
        batch=batch,
        fraction=1,
        device=device,
        close_mosaic=10,
        patience=50,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
    )

    # Ultralytics returns the actual save directory in results.save_dir
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Best weights: {best_weights}")
    model = YOLO(str(best_weights))
    metrics = model.val(plots=True)
    print(metrics)


if __name__ == "__main__":
    #print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
    main()
    