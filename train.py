import config
import torch
from tqdm import tqdm
from scene import Scene
from utils.pre_utils import prepare

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training():
    tb_writer = prepare(config.dcs.model_path)
    scene = Scene(config.dcs, config.ocs)
    scene.training_setup(config.ocs)
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    progress_bar = tqdm(range(0, config.ocs.iterations), desc="Training progress")

    for iteration in range(1, config.ocs.iterations + 1):
        iter_start.record()
        scene.optimization(iteration, config.ocs)
        iter_end.record()
        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.update(10)
            if iteration == config.ocs.iterations:
                progress_bar.close()
            
            if iteration == config.ocs.iterations:
                scene.save_scene_as_ply(config.dcs.model_path, iteration)
            
            scene.gaussian_densify(iteration, config.dcs, config.ocs)
            
            if iteration < config.ocs.iterations:
                # Step both optimizers: Gaussians and Opacoxel logits
                scene.gaussians._optimizer.step()
                scene.opacity_optimizer.step()
                scene.gaussians._optimizer.zero_grad(set_to_none = True)
                scene.opacity_optimizer.zero_grad(set_to_none = True)
    
if __name__ == "__main__":
    print("Optimizing " + config.dcs.model_path)

    training()

    print("\nTraining complete.")
