import config
from scene import Scene
if __name__ == "__main__":
    print("Rendering " + config.dcs.model_path)
    iteration = config.ocs.iterations
    scene = Scene(config.dcs, config.ocs, load_scene_iteration=iteration, shuffle=False)