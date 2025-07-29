import os
import shutil
backbones = ["r3m","vc1","resnet","dino","dino_cls","scratch","full_scratch"]
tasks = ["cargoal","maniskill","carla","dubins"]
flags = ["with-proprio","without-proprio"]

for backbone in backbones:
    for task in tasks:
        for flag in flags:
            ckpt_dir = f"/storage1/sibai/Active/ihab/research_new/checkpt_dino/outputs2/{task}/{backbone}/classifier"
            # os.removedirs(ckpt_dir)
            shutil.rmtree(ckpt_dir,ignore_errors=True)
            print(ckpt_dir)

shutil.rmtree("/storage1/sibai/Active/ihab/research_new/dino_wm/runs",ignore_errors=True)