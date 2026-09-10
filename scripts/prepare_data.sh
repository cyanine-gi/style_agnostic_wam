#download Tienkung
modelscope download  --dataset X-Humanoid/RoboMIND2.0-Tienkung \
    --include data/tienkung/tidy_desktop/success_episodes/0116_* \
    --local_dir ./RoboMIND2.0-Tienkung
#download Tienkung-sim
git clone https://www.modelscope.cn/datasets/X-Humanoid/RoboMIND2.0-Tienkung-sim.git
#download robotwin
git clone https://www.modelscope.cn/datasets/Dexmal/robotwin2-full.git
