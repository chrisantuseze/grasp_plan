# GraspPlan
### [project website](https://grasp_plan.github.io) &emsp; &emsp; [arxiv paper](https://arxiv.org/abs/2311.00926) &emsp; &emsp; [model weights](https://drive.google.com/drive/folders/1qlvHVi1-Jk4ET-NyHwnqZOxALVy9kTO5)
<!-- ![robot](figures/real_robot.gif) -->

## Installation
1. Create and activate conda environment
```bash
conda create -n grasp_plan python=3.11
```
```bash
conda activate grasp_plan
```
2. Install CUDA. Replace `12.1.0` with the CUDA version compatible with your nvidia driver. You can check your CUDA version using `nvidia-smi`.
```bash
conda install cuda -c nvidia/label/cuda-12.1.0
```
3. Install PyTorch which matches the cuda version. Check https://pytorch.org/get-started/previous-versions.
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
4. Install PointNet++ custom ops.
```bash
pip install pointnet2_ops_lib/
```
5. Install additional dependencies.
```bash
pip install -r requirements.txt
```
6. Install m2t2 as a package.
```bash
pip install .
```

## Running M2T2
<!-- 1. Download [[model weights]](https://huggingface.co/wentao-yuan/m2t2). `m2t2.pth` is the generic pick-and-place model which outputs all possible grasping and placement poses, whereas `m2t2_language.pth` is a model that outputs a single grasping or placement pose conditioned on a language goal.

2. Run `meshcat-server` in a separate terminal. Open `http://127.0.0.1:7000/static/` in a browser window for visualization.

3. Try out M2T2 on different scenes under `sample_data`.
    1. `real_world/00` and `real_world/02` contain real-world scenes before grasping
    ```
    python demo.py eval.checkpoint=m2t2.pth eval.data_dir=sample_data/real_world/00 eval.mask_thresh=0.4 eval.num_runs=5
    ```
    Use `eval.mask_thresh` to control the confidence threshold for output grasps. Use `eval.num_runs` to aggregate multiple forward passes for more complete prediction.

    2. `real_world/00` and `real_world/02` contain real-world scenes before placement. It can be run with the same command. Just replace the path in `eval.data_dir`.

    3. Under `rlbench` there are two episodes of the task `meat_off_grill`. Running the following command to see the grasp prediction
    ```
    python demo_rlbench.py eval.checkpoint=m2t2_language.pth rlbench.episode=4
    ```
    and
    ```
    python demo_rlbench.py eval.checkpoint=m2t2_language.pth rlbench.episode=4 rlbench.frame_id=90
    ```
    to see the placement prediction. -->

## Training
<!-- We are not able to release the training dataset at this point due to copyright issues, but we do provide the training script and a few simulated scenes under `sample_data/simulation` to show the data format. Here's how you would train the model.
```
python train.py m2t2.action_decoder.max_num_pred=512 data.root_dir=sample_data/simulation train.log_dir=logs/test train.num_gpus=1 train.batch_size=4 train.print_freq=1 train.plot_freq=4
```
You can visualize training curves and plots by running tensorboard on the log directory:
```
tensorboard --logdir logs/test
``` -->

If you find our work useful, please consider citing our paper:
```
@inproceedings{eze2026,
  title     = {GraspPlan},
  author    = {Eze, Chrisantus and Crick, Christopher},
  booktitle = {},
  year      = {2026}
}
```