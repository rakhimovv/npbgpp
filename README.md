# \[CVPR 2022\] NPBG++: Accelerating Neural Point-Based Graphics
### [Project Page](https://rakhimovv.github.io/npbgpp) | [Paper](https://arxiv.org/pdf/2203.13318.pdf)

This repository contains the official Python implementation of the paper.

The repository also contains faithful implementation of [_NPBG_]().

We introduce the pipelines working with following datasets: ScanNet, NeRF-Synthetic, H3DS, DTU.

We follow the PyTorch3D [_convention_](https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md) for coordinate systems and cameras.

# Changelog

- `[April 27, 2022]` Added more example data and point clouds
- `[April 5, 2022]` Initial code release

## Dependencies

```bash
python -m venv ~/.venv/npbgplusplus
source ~/.venv/npbgplusplus/bin/activate
pip install -r requirements.txt

# install pytorch3d
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
pip install "git+https://github.com/facebookresearch/pytorch3d.git@d0ca3b9e0cf6b1cfba46a367a98b8738cc5acad5" --no-cache-dir --verbose

# install torch_scatter
pip install torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-1.9.1+${CUDA}.html
# where ${CUDA} should be replaced by either cpu, cu101, cu102, or cu111 depending on your PyTorch installation.
# {CUDA} must match with torch.version.cuda (not with runtime or driver version)
# using 1.7.1 instead of 1.7.0 produces "incompatible cuda version" error

python setup.py build develop
```

Below you can see the examples on how to run the particular stages of different models on different datasets.

## How to run NPBG++

Checkpoints and example data are available [_here_](https://disk.yandex.ru/d/-1kx0XUlRHNumQ).

##### Run training

```bash
python train_net.py trainer.gpus=4 hydra.run.dir=experiments/npbgpp_scannet datasets=scannet_pretrain datasets.n_point=6e6 system=npbgpp_sphere system.visibility_scale=0.5 trainer.max_epochs=39 dataloader.train_data_mode=each trainer.reload_dataloaders_every_n_epochs=1
python train_net.py trainer.gpus=4 hydra.run.dir=experiments/npbgpp_nerf datasets=nerf_blender_pretrain system=npbgpp_sphere system.visibility_scale=1.0 trainer.max_epochs=24 dataloader.train_data_mode=each weights_path=experiments/npbgpp_scannet/checkpoints/epoch38.ckpt
python train_net.py trainer.gpus=4 hydra.run.dir=experiments/npbgpp_h3ds datasets=h3ds_pretrain system=npbgpp_sphere system.visibility_scale=1.0 trainer.max_epochs=24 dataloader.train_data_mode=each trainer.reload_dataloaders_every_n_epochs=1 weights_path=experiments/npbgpp_scannet/checkpoints/epoch38.ckpt
python train_net.py trainer.gpus=4 hydra.run.dir=experiments/npbgpp_dtu datasets=dtu_pretrain system=npbgpp_sphere system.visibility_scale=1.0 trainer.max_epochs=36 dataloader.train_data_mode=each trainer.reload_dataloaders_every_n_epochs=1  weights_path=experiments/npbgpp_scannet/checkpoints/epoch38.ckpt
```

##### Run testing

```bash
python train_net.py trainer.gpus=1 hydra.run.dir=experiments/npbgpp_eval_scan118 datasets=dtu_one_scene datasets.data_root=$\{hydra:runtime.cwd\}/example/DTU_masked datasets.scene_name=scan118 system=npbgpp_sphere system.visibility_scale=1.0 weights_path=./checkpoints/npbgpp_dtu_nm_mvs_ft_epoch35.ckpt eval_only=true dataloader=small
```

##### Run finetuning of coefficients

```bash
python train_net.py trainer.gpus=1 hydra.run.dir=experiments/npbgpp_5ae021f2805c0854_ft datasets=h3ds_one_scene datasets.data_root=$\{hydra:runtime.cwd\}/example/H3DS datasets.selection_count=0 datasets.train_num_samples=2000 datasets.train_image_size=null datasets.train_random_shift=false datasets.train_random_zoom=[0.5,2.0] datasets.scene_name=5ae021f2805c0854 system=coefficients_ft system.max_points=1e6 system.descriptors_save_dir=$\{hydra:run.dir\}/descriptors trainer.max_epochs=20 system.descriptors_pretrained_dir=experiments/npbgpp_eval_5ae021f2805c0854/descriptors weights_path=$\{hydra:runtime.cwd\}/checkpoints/npbgpp_h3ds.ckpt dataloader=small
```

##### Run testing with finetuned coefficients

```bash
python train_net.py trainer.gpus=1 hydra.run.dir=experiments/npbgpp_5ae021f2805c0854_test datasets=h3ds_one_scene datasets.data_root=$\{hydra:runtime.cwd\}/example/H3DS datasets.selection_count=0 datasets.scene_name=5ae021f2805c0854 system=coefficients_ft system.max_points=1e6 system.descriptors_save_dir=$\{hydra:run.dir\}/descriptors system.descriptors_pretrained_dir=experiments/npbgpp_5ae021f2805c0854_ft/descriptors weights_path=experiments/npbgpp_5ae021f2805c0854_ft/checkpoints/last.ckpt dataloader=small eval_only=true
```

## How to run NPBG

##### Run pretraining

```bash
python train_net.py trainer.gpus=4 hydra.run.dir=experiments/npbg_scannet datasets=scannet_pretrain datasets.train_random_zoom=[0.5,2.0] datasets.train_image_size=512 datasets.selection_count=0 system=npbg system.descriptors_save_dir=experiments/npbg_scannet/result/descriptors trainer.max_epochs=39 dataloader.train_data_mode=each trainer.reload_dataloaders_every_n_epochs=1 trainer.limit_val_batches=0 system.max_points=11e6
python train_net.py trainer.gpus=4 hydra.run.dir=experiments/npbg_nerf datasets=nerf_blender_pretrain datasets.train_random_zoom=[0.5,2.0] datasets.train_image_size=512 datasets.selection_count=0 system=npbg system.descriptors_save_dir=experiments/npbg_nerf/result/descriptors trainer.max_epochs=24 dataloader.train_data_mode=each trainer.reload_dataloaders_every_n_epochs=1 trainer.limit_val_batches=0 system.max_points=4e6
python train_net.py trainer.gpus=4 hydra.run.dir=experiments/npbg_h3ds datasets=h3ds_pretrain datasets.train_random_zoom=[0.5,2.0] datasets.train_image_size=null datasets.train_random_shift=false datasets.selection_count=0 system=npbg system.descriptors_save_dir=experiments/npbg_h3ds/result/descriptors trainer.max_epochs=24 dataloader.train_data_mode=each trainer.reload_dataloaders_every_n_epochs=1 trainer.limit_val_batches=0 system.max_points=3e6  # Submitted batch job 1175175
python train_net.py trainer.gpus=4 hydra.run.dir=experiments/npbg_dtu_nm datasets=dtu_pretrain datasets.train_random_zoom=[0.5,2.0] datasets.train_image_size=512 datasets.selection_count=0 system=npbg system.descriptors_save_dir=experiments/npbg_dtu_nm/result/descriptors trainer.max_epochs=36 dataloader.train_data_mode=each trainer.reload_dataloaders_every_n_epochs=1 trainer.limit_val_batches=0 system.max_points=3e6
```

##### Run fine-tuning on 1 scene

```bash
python train_net.py trainer.gpus=4 hydra.run.dir=experiments/npbg_scannet_0045 datasets=scannet_one_scene datasets.scene_name=scene0045_00 datasets.n_point=6e6 datasets.train_random_zoom=[0.5,2.0] datasets.train_image_size=512 datasets.selection_count=0 system=npbg system.descriptors_save_dir=experiments/npbg_scannet_0045/result/descriptors system.max_scenes_per_train_epoch=1 trainer.max_epochs=20 weights_path=experiments/npbg_scannet/result/checkpoints/epoch38.ckpt system.max_points=6e6
python train_net.py trainer.gpus=4 hydra.run.dir=experiments/npbg_nerf_hotdog datasets=nerf_blender_one_scene datasets.scene_name=hotdog datasets.train_random_zoom=[0.5,2.0] datasets.train_image_size=512 datasets.selection_count=0 system=npbg system.descriptors_save_dir=npbgplusplus/experiments/npbg_nerf_hotdog/result/descriptors system.max_scenes_per_train_epoch=1 trainer.max_epochs=20 weights_path=experiments/npbg_nerf/result/checkpoints/epoch23.ckpt system.max_points=4e6
python train_net.py trainer.gpus=4 hydra.run.dir=experiments/npbg_h3ds_5ae021f2805c0854 datasets=h3ds_one_scene datasets.scene_name=5ae021f2805c0854 datasets.train_random_zoom=[0.5,2.0] datasets.train_image_size=null datasets.train_random_shift=false datasets.selection_count=0 system=npbg system.descriptors_save_dir=experiments/npbg_h3ds_5ae021f2805c0854/result/descriptors system.max_scenes_per_train_epoch=1 trainer.max_epochs=20 weights_path=experiments/npbg_h3ds/result/checkpoints/epoch23.ckpt system.max_points=3e6
python train_net.py trainer.gpus=4 hydra.run.dir=experiments/npbg_dtu_nm_scan110 datasets=dtu_one_scene datasets.scene_name=scan110 datasets.train_random_zoom=[0.5,2.0] datasets.train_image_size=512 datasets.selection_count=0 system=npbg system.descriptors_save_dir=experiments/npbg_dtu_nm_scan110/result/descriptors system.max_scenes_per_train_epoch=1 trainer.max_epochs=20 weights_path=experiments/npbg_dtu_nm/result/checkpoints/epoch35.ckpt system.max_points=3e6
```

## Citation

If you find our work useful in your research, please consider citing:

```BibTeX
@InProceedings{Rakhimov_2022_CVPR,
    author={Rakhimov, Ruslan and Ardelean, Andrei-Timotei and Lempitsky, Victor and Burnaev, Evgeny},
    title={NPBG++: Accelerating Neural Point-Based Graphics},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2022},
    pages={15969-15979}
}
```

## License

See the [LICENSE](LICENSE) for more details.
