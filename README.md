# rl learnnig

## CUDA installation

-   (apparently only nvidia driver is necessary - pytorch installs its own cuda packages)
-   installed nvidia drivers from rpm fusion - https://rpmfusion.org/Howto/NVIDIA#Installing_the_drivers
-   installed cuda toolik from rpm fusion - https://rpmfusion.org/Howto/CUDA
-   did any `.zshrc` settings (not sure if necessary) from https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions

```bash
export PATH=${PATH}:/usr/local/cuda-12.9/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.9/lib64
export NVCC_CCBIN='g++-13'
```

-   tested ./deviceQuery, nvidia-smi working

### other useful cuda resources

-   https://discussion.fedoraproject.org/t/issues-with-cuda-installation-on-fedora-41-and-gnome-conflicts/141096/4
-   https://www.reddit.com/r/Fedora/comments/1gpp5a4/how_to_install_nvcc_on_fedora_41/
-   https://www.reddit.com/r/Fedora/comments/p93gnl/unable_to_use_cuda_with_pytorch/

### pytorch install

-   `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
