# Miscellaneous Notes - tidbits to remember

## Virtual Box 
Duplicate a virtual machine and change it to a variable size:
```
VBoxManage clonemedium ~/"VirtualBox VMs"/UBUNTU_18/UBUNTU_18.vdi ~/"VirtualBox VMs"/UBUNTU_18/UBUNTU_18_variablesize.vdi --variant Standard
VBoxManage modifyhd --resize 40000 ~/"VirtualBox VMs"/UBUNTU_18/UBUNTU_18_variablesize.vdi
```

## New environment in python
```
conda create —name <envs_name>
conda list
conda activate ./envs
conda info --envs
conda env remove -n ENV_NAME
conda create -n yourenvname python=3.7 anaconda
```

Make sure python is 3.7 when running all packages!
```
pip freeze > requirements.txt
pip download --destination-directory DIR -r requirements.txt
```

## Adding features to zsh
```
vim ~/.zshrc
```
