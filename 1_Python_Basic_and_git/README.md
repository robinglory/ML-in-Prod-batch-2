#### Basic Anaconda commands

```bash
# list envs
conda env list   

# create env
conda create -n test python==3.11

# activate env
conda activate test

# delete env
conda remove test_env --all 
```


### Git basic commands
```bash
git status
git add .
git commit -m "this is a message 2"
git push

```



### Create SSH connection connect
##### generate RSA public key
```bash
ssh-keygen -t rsa -f tharhtetsan -C tharhtetsan.ai@gmail.com

# for window check
#C:\Users\username\

# for mac check .ssh under username
# eg: /Users/tharhtet/.ssh/id_e
```