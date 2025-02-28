

### 1. Install Google cloud CLI
1. [Google cloud CLI](https://cloud.google.com/sdk/docs/install)
2. Login to you GCP account
```bash
gcloud auth login
gcloud auth application-default login
```
3. Create IAM for Cloud Storage Admin.
```bash
IAM & Admin -> Grant Access -> Add email -> Role :  Storage Admin
```
4. Create GCP Project and set this Project ID to gcloud
```bash
gcloud config set project PROJECT_ID
```



### 2. DVC
#### 1. Install [DVC](https://dvc.org/)
###  2. Install [DVC-gs] (for GCP Only)
```bash
pip install dvc-gs
```

#### 3. Init DVC
```bash
dvc init
git status
Changes to be committed:
        new file:   .dvc/.gitignore
        new file:   .dvc/config
        ...
git commit -m "Initialize DVC"
git push
```

#### Use dvc add to start tracking the dataset file:
```bash
dvc add 7_Data_version_control/cat_and_dog_dataset

```

Now dvc is creating a  cache file, please check <b> .dvc/cache/files/md5/.. </b>

#### 
```bash
git status
git add gcp_bucket.dvc
```

#### Configuring a remote with google drive
```bash
dvc remote add --default 7_Data_version_control/cat_and_dog_dataset gs://ths_dvc_test -f
dvc commit
sudo dvc push
```


#### Making Local changes
```bash
dvc add gcp_bucket
sudo dvc push
```



### Pull the data back
```bash
git checkout 6d65b91 #v2
git checkout fc703da #v1
dvc checkout
sudo dvc pull -f
```


#### now create new branch and make PR
```bash
 git switch -c new_branch
```