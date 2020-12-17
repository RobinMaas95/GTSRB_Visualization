# GTSRB_Visualization
## Instructions for validation

### Using Colab (recommended)
1. Fork this repository
2. Open Colab (`http://colab.research.google.com`)
3. 	`File` -> `Open Notebook` -> `GitHub` (You must be logged in to see the GitHub option)
4. Select your Fork in the `Repository` dropdown
5. Select `code/run_test_performance.ipynb`
6. Run cells
7. Due to the size, the `perspective` and `rotated` dataset could not be made available via GitHub. The files are hosted on `OneDrive` and will be downloaded within the notebook. If the links no longer work, you will need to generate the datasets yourself. See section `Instructions for reproduction`/`Generate perspective un rotated` 	

### Using local environment
1. Clone this repository
2. Install dependencies (Using an venv is recommend)
	- Install pytorch/torchvision. These versions are for use without Cuda, newer versions and CUDA compatible versions should work. Selection of versions to install for CUDA support depends on your system)
		- `pip install torch===1.7.0 torchvision===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`
	- Install other dependencies
		- `pip install -r requirements.txt`
3. Unzip checkpoints (necessary because checkpoints exceed GitHub file size limit)
	- Paths may need to be adjusted 
	- Unix:
		- `cat /checkpoints/splitted_trained_model.zip.00* > trained_model.zip`
		- `cat /checkpoints/splitted_trained_model_org.zip.00* > trained_model_org.zip`
		- `unzip -qq trained_model.zip`
		- `unzip -qq trained_model_org.zip`
	- Windows
		- `type /checkpoints/splitted_trained_model.zip.001 /checkpoints/splitted_trained_model.zip.002 /checkpoints/splitted_trained_model.zip.003 > trained_model.zip`
		- `type /checkpoints/splitted_trained_model_org.zip.001 /checkpoints/splitted_trained_model_org.zip.002 /checkpoints/splitted_trained_model_org.zip.003 > trained_model_org.zip`
		- Unzip `trained_model.zip` and `trained_model_org.zip` with an archive manager of your choice
4. Due to the size, the perspective and rotated dataset could not be made available via GitHub. The files are hosted on `OneDrive` but must be downloaded using the following links and then unpacked. If the links no longer work, you will need to generate the datasets yourself. See section `Instructions for reproduction`/`Generate perspective un rotated` 
	- [Perspective](https://onedrive.live.com/download?cid=A75A55F06E327965&resid=A75A55F06E327965%212612&authkey=AJMj_aXD29yyN-8)
	- [Rotated](https://onedrive.live.com/download?cid=A75A55F06E327965&resid=A75A55F06E327965%212613&authkey=AMvt8Bhj5RLHo9g)

4. Start `jupyter lab` and navigate to this repository. Open `code/run_test_performance.ipynb`
5. Run cells 
	- **Paths must be manually adjusted! See Section in the notebook
`Set Paths (LOCAL ENV)`**

## Instructions for reproduction TODO

1. Run **get_cropped_datasets.ipynb**
	* 	Downloads all neccessary datasets
	*  Crops trainings and test dataset
	*  Results of this notebook are in the directories **cropped_training** and 	**cropped_test**

2. Run **flip_images.ipynb**
	* Flips images horiontal/vertical (depending on image class)

3. Run **upsample_dataset.ipynb**
	* Duplicates images to prevent overrepresentation of singel classes

4. Run **rotate_images.ipynb**
	* Rotates trainings dataset corresponding to a normal distribution (+-20 degree).

# Commads

``` Bash
! python mask_images.py --heatmaps "/content/visualizations" \
--json_target /content/masking_jsons --json_file _heatmap_masked.csv \
--org_images "/content/cropped_test" \
--target /content/masking_results/ 
```
