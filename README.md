# GTSRB_Visualization
Analysis of visualization methods for relevant areas within images for the CNN. Used the [GTSRB](https://benchmark.ini.rub.de/) dataset as well as Activation Maximization[1], Saliency Map[2], GradCam[3], and Gradcam++[4] methods.
Code and evaluations were developed as part of my master's thesis.

Code from the following repositories was used within this work:
  
* [nn_interpretability](https://github.com/hans66hsu/nn_interpretability) von hans66hsu
* [rotate\_3d](https://github.com/eborboihuc/rotate_3d) von Hou-Ning Hu / [@eborboihuc](https://eborboihuc.github.io/)
* [SmoothGradCAMplusplus](https://github.com/yiskw713/SmoothGradCAMplusplus) von Yuchi Ishikawa [@yiskw713](https://yiskw713.github.io/) 

## Instructions for validation

### Using Colab (recommended)
1. Fork this repository
2. Open Colab (`http://colab.research.google.com`)
3. 	`File` -> `Open Notebook` -> `GitHub` (You must be logged in to see the GitHub option)
4. Select your Fork in the `Repository` dropdown
5. Select `code/run_test_performance.ipynb`
6. Run cells
7. Due to the size, the `perspective` and `rotated` dataset could not be made available via GitHub. The files are hosted on `OneDrive` and will be downloaded within the notebook. If the links no longer work, you will need to generate the datasets yourself. See section `Instructions for reproduction`/`Step 3 and 4` 	

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
4. Due to the size, the perspective and rotated dataset could not be made available via GitHub. The files are hosted on `OneDrive` but must be downloaded using the following links and then unpacked. If the links no longer work, you will need to generate the datasets yourself. See section `Instructions for reproduction`/`Step 3 and 4` 
	- [Perspective](https://onedrive.live.com/download?cid=A75A55F06E327965&resid=A75A55F06E327965%212612&authkey=AJMj_aXD29yyN-8)
	- [Rotated](https://onedrive.live.com/download?cid=A75A55F06E327965&resid=A75A55F06E327965%212613&authkey=AMvt8Bhj5RLHo9g)

4. Start `jupyter lab` and navigate to this repository. Open `code/run_test_performance.ipynb`
5. Run cells 
	- **Paths must be manually adjusted! See Section in the notebook
`Set Paths (LOCAL ENV)`**

## Instructions for reproduction

1. Run **/code/gtsrb\_visualization.ipynb**
	* 	Downloads all neccessary datasets
	*  Crops trainings and test dataset (if you choose so)
	*  Prepares data sets (mirroring, upsampling, rotation)
	*  Performs training of the CNN
	*  Applies visualization methods
	*  Creates datasets where the cells of the grid are masked based on the evaluation of the visualization methods
	*  Note: Activation Maximation is performed here, but the results are not used further (presented exclusively in the Thesis).

2. Run **/code/calc\_sticker\_position\_by\_grid.ipynb**
	* Placed stickers on stock images based on the evaluations of visualization methods
	* /data/raw_sticker/ contains the different sticker sizes for the signs
	* /data/sticker/orginal contains the stock photo signs without stickers	
3. Run **/code/rotate_dataset.ipynb**
	* Rotates images based on the x-axis. Creates a separate dataset for each degree number

4. Run **/code/perspective/main.py**
	* Rotates images based on the x-axis. Creates a separate dataset for each degree number
	* Call via console `python main.py`
	* For the necessary arguments use `--help` or read in the source code

5. Run **/code/run\_test\_performance.ipynb**
	* Performs the following tests dTests CNN with non-manipulated data set
	* Testing the rotated (x and y axis)
	* Testing the masked data sets
	* Testing the datasets with stickers
	* Testing the real examples 

## References
[1] Montavon, Grégoire and Samek, Wojciech and Müller, Klaus-Robert. Methods for interpreting and understanding deep neural networks. Digital Signal Processing, 73:1–15, Feb 2018. | [Paper](https://arxiv.org/abs/1706.07979)

[2] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks:Visualising image classification models and saliency maps, 2013. | [Paper](https://arxiv.org/pdf/1312.6034.pdf)  

[3] Ramprasaath R. Selvaraju, Abhishek Das, Ramakrishna Vedantam, Michael Cogswell, DeviParikh, and Dhruv Batra. Grad-cam: Why did you say that? visual  explanations from deepnetworks via gradient-based localization.CoRR, abs/1610.02391, 2016. | [Paper](https://arxiv.org/pdf/1610.02391.pdf)

[4] Chattopadhay, A., Sarkar, A., Howlader, P., Balasubramanian, V. N. Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks, in: 2018 IEEE winter conference on applications of computer vision (WACV), 2018, S. 839–847 | [Paper](https://arxiv.org/abs/1710.11063) 
