# HS-GAT

# 1.title

**Hybrid Structural Graph Attention Network for POI Recommendation**

# 2.dataset

Yelp:  https://www.yelp.com/dataset

Boston: https://drive.google.com/drive/folders/11eikgU0Pu_63-VMKPluyTK0mXA10j6Jk?usp=drive_link

# 3.code

3.1 Install third-party packages according to `requirement.txt`

3.2 Open `Yelp` or `Boston` folder and run `main.py` to train the model

## 4.Overall framework

Look at the model.png file.

## 5.Directory structure

The section illustrates the directory structure and provides explanations for the functionalities of files in our project.

├────.gitignore																# **git ignore files**
├────baseline/																# **baseline implementation code**
│    ├────AGCN/			                  								#**AGCN model** 				  
│    ├────ATST_LSTM/                   								 # **ATST_LSTM model**
│    ├────DCF/                 												# **DCF model**
│    ├────DeepCLFM/					 								# **DeepCLFM model**
│    ├────FG-CF/							  								# **FG-CF model**
│    ├────HAMAP/						   								# **HAMAP model**
│    ├────MF/								   								# **MF model**
│    ├────MLP/							     								# **MLP model**
│    ├────NMF/                   			  								# **NMF model**
│    ├────SEMA/							   								# **SEMA model**
│    └────TLR/								   								# **TLR model**
├────Boston/								    								# **our approach in the Boson dataset**	
│    ├────config.py						   								# **constant configuration file**
│    ├────dataloader.py				  								# **data Loading and Processing Files**
│    ├────main.py							 								# **entry file**
│    ├────main_eval.py													# **model test file**
│    ├────model/							   								# **model saving directory**
│    ├────model_boston_cpu_test2.py						  # **model file**
│    └────utils.py															   # **utility function file**
├────README.md															# **project description file**
├────requirements.txt													# **environment configuration file**
├────tree.py																	  # **directory structure generation file**
└────Yelp/																		  # **our approach in the Yelp dataset**	
