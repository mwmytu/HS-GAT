# HS-GAT

# 1.title

**Hybrid Structural Graph Attention Network for POI Recommendation**

# 2.dataset

Yelp:  https://www.yelp.com/dataset

Boston: https://drive.google.com/drive/folders/1d8wMxfqKYHS1MKDVTBkx6WjZv-d9IsnB?usp=sharing

Chicago:https://drive.google.com/drive/folders/1-sVXrjprYmCiNySIF-JVsVc3URI7Po2y?usp=sharing

London:https://drive.google.com/drive/folders/1FTQ3gM1S8kO4vyPVBhMlsyUCDb4XZOrv?usp=sharing

# 3.code

3.1 Install third-party packages according to `requirement.txt`

3.2 Open `Yelp` or `Boston` folder and run `main.py` to train the model

# 4.Overall framework

Look at the model.png file.

![model _2_.png](https://s2.loli.net/2024/01/12/Qoy61jJnmSZq9Y4.png)

# 5.Directory structure

The section illustrates the directory structure and provides explanations for the functionalities of files in our project.

├────.gitignore																  
├────baseline/																  
│    ├────AGCN/			                  											  
│    ├────ATST_LSTM/                   								   
│    ├────DCF/                 												   
│    ├────DeepCLFM/					 								     
│    ├────FG-CF/							  							   
│    ├────HAMAP/						   								  
│    ├────MF/								   							  
│    ├────MLP/							     								   
│    ├────NMF/                   			  								  
│    ├────SEMA/							   								   
│    └────TLR/								   								  
├────Boston/								    									   
│    ├────config.py						   								  
│    ├────dataloader.py				  								  
│    ├────main.py							 								   
│    ├────main_eval.py													   
│    ├────model/							   								   
│    ├────model_boston_cpu_test2.py						   
│    └────utils.py															  
├────README.md															  
├────requirements.txt													  
├────tree.py																	   
└────Yelp/																		      
