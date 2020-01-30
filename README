train_main.ipynb  ---> training notebook
             

├── _project
|   ├── train_main.ipynb    ---> training notebook,  needs :  helper.py
|   ├── helpers.py          ---> reads matlab data and applies downsampling to batches, needed by train_main.ipynb  
|   ├── test_interp.ipynb   ---> notebook to test saved models as a sanity check for training,
|   |                              it also does benchmark interpolation with SciPy
|   ├── figures4report.ipynb---> notebook to prepare figures for the final report,
|   |                                needs:   infer_dcnnCPU.py
|   └── infer_dcnnCPU.py    ---> does inference on CPU, needed by figures4report.ipynb
| 
├── _saved_models
|   └── model_0??.pth  ---> saved models in PyTorch format
|   
├── _data
|   ├── validation/saltdome_0???.mat ---> 10 examples of 150x300 for model validation 
|   └── landmass1/*.mat              ---> 77 examples of 99x99 for training 
|   
├── report.pdf
├── proposal.pdf
└── README (this file)
https://review.udacity.com/#!/reviews/2083305