import papermill as pm

param_list = {
    'num_components': [2, 4, 8, 10, 16, 20, 32],
    'num_deciders': [100, 110, 120, 130, 140, 150, 200]
}

pm.execute_notebook('/home/ubuntu/Code/notebooks/Papermill.ipynb', '/home/ubuntu/Code/notebooks/Papermill_nb/