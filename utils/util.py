import os

def make_directory(root, session, checkpoints, tensorboard):
    if not os.path.exists(root):
        os.makedirs(root)
    
    if not os.path.exists(session):
        os.makedirs(session)
    
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)
        
    if not os.path.exists(tensorboard):
        os.makedirs(tensorboard)