import os
import shutil
import random

def split_dataset(base_dir, split_ratio=0.2):
    train_dir = os.path.join(base_dir, 'train')
    valid_dir = os.path.join(base_dir, 'valid')
    
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} does not exist.")
        return

    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    for cls in classes:
        cls_train_path = os.path.join(train_dir, cls)
        cls_valid_path = os.path.join(valid_dir, cls)
        
        if not os.path.exists(cls_valid_path):
            os.makedirs(cls_valid_path)
            print(f"Created validation directory for {cls}")
        
        # Count files in valid dir
        valid_files_count = len([f for f in os.listdir(cls_valid_path) if os.path.isfile(os.path.join(cls_valid_path, f))])
        
        if valid_files_count > 0:
            print(f"Skipping {cls} as it already has {valid_files_count} files in validation.")
            continue
            
        images = [f for f in os.listdir(cls_train_path) if os.path.isfile(os.path.join(cls_train_path, f))]
        num_to_move = int(len(images) * split_ratio)
        
        if num_to_move == 0 and len(images) > 0:
            num_to_move = 1
            
        print(f"Moving {num_to_move} images from {cls} train to valid...")
        
        random.shuffle(images)
        move_images = images[:num_to_move]
        
        for img in move_images:
            src = os.path.join(cls_train_path, img)
            dst = os.path.join(cls_valid_path, img)
            shutil.move(src, dst)

if __name__ == "__main__":
    dataset_path = r'c:\Users\ROSHAN MISHRA\.gemini\antigravity\playground\ethereal-cassini\dataset'
    split_dataset(dataset_path)
