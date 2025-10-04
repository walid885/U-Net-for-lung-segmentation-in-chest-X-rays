import os

def explore_data_structure(data_dir):
    print(f"Exploring: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory {data_dir} does not exist!")
        return
    
    subdirs = ['CXR_png', 'ManualMask', 'ClinicalReadings']
    
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.exists(subdir_path):
            print(f"\n{subdir}:")
            files = os.listdir(subdir_path)
            print(f"  Total files: {len(files)}")
            print(f"  First 5 files: {files[:5]}")
            
            png_files = [f for f in files if f.endswith('.png')]
            print(f"  PNG files: {len(png_files)}")
            
            if subdir == 'ManualMask':
                left_masks = [f for f in files if 'left' in f.lower()]
                right_masks = [f for f in files if 'right' in f.lower()]
                print(f"  Left masks: {len(left_masks)}")
                print(f"  Right masks: {len(right_masks)}")
                print(f"  Sample left: {left_masks[:3] if left_masks else 'None'}")
                print(f"  Sample right: {right_masks[:3] if right_masks else 'None'}")
        else:
            print(f"\n{subdir}: NOT FOUND")

if __name__ == "__main__":
    data_dir = "../Data/NLM-MontgomeryCXRSet/MontgomerySet"
    explore_data_structure(data_dir)