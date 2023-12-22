from PIL import Image
import os
os.sys.path.append("")
from tqdm import tqdm
def get_centered_square(image):
    width, height = image.size
    size = min(width, height)
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2
    square = image.crop((left, top, right, bottom))
    return square

dataset_folder = 'source_data'
cnt=0
for root, dirs, files in tqdm(os.walk(dataset_folder)):
    for file in files:
        # if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
        if file.lower().endswith(('.bmp')):
            cnt+=1
            image_path = os.path.join(root, file)
            img = Image.open(image_path)
            
            # Get the centered square
            centered_square = get_centered_square(img)
            
            # Resize the centered square to 256x256 pixels
            centered_square = centered_square.resize((256, 256), Image.LANCZOS)

            centered_square = centered_square.convert('RGB')
            
            # Save the resized centered square over the original image
            new_root = root.replace("source_data","data")
            os.makedirs(new_root,exist_ok=True)
            centered_square.save(os.path.join(new_root, file), quality=80)
            
            # Close the opened image
            print(cnt,end="\r")
            img.close()