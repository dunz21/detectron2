import os
import pandas as pd

# This function will parse the image filename and extract the attributes
def parse_image_name(image_name):
    # Assuming the format is img_{ID}_{frame}_{x1}_{y1}_{x2}_{y2}.ext
    parts = image_name.split('_')
    if len(parts) >= 6 and parts[0] == 'img':
        _, id_, frame_, x1_, y1_, x2_, y2_ = parts[:7]
        # Extract the actual numbers and handle file extensions
        id_ = int(id_)
        frame_ = int(frame_)
        x1_ = int(x1_)
        y1_ = int(y1_)
        x2_ = int(x2_)
        y2_ = int(y2_.split('.')[0])  # Remove the file extension
        return id_, frame_, x1_, y1_, x2_, y2_
    else:
        # If the format does not match, return None
        return None

# Function to read through the folders and files, and create the dataframe
def create_dataframe_from_images(folder_name):
    data = []  # List to hold all the data
    for subdir in os.listdir(folder_name):
        subdir_path = os.path.join(folder_name, subdir)
        if os.path.isdir(subdir_path):
            # Iterate through files in each subfolder
            for filename in os.listdir(subdir_path):
                if filename.startswith("img_") and filename.endswith(".png"):
                    parsed_data = parse_image_name(filename)
                    if parsed_data:
                        id_, frame_, x1_, y1_, x2_, y2_ = parsed_data
                        data.append({
                            'name': os.path.basename(subdir),
                            'id': id_,
                            'frame': frame_,
                            'x1': x1_,
                            'y1': y1_,
                            'x2': x2_,
                            'y2': y2_
                        })
    # Convert the list of data into a pandas DataFrame
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = create_dataframe_from_images('CONCEPCION_CH1_PARTE10.6_160_0.95_images_subframe')
    df.to_excel('images_info.xlsx', index=False)
    df.head()  # Show the dataframe created