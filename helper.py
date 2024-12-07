import cv2
import numpy as np
import pandas as pd
import json
import os


class Label:
    def __init__(self, name, id, csId, csTrainId, level4id, level3Id, category, level2Id, level1Id, hasInstances, ignoreInEval, color):
        self.name = name
        self.id = id
        self.csId = csId
        self.csTrainId = csTrainId
        self.level4id = level4id
        self.level3Id = level3Id
        self.category = category
        self.level2Id = level2Id
        self.level1Id = level1Id
        self.hasInstances = hasInstances
        self.ignoreInEval = ignoreInEval
        self.color = color

# Your label definitions here (from your provided list)

labels = [
    #       name                     id    csId     csTrainId level4id        level3Id  category           level2Id      level1Id  hasInstances   ignoreInEval   color
    Label(  'road'                 ,  0   ,  7 ,     0 ,       0   ,     0  ,   'drivable'            , 0           , 0      , False        , False        , (128, 64,128)  ),
    Label(  'parking'              ,  1   ,  9 ,   255 ,       1   ,     1  ,   'drivable'            , 1           , 0      , False        , False         , (250,170,160)  ),
    Label(  'drivable fallback'    ,  2   ,  255 ,   255 ,     2   ,       1  ,   'drivable'            , 1           , 0      , False        , False         , ( 81,  0, 81)  ),
    Label(  'sidewalk'             ,  3   ,  8 ,     1 ,       3   ,     2  ,   'non-drivable'        , 2           , 1      , False        , False        , (244, 35,232)  ),
    Label(  'rail track'           ,  4   , 10 ,   255 ,       3   ,     3  ,   'non-drivable'        , 3           , 1      , False        , False         , (230,150,140)  ),
    Label(  'non-drivable fallback',  5   , 255 ,     9 ,      4   ,      3  ,   'non-drivable'        , 3           , 1      , False        , False        , (152,251,152)  ),
    Label(  'person'               ,  6   , 24 ,    11 ,       5   ,     4  ,   'living-thing'        , 4           , 2      , True         , False        , (220, 20, 60)  ),
    Label(  'animal'               ,  7   , 255 ,   255 ,      6   ,      4  ,   'living-thing'        , 4           , 2      , True         , True        , (246, 198, 145)),
    Label(  'rider'                ,  8   , 25 ,    12 ,       7   ,     5  ,   'living-thing'        , 5           , 2      , True         , False        , (255,  0,  0)  ),
    Label(  'motorcycle'           ,  9   , 32 ,    17 ,       8   ,     6  ,   '2-wheeler'           , 6           , 3      , True         , False        , (  0,  0,230)  ),
    Label(  'bicycle'              , 10   , 33 ,    18 ,       9   ,     7  ,   '2-wheeler'           , 6           , 3      , True         , False        , (119, 11, 32)  ),
    Label(  'autorickshaw'         , 11   , 255 ,   255 ,     10   ,      8  ,   'autorickshaw'        , 7           , 3      , True         , False        , (255, 204, 54) ),
    Label(  'car'                  , 12   , 26 ,    13 ,      11   ,     9  ,   'car'                 , 7           , 3      , True         , False        , (  0,  0,142)  ),
    Label(  'truck'                , 13   , 27 ,    14 ,      12   ,     10 ,   'large-vehicle'       , 8           , 3      , True         , False        , (  0,  0, 70)  ),
    Label(  'bus'                  , 14   , 28 ,    15 ,      13   ,     11 ,   'large-vehicle'       , 8           , 3      , True         , False        , (  0, 60,100)  ),
    Label(  'caravan'              , 15   , 29 ,   255 ,      14   ,     12 ,   'large-vehicle'       , 8           , 3      , True         , True         , (  0,  0, 90)  ),
    Label(  'trailer'              , 16   , 30 ,   255 ,      15   ,     12 ,   'large-vehicle'       , 8           , 3      , True         , True         , (  0,  0,110)  ),
    Label(  'train'                , 17   , 31 ,    16 ,      15   ,     12 ,   'large-vehicle'       , 8           , 3      , True         , True        , (  0, 80,100)  ),
    Label(  'vehicle fallback'     , 18   , 355 ,   255 ,     15   ,      12 ,   'large-vehicle'       , 8           , 3      , True         , False        , (136, 143, 153)),  
    Label(  'curb'                 , 19   ,255 ,   255 ,      16   ,     13 ,   'barrier'             , 9           , 4      , False        , False        , (220, 190, 40)),
    Label(  'wall'                 , 20   , 12 ,     3 ,      17   ,     14 ,   'barrier'             , 9           , 4      , False        , False        , (102,102,156)  ),
    Label(  'fence'                , 21   , 13 ,     4 ,      18   ,     15 ,   'barrier'             , 10           , 4      , False        , False        , (190,153,153)  ),
    Label(  'guard rail'           , 22   , 14 ,   255 ,      19   ,     16 ,   'barrier'             , 10          , 4      , False        , False         , (180,165,180)  ),
    Label(  'billboard'            , 23   , 255 ,   255 ,     20   ,      17 ,   'structures'          , 11           , 4      , False        , False        , (174, 64, 67) ),
    Label(  'traffic sign'         , 24   , 20 ,     7 ,      21   ,     18 ,   'structures'          , 11          , 4      , False        , False        , (220,220,  0)  ),
    Label(  'traffic light'        , 25   , 19 ,     6 ,      22   ,     19 ,   'structures'          , 11          , 4      , False        , False        , (250,170, 30)  ),
    Label(  'pole'                 , 26   , 17 ,     5 ,      23   ,     20 ,   'structures'          , 12          , 4      , False        , False        , (153,153,153)  ),
    Label(  'polegroup'            , 27   , 18 ,   255 ,      23   ,     20 ,   'structures'          , 12          , 4      , False        , False         , (153,153,153)  ),
    Label(  'obs-str-bar-fallback' , 28   , 255 ,   255 ,     24   ,      21 ,   'structures'          , 12          , 4      , False        , False        , (169, 187, 214) ),  
    Label(  'building'             , 29   , 11 ,     2 ,      25   ,     22 ,   'construction'        , 13          , 5      , False        , False        , ( 70, 70, 70)  ),
    Label(  'bridge'               , 30   , 15 ,   255 ,      26   ,     23 ,   'construction'        , 13          , 5      , False        , False         , (150,100,100)  ),
    Label(  'tunnel'               , 31   , 16 ,   255 ,      26   ,     23 ,   'construction'        , 13          , 5      , False        , False         , (150,120, 90)  ),
    Label(  'vegetation'           , 32   , 21 ,     8 ,      27   ,     24 ,   'vegetation'          , 14          , 5      , False        , False        , (107,142, 35)  ),
    Label(  'sky'                  , 33   , 23 ,    10 ,      28   ,     25 ,   'sky'                 , 15          , 6      , False        , False        , ( 70,130,180)  ),
    Label(  'fallback background'  , 34   , 255 ,   255 ,     29   ,      25 ,   'object fallback'     , 15          , 6      , False        , False        , (169, 187, 214)),
    Label(  'unlabeled'            , 35   ,  0  ,     255 ,   255   ,      255 ,   'void'                , 255         , 255    , False        , True         , (  0,  0,  0)  ),
    Label(  'ego vehicle'          , 36   ,  1  ,     255 ,   255   ,      255 ,   'void'                , 255         , 255    , False        , True         , (  0,  0,  0)  ),
    Label(  'rectification border' , 37   ,  2  ,     255 ,   255   ,      255 ,   'void'                , 255         , 255    , False        , True         , (  0,  0,  0)  ),
    Label(  'out of roi'           , 38   ,  3  ,     255 ,   255   ,      255 ,   'void'                , 255         , 255    , False        , True         , (  0,  0,  0)  ),
    Label(  'license plate'        , 39   , 255 ,     255 ,   255   ,      255 ,   'vehicle'             , 255         , 255    , False        , True         , (  0,  0,142)  ),
    
]           

      

# Function to get label information from pixel color
def get_label_by_color(color):
    for label in labels:
        if label.color == tuple(color):  # Compare the pixel color to the label color
            return label
    return None

# Function to convert the segmented image into polygon-based format and prepare for CSV
def image_to_polygon_format(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path '{image_path}' not found.")
    
    segmented_image = cv2.imread(image_path)  # Load the segmented image
    if segmented_image is None:
        raise ValueError(f"Failed to load image at '{image_path}'. Make sure the file is a valid image.")

    height, width, _ = segmented_image.shape

    objects = []  # To hold the polygon data for each object

    # Loop through all the labels
    for label in labels:
        if label.ignoreInEval:
            continue  # Skip labels that should be ignored

        # Convert the label color to a binary mask
        mask = cv2.inRange(segmented_image, np.array(label.color), np.array(label.color))

        # Find contours (polygons) in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) < 3:  # Skip small or invalid polygons
                continue

            # Simplify the contour to polygon format
            polygon = contour.reshape(-1, 2).tolist()  # Flatten the contour to a list of points

            # Create the object data for the current label and polygon
            object_data = {
                "label": label.name,
                "polygon": polygon
            }

            # Append the object to the result
            objects.append(object_data)

    return objects

# Function to save to the CSV format as requested
def save_to_csv(objects_dict, output_csv_path):
    # Create a DataFrame with id and objects
    data = [{"id": filename, "objects": json.dumps(objects)} for filename, objects in objects_dict.items()]
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)

# Main function to process multiple images and save the output CSV
def process_images(image_folder, output_csv_path):
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder '{image_folder}' not found.")
    
    objects_dict = {}

    # Process each image and store the objects for each row ID
    segmented_image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    for image_path in segmented_image_paths:
        # Get the filename without the extension
        filename = os.path.splitext(os.path.basename(image_path))[0].replace('_gtFine_labelColors', '')
        polygon_data = image_to_polygon_format(image_path)
        objects_dict[filename] = polygon_data

    # Save the results to CSV
    save_to_csv(objects_dict, output_csv_path)

# Entry point for using the helper in other scripts or importing it as a module
def main(image_folder=None, output_csv_path=None):
    if image_folder is None or output_csv_path is None:
        print("Please provide valid image folder and output CSV path when importing this module.")
        return
    try:
        process_images(image_folder, output_csv_path)
        print(f"Solution file saved to {output_csv_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Default paths for testing; these should be overridden when imported as a module
    main("predicted_images", "submission.csv")