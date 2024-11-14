import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        # if confused on this, watch video on custom dataset loading
        return len(self.annotations)
    
    def __getitem__(self, index):
        # Gets the path of the label
        # It gets the 1st column at index of the csv file
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        # Convert labels into data
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                
                boxes.append([class_label, x, y, width, height])

        # Conver Images to data
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        # Convert box to tensors in case we're transforming
        boxes = torch.tensor(boxes)

        # If transforming
        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            # Converts x and y to the correct grid location
            # Right now, x and y are a percentage of the width or height
            # For example, if x = 0.5, and y = 5, its in the middle of the image
            # By multiplying by S and flooring it, we get what box its in
            i, j = int(self.S * y), int(self.S * x)
            # Location relative to the cell
            x_cell, y_cell = self.S * X - j, self.S * y - i
            # Width relative to cell size
            width_cell, height_cell = (
                width * self.S,
                height * self.S
            )

            if label_matrix[i, j, 20] == 0: # if there's no object in i j already
                # Sets an item in i j
                label_matrix[i, j, 20] = 1  
                # Calculate coordinates for label
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                # Set coordinates of label
                label_matrix[i, j, 21:25] = box_coordinates
                # Set which class it is
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
