import os
import zipfile

import matplotlib

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

local_zip = "C:\\Users\\sivas\\Documents\\TensorFlowStudy\\cats_and_dogs_filtered.zip"

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('C:\\Users\\sivas\\Documents\\TensorFlowStudy')

zip_ref.close()

base_dir = ("C:\\Users\\sivas\\Documents\\TensorFlowStudy\\cats_and_dogs_filtered")

# C:\Users\sivas\Documents\TensorFlowStudy\cats_and_dogs_filtered

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

print(train_cat_fnames[:10])
print(train_dog_fnames[:10])

# ['cat.829.jpg', 'cat.189.jpg', 'cat.432.jpg', 'cat.647.jpg', 'cat.295.jpg', 'cat.496.jpg', 'cat.974.jpg', 'cat.301.jpg', 'cat.608.jpg', 'cat.946.jpg']
# ['dog.160.jpg', 'dog.216.jpg', 'dog.848.jpg', 'dog.99.jpg', 'dog.936.jpg', 'dog.67.jpg', 'dog.995.jpg', 'dog.634.jpg', 'dog.461.jpg', 'dog.791.jpg']

print('total training cat images :', len(os.listdir(train_cats_dir)))
print('total training dog images :', len(os.listdir(train_dogs_dir)))

print('total validation cat images :', len(os.listdir(validation_cats_dir)))
print('total validation dog images :', len(os.listdir(validation_dogs_dir)))

# Parameters for our graph; we'll output images in a 4x4 configuration

nrows = 4
ncols = 4

pic_index = 0  # Index for iterating over images

fig = plt.gcf()
fig.set_size_inches(ncols * 8, nrows * 8)

pic_index += 8

next_cat_pix = [os.path.join(train_cats_dir, fname)
                for fname in train_cat_fnames[pic_index - 8:pic_index]
                ]

next_dog_pix = [os.path.join(train_dogs_dir, fname)
                for fname in train_dog_fnames[pic_index - 8:pic_index]
                ]

for i, img_path in enumerate(next_cat_pix + next_dog_pix):
    # Set up subplot; subplot indices start at 1

    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('ON')  # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()
