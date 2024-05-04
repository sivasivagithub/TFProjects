
import os
import zipfile


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

train_cat_fnames = os.listdir( train_cats_dir )
train_dog_fnames = os.listdir( train_dogs_dir )

print(train_cat_fnames[:10])
print(train_dog_fnames[:10])

#['cat.829.jpg', 'cat.189.jpg', 'cat.432.jpg', 'cat.647.jpg', 'cat.295.jpg', 'cat.496.jpg', 'cat.974.jpg', 'cat.301.jpg', 'cat.608.jpg', 'cat.946.jpg']
#['dog.160.jpg', 'dog.216.jpg', 'dog.848.jpg', 'dog.99.jpg', 'dog.936.jpg', 'dog.67.jpg', 'dog.995.jpg', 'dog.634.jpg', 'dog.461.jpg', 'dog.791.jpg']

print('total training cat images :', len(os.listdir(      train_cats_dir ) ))
print('total training dog images :', len(os.listdir(      train_dogs_dir ) ))

print('total validation cat images :', len(os.listdir( validation_cats_dir ) ))
print('total validation dog images :', len(os.listdir( validation_dogs_dir ) ))

