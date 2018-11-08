# Tfrecord detection augmentor

An implementation of Tfrecord in muti-oriented detection task when data augmentation is needed. The bounding box annotation here is x1,y1,x2,y2,x3,y3,x4,y4 which is often used in text detection.

### Requirements

- Python2.7
- TensorFlow1.0

### Annotaions

One image corresponds to a txt file. The annotations format in txt file is as follows:

```
   x1,y1,x2,y2,x3,y3,x4,y4,label_name
   x1,y1,x2,y2,x3,y3,x4,y4,label_name
```

### Dataset Path
You should prepare both train and val datasets. The file struture should be like this:

```
-$ROOT_PATH
	-Dataset_train
		-JPEGImages
			-your images
		-Annotations
			-your txt_file	
	-Dataset_val
		-JPEGImages
			-your images
		-Annotations
			-your txt_file
```

### Usage

1) First, you should set the right path in `config.py`. 

2) Create `.tfrecord` files.
```
python txt_to_tfrecord.py --type train
python txt_to_tfrecord.py --type val
```

3) When training, use the function `next_batch` in `read_tfrecord.py`.
