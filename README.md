# Car vs truck classification in Pytorch
- Computer Vision course final project.

- Intructed by [Dr. Shahbahrami](https://scholar.google.com/citations?user=RbUZMnEAAAAJ&hl=en "‪Asadollah Shahbahrami‬ - ‪Google Scholar‬") at [University of Guilan](https://guilan.ac.ir/en/home "University of Guilan‬").

## Dataset
I used an unofficial handheld dataset collected in Kaggle, Which is available at https://www.kaggle.com/enesumcu/car-and-truck, And re-folded it according to the following structure:

Train set:

/data/train/car

/data/train/truck

Test set:

/data/test/car

/data/test/truck

The original dataset include **393** car images and **396** truck images. Since one of the car images was broken, it was removed before the process and the car images were reduced to **392**, then 80% of each were selected for trainset and the rest 20% for testset.

**Attention**, the image in this path of original dataset  `/datasets/Datasets/car/181535.jpg` is broken!
    
## Implementation
In this project, I have developed a model which classifies cars and trucks. Instead of training a Neural Network from scratch, I used a pretrained ResNet50 model and added some Dense layers on top.

I implement two Pytorch Dataset class for train and test datas.
```python
class TrainDataset(Dataset):
  def __init__(self, transforms=None):
    self.transforms = transforms
    self.imgs_path = "/content/data/train/"
    file_list = glob.glob(self.imgs_path + "*")
    self.data = []
    for class_path in file_list:
      class_name = class_path.split("/")[-1]
      for img_path in glob.glob(class_path + "/*.jpg"):
        self.data.append([img_path, class_name])
    self.class_map = {"car" : 0, "truck": 1}
    self.img_dim = (208, 208)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    img_path, class_name = self.data[idx]
    img = cv2.imread(img_path)
    img = cv2.resize(img, self.img_dim)
    class_id = self.class_map[class_name]
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1)
    if self.transforms is not None:
      img_tensor = self.transforms(img_tensor)
    return img_tensor, class_id

class TestDataset(Dataset):
  def __init__(self, transforms=None):
    self.transforms = transforms
    self.imgs_path = "/content/data/test/"
    file_list = glob.glob(self.imgs_path + "*")
    self.data = []
    for class_path in file_list:
      class_name = class_path.split("/")[-1]
      for img_path in glob.glob(class_path + "/*.jpg"):
        self.data.append([img_path, class_name])
    self.class_map = {"car" : 0, "truck": 1}
    self.img_dim = (208, 208)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    img_path, class_name = self.data[idx]
    img = cv2.imread(img_path)
    img = cv2.resize(img, self.img_dim)
    class_id = self.class_map[class_name]
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1)
    if self.transforms is not None:
      img_tensor = self.transforms(img_tensor)
    return img_tensor, class_id
```

For each train and test Dataset classes, First i read all the images that are in the folders and give them a 0 or 1 label acording to from wich folder(cars or trucks) they were read.

If the image was read from the car folder, It would get label 0, And if it was read from the truck folder, It would get label 1.

You can do data augumantion if you want by using this code block:
```python
data_transforms = {
    'train':
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-90, 90)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ]),
    'test':
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
}
```

And then pass the `data_transforms` to `data_sets`:
```python
data_sets = {
    'train': TrainDataset(transforms=data_transforms['train']),
    'test': TestDataset(transforms=data_transforms['test'])
}
```

The model is a pretrained ResNet50 with a .5 Dropout and 6 Linear layers that each one has a .2 Dropout as fc **(fully connected layer)** for top of the model. 

I used `CrossEntropyLoss()` for criterion and `SGD` optimizer for optimizition.
```python
model = models.resnet50(pretrained=True)

model = model.cuda() if use_cuda else model
    
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    torch.nn.Dropout(0.2),
    torch.nn.Linear(num_ftrs, 1024),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(1024, 512),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(512, 256),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(256, 128),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(128, len(data_sets['train'].class_map))
)
model.fc = model.fc.cuda() if use_cuda else model.fc
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
```
A `torch.optim.lr_scheduler.StepLR` was used to decays the learning rate of each parameter group by gamma every step_size epochs [see docs here](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.StepLR) 

Example from docs:
```python
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90
# ...
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

## Results
After
- *20* epochs of train
- With batch size of *4*

and

- Learning rate of **1e-3**

The following results were obtained:

![](https://github.com/shuoros/car-vs-truck-classification/blob/main/etc/acc.png)

![](https://github.com/shuoros/car-vs-truck-classification/blob/main/etc/loss.png)

## test

For the final test, 12 images were randomly read from the test set and given to the model to classify them.

The prediction of model is at the top of each image.

If that prediction is true the word "True" would be printed next to it and if it is not true the word "False" would be printed next to it.

![](https://github.com/shuoros/car-vs-truck-classification/blob/main/etc/result.png)

## Download
You can downlad weights of model in a h5 file with accuracy of 96.17% in testset and 92.71% in trainset from [here](https://drive.google.com/file/d/1-nk2p52PrcxMpPjf4a_HCOyIC3gzHDbV/view?usp=sharing)
