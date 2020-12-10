# food_recognition_proj
Create deep learning models to recognize 4 types of food: Hamburger, Pizza, Ramen and Sushi from images

The images and 4 CNN models can be downloaded in this link: https://drive.google.com/drive/folders/1iIDctjeGEmXZvykWMAaUha2LtfwGAqFI?usp=sharing 

You can unzip the images and 4 CNN models and then put them under the folder: "food_recognition_proj". 

You can find the structure of folder "food_recognition_proj" in "file_structure.png". 

Note:  
    To avoid file structure confusion.      
    just under "images" folder, you can see folder "train" and "test", whcih contains images for training and testing.     
    just under each of the model folders, such as "Inception_model_03", you can find "assets", "variables" and "saved_model.pb".     

Framework for this project:      
Tensorflow 2.3.0.   
Training is done using googe colab for the GPU.     

Run Demo    
Demo images are in folder "demo_images" and "online_images". Images in demo_images are selected from test data. Images in online_images folder are selected by me from google search       
Two methods to run the demo file,       
    Method 1:  under the folder "food_recognition_proj", type the following command "python demo.py --input {image_folder}".    
                For example: "python demo.py --input demo_images" or command "python demo.py --input online_images"   
    Method 2:  under the folder "food_recognition_proj", type the following command "python demo.py --input {image_path}".  
                For example: "python demo.py --input demo_images/3545.jpg"  or   "python demo.py --input online_images/hamburger01.jpg"   
   
Dataset details      
    The images for training and testing are from this kaggle project: https://www.kaggle.com/kmader/food41 . There are 101 caegories in this dataset    
    I select four categories: Hamburger, Pizza, Ramen, and Sushi from this those 101 categories. Each category has 1000 images. 

Data Cleaning   
I made the following data cleaning process:      
        1. Remove wrong labels: In the four categories, some of them are wrong labels. I removed them. So each has about 960 images left.   
        2. Split images into train data and test data. For each category, I randomly select 150 as test images, the rest about 800 images as train images   
    
Build models    
The fine-tuning code is in file "model_creation.ipynb". I used transfer learning to creat 4 models based on the following CNN models:   
        * InceptionV3   
        * ResNet152V2   
        * Xception  
        * InceptionResNetV2 
    
Test result      
In file "model_selection_and_test.ipynb", I test the four models. From the 4 models I select top 3 models (InceptionV3, ResNet152V2 and Xception) and then perform majority votting.    
For each of the 4 models, the F1-score is about 0.98. After majority votting, the F1-score is 0.99.  

