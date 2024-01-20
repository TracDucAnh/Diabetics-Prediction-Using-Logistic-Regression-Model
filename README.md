# Diabetics-Prediction-Using-Logistic-Regression-Model
A Research Project on Applying Logistic Regression to Predict Result of Diabetes Diagnosis
## Introduction
The rise of Logistic Regression in machine learning stemmed from the need for a model capable of predicting categorical outcomes. Typically, these involve binary values like 0 and 1, representing opposing states like true/false, benign/malignant, or pregnant/not pregnant. Mastering Logistic Regression and its implementation paves the way for accurate solutions in classification problems, particularly those with binary outputs.

Diabetes mellitus encompasses a group of diseases disrupting the body's blood sugar utilization. Chronic diabetes, including types 1 and 2, often presents with similar symptoms that can go undiagnosed for years. These symptoms, which may appear suddenly, can include frequent urination, weight loss, thirst, fatigue, and vision changes. Regardless of the type, all diabetic conditions lead to an excess of blood glucose due to insulin deficiency, a hormone crucial for converting glucose into energy. High blood sugar poses a significant risk for various serious health complications.

Early diagnosis is crucial, yet traditional methods such as Fasting Plasma Glucose (FPG), Oral Glucose Tolerance Test (OGTT), or Glycated Hemoglobin (HbA1c) often prove cumbersome. This research delves into the potential of Logistic Regression to accurately predict diabetes based on readily available data, potentially paving the way for more accessible and reliable diagnoses. Notably, predicting type 1 diabetes also relies on the Logistic Regression algorithm, aiming to estimate the probability of a patient receiving a diagnosis. This problem demands a binary output, where 1 indicates a type 1 diabetes diagnosis and 0 signifies its absence.
## About dataset
### Overview:
The dataset was collected by the National Institute of Diabetes and Digestive and Kidney Diseases. This dataset is a part of a larger dataset of the Pima Indian Diabetes Database. This dataset only focuses on female Pima Indian heritage (a subgroup of Native Americans) patients above 21 years old.

![Alt text](https://scontent.fsgn12-1.fna.fbcdn.net/v/t1.15752-9/416094909_272934112474072_7879662365743638370_n.png?_nc_cat=110&ccb=1-7&_nc_sid=8cd0a2&_nc_eui2=AeGvbd5Y8F5Qm0pPHW5KNJTOtI6MhHtE9QC0joyEe0T1AIJ-kS4lGgfKhY3u4rG3BLa5X2Ho92hTsRiNERBE7gPJ&_nc_ohc=pDj62d3cqHkAX_IjRXb&_nc_ht=scontent.fsgn12-1.fna&oh=03_AdTR8iou-iWUzDpr0f_-eh69O24bDBfVTFLqoPi_kVBSOQ&oe=65D341DE "Overview")

The dataset was collected via: <https://www.kaggle.com/datasets/kandij/diabetes-dataset>

You can check the dataset in more detail when running Diabetes_patients_report.py, the program will return Diabetes_patents_report.html and you can see the statistics in your browsers.

## How to run the program?

I create a repository on github where I store and share my code. The repository contains 1 folder that includes the dataset and 5 files, just focusing on Diabetes_Prediction.py because that file is where we implement the Logistic Regression model. Running the program is simple, you download all files in this repository and use any IDE for running your Python program, make sure you satisfy the requirement in requirement.txt, the version of libraries and packages should be latest. As mentioned above, Diabetes_patients_report.py is just a program to explore the dataset.

## Results

Having run the model many times with effort, I have found the highest accuracy that the model performs is approximately 80.5%. This might not be the best result that the model could return thus you can improve the model with your knowledge and experience. I believe that changing the learning rate, the iteration, or  the random state in the code could explore more surprising results. Another approach also improves the performance as well.

This is the plot showing the accuracy and the cost after 10 times changing the learning rate alpha of the original code:
<div style="display: flex; justify-content: center;">
    <img 
        style="display: block; 
               margin-left: auto;
               margin-right: auto;
               width: 50%;"
        src="https://scontent.xx.fbcdn.net/v/t1.15752-9/413489702_727311096185996_1394957675497422875_n.png?_nc_cat=107&ccb=1-7&_nc_sid=510075&_nc_eui2=AeEhzj9nO_zRf8uQ_nrKL-tIdaAjxtM2WO51oCPG0zZY7k_BkWFdArz1aegH_UWJfses7WywG0umMt-WvnsBti4M&_nc_ohc=3fxwpVPeqtMAX-hRYLV&_nc_ad=z-m&_nc_cid=0&_nc_ht=scontent.xx&oh=03_AdTfDqaPkHgJv919Jd4i95q5bFMJIWLX6KFL4ZmOWK2h1g&oe=65D36B00" 
        alt="Accuracy/Cost according to learning rate">
    </img>
</div>

## More information

This research project is inspired respectfully by a book by associate professor Tho Quan, "Mang No Ron Nhan Tao Tu Hoi Quy Den Hoc Sau". However, the project focuses on explaining and implementing the Logistic Regression algorithm without building any neural network for the sake of simplicity and consistency.

I'm happy to share with 
