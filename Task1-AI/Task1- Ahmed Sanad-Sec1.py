# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 09:28:55 2021

@author: workstation
"""
# (1) Reading File
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("Wuzzuf_Jobs.csv")

# (2) Displaying Structure
dataset.describe()

# sorting by title
dataset.sort_values("Title", inplace = True)

# (3) dropping ALL duplicate values
dataset.drop_duplicates(subset ="Title", keep = "first", inplace = True)
# there is no null values in the datatset


# (4) Counting Jobs For Each Company
x=dataset['Company'].value_counts()
print(x)
print("the most demanding companies for jobs is :  Confidential " )

# (5) Pie Chart Of Step (4)

plt.pie(x)
plt.show() 

# (6) Most Popular Job titles
y=dataset['Title'].value_counts()
print(y)
print("the most popular job title is :  Accountant " )


# (7) Bar Chart Of Step (6)
titles = list(dataset['Title'])
valuees = list(dataset['Title'].value_counts())
# creating the bar plot
plt.bar(titles, valuees, color ='maroon')

plt.xlabel("Job Title")
plt.ylabel("No. of employees")
plt.title("Most Popuular Job Title")
plt.show()


# (8) Most Popular Areas
z=dataset['Country'].value_counts()
print(z)
print("the most popular job title is :  Accountant " )

# (9) Bar Chart Of Step (8)
countries = list(dataset['Country'])
results = list(dataset['Country'].value_counts())
# creating the bar plot
plt.bar(countries, results, color ='maroon')

plt.xlabel("Job area")
plt.ylabel("No. of jobs")
plt.title("Most Popuular Areas")
plt.show()


# (10) Most Important Skills
f=dataset['Skills'].value_counts()
print(f)
print("the most important skills for jobs are :  Corporate Sales, Real Estate, Advertising, Marketing, Sales Skills, Insurance, Sales Target, Outdoor Sales, Telesales, Property ")












