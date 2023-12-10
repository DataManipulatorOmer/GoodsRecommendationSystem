#!/usr/bin/env python
# coding: utf-8

# In[2]:


#LIBRARIES


# In[1]:


pip install folium


# In[2]:


pip install haversine


# In[3]:


import tkinter as tk
from sklearn.impute import SimpleImputer
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tkinter import scrolledtext
from tkinter import messagebox
import os
from haversine import haversine
import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# Load the dataset
data_df = pd.read_csv("goods_transport_data (3) (2).csv")

# Basic Information
print(data_df.info())


# In[24]:


# SUMMARY STATISTICS

print(data_df.describe())

#WHAT DOES THIS PROVIDES:

#Summary statistics provide a concise overview of the main characteristics and distribution of a dataset.


# In[25]:


# Check for missing values

print(data_df.isnull().sum())


# In[10]:


# Explore unique values in categorical columns

categorical_columns = ["Source City", "Destination City", "Transport Mode", "Optimal Method", "Intermediate City"]
for col in categorical_columns:
    
    print("Unique", col, "s:", data_df[col].nunique())
    
    print(data_df[col].value_counts())
    
    print()

#The number of unique values in categorical columns can provide useful insights
#It indicates how many distinct categories or groups there are for that variable.


# In[11]:


# Explore numerical columns
numerical_columns = ["Distance (km)", "Cost", "Duration (days)", "Capacity (tons)", 
                     "Cost per KM (Roads (Truck))", "Cost per KM (Train/Air)"]

for col in numerical_columns:
    # Plot histograms to visualize the distribution of each numerical column
    plt.figure(figsize=(3, 3))
    
    sns.histplot(data_df[col], kde=True, color='orange', bins=20)
    
    plt.title(col + " Distribution")
    
    plt.xlabel(col)
    
    plt.ylabel("Frequency")
    
    plt.show()


# In[12]:


# Explore relationships between variables
# lets create scatter plots or pair plots to visualize relationships

plt.figure(figsize=(3,3))

sns.scatterplot(x="Distance (km)", y="Cost", hue="Transport Mode", data=data_df)

plt.title("Cost vs Distance by Transport Mode")

plt.show()


# In[ ]:


# Explore correlations between numerical columns

correlation_matrix = data_df[numerical_columns].corr()

plt.figure(figsize=(4,4))

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")

plt.title("Correlation Matrix")

plt.show()


# In[13]:


# Explore relationships between categorical and numerical variables using boxplots

for col in categorical_columns:
    if col != "Optimal Method":
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col, y="Cost", data=data_df)
        plt.title("Cost by " + col)
        plt.xticks(rotation=90)  # Rotate x-axis labels to 90 degrees
        plt.show()


# In[ ]:


#STORY TELLING QUESTIONS


# In[ ]:


# 1. How does the choice of "Optimal Method" vary with the "Distance (km)" of transportation?


# In[14]:


sns.boxplot(x="Optimal Method", y="Distance (km)", data=data_df)

plt.title("Optimal Method vs. Distance (km)")

plt.show()


# In[ ]:


# 3. Among the different "Transport Modes" available, which one is the most frequently used for transporting goods?


# In[15]:


plt.figure(figsize=(3,3))

sns.countplot(x="Transport Mode", data=data_df)

plt.title("Frequency of Transport Modes")

plt.show()


# In[ ]:


# 4. Are there any notable relationships between the geographical locations and the "Cost per KM" for different transport methods?
# Create a map centered on the mean latitude and longitude of the data


# In[5]:


mean_lat = data_df["Source Latitude"].mean()

mean_long = data_df["Source Longitude"].mean()

real_map = folium.Map(location=[mean_lat, mean_long], zoom_start=5)

# Add heat map layer

heat_data = [[row["Source Latitude"], row["Source Longitude"], row["Cost per KM (Roads (Truck))"]] for index, row in data_df.iterrows()]

folium.plugins.HeatMap(heat_data).add_to(real_map)

# Display the map
real_map


# In[ ]:


# 5. How do "Near Sea Port," "Near Airport," and "Near Train Route" affect the "Duration (days)" and "Cost" of transportation?


# In[17]:


#NEAR SEA PORT
sns.boxplot(x="Near Sea Port", y="Duration (days)", data=data_df)
plt.title("Duration (days) vs. Near Sea Port")
plt.show()

#NEAR AIR PORT
sns.boxplot(x="Near Airport", y="Cost", data=data_df)
plt.title("Cost vs. Near Airport")
plt.show()

#NEAR TRAIN ROUTE
sns.boxplot(x="Near Train Route", y="Duration (days)", data=data_df)
plt.title("Duration (days) vs. Near Train Route")
plt.show()


# In[18]:


# Define the cost ranges
#SINCE THE MIN WAS ABOVE 1000 AND MAX WAS LESS THAN 1000

cost_ranges = [1000, 3000, 6000, 10000]

# Categorize the "Cost" values into three bins based on the cost_ranges
data_df['Cost Range'] = pd.cut(data_df['Cost'], bins=cost_ranges, labels=["Low", "Medium", "High"])

# Calculate the average cost for each cost range
average_cost_by_range = data_df.groupby("Cost Range")["Cost"].mean().reset_index()

# Create a bar graph
plt.figure(figsize=(8, 6))
sns.barplot(x="Cost Range", y="Cost", data=average_cost_by_range, palette="muted")
plt.title("Average Cost by Cost Range")
plt.xlabel("Cost Range")
plt.ylabel("Average Cost")
plt.show()


# In[ ]:


#PROJECT


# In[20]:


import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox
import os
from haversine import haversine
import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import folium
from folium.plugins import MarkerCluster
# Load the dataset from the CSV file
data_df = pd.read_csv("goods_transport_data (3).csv")

# Label encode the "Intermediate City" column (assuming it is ordinal)
label_encoder = LabelEncoder()
data_df["Intermediate City"] = label_encoder.fit_transform(data_df["Intermediate City"])

# One-hot encode the categorical columns "Transport Mode" and "Optimal Method"
data_df_encoded = pd.get_dummies(data_df, columns=["Transport Mode", "Optimal Method"], drop_first=True)

# Train the decision tree classifier
X = data_df_encoded.drop(["Source City", "Destination City", "Optimal Method_Roads (Truck)"], axis=1)
y = data_df_encoded["Optimal Method_Roads (Truck)"]  # Use "Roads (Truck)" as the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X, y)


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
def display_accuracy():
    accuracy_label.config(text="Accuracy: " + str(round(accuracy, 2)))
    
def calculate_time(distance, method):
    # Travel time in minutes based on transport method (Truck, Train, Air, Sea)
    time_per_km = {
        "Truck": 3,
        "Train": 1,
        "Air": 0.5,
        "Sea": 2
    }
    return distance * time_per_km[method]

def calculate_distance(source_city, destination_city):
    # Function to calculate distance between two cities based on their coordinates
    # For simplicity, let's assume the data_df already contains columns "Source Latitude", "Source Longitude",
    # "Destination Latitude", and "Destination Longitude" with the coordinates of the cities
    source_coords = (data_df[data_df["Source City"] == source_city]["Source Latitude"].values[0],
                     data_df[data_df["Source City"] == source_city]["Source Longitude"].values[0])
    dest_coords = (data_df[data_df["Destination City"] == destination_city]["Destination Latitude"].values[0],
                   data_df[data_df["Destination City"] == destination_city]["Destination Longitude"].values[0])
    
    # Calculate the distance using the haversine formula
    distance = haversine(source_coords, dest_coords, unit='km')
    return distance

def on_feedback():
    feedback_window = tk.Toplevel(app)
    feedback_window.title("Feedback Form")
    feedback_window.geometry("400x300")

    feedback_label = ttk.Label(feedback_window, text="Please provide your feedback:")
    feedback_label.pack(pady=10)

    feedback_text_area = scrolledtext.ScrolledText(feedback_window, width=40, height=10, wrap=tk.WORD)
    feedback_text_area.pack(pady=5)

    def submit_feedback():
        feedback = feedback_text_area.get("1.0", tk.END).strip()
        if feedback:
            # Here you can save the feedback to a database or file for further analysis
            # For now, we will just display a message box indicating feedback submission
            messagebox.showinfo("Feedback Submitted", "Thank you for your feedback!")
            feedback_window.destroy()
        else:
            messagebox.showwarning("Missing Feedback", "Please enter your feedback before submitting.")

    submit_button = ttk.Button(feedback_window, text="Submit Feedback", command=submit_feedback)
    submit_button.pack(pady=10)

def add_random_costs(data_df):
    # Random cost per km for truck
    data_df["Cost per KM (Truck)"] = [round(random.uniform(5, 20), 2) for _ in range(len(data_df))]
    
    # Random cost per km for train
    data_df["Cost per KM (Train)"] = [round(random.uniform(3, 15), 2) for _ in range(len(data_df))]
    
    # Random cost per km for air
    data_df["Cost per KM (Air)"] = [round(random.uniform(10, 30), 2) for _ in range(len(data_df))]
    
    # Random cost per km for sea
    data_df["Cost per KM (Sea)"] = [round(random.uniform(1, 5), 2) for _ in range(len(data_df))]
    
    return data_df

# Add random costs per kilometer for each method
data_df = add_random_costs(data_df)

def find_nearby_city(destination_city):
    max_distance = 500  # Maximum distance in kilometers to consider as nearby
    destination_distance = data_df[data_df["Destination City"] == destination_city]["Distance (km)"]

    if not destination_distance.empty:
        nearby_cities = data_df[
            (data_df["Destination City"] != destination_city)
            & ((data_df["Near Sea Port"] == 1) | (data_df["Near Airport"] == 1) | (data_df["Near Train Route"] == 1))
        ]
        nearby_cities["Distance to Destination"] = abs(nearby_cities["Distance (km)"] - destination_distance.values[0])
        nearby_cities = nearby_cities[nearby_cities["Distance to Destination"] <= max_distance]

        if not nearby_cities.empty:
            return nearby_cities.iloc[0]["Destination City"]

    return None

def get_transporter_contact_details(method):
    transporters = {
        "Truck": {"Name": "Truck Transporters Corp.", "Phone": "123-456-7890", "Email": "truck@example.com", "Fixed Cost": 1000},
        "Train": {"Name": "Train Transporters Corp.", "Phone": "987-654-3210", "Email": "train@example.com", "Fixed Cost": 1500},
        "Air": {"Name": "Air Transporters Corp.", "Phone": "111-222-3333", "Email": "air@example.com", "Fixed Cost": 2000},
        "Sea": {"Name": "Sea Transporters Corp.", "Phone": "444-555-6666", "Email": "sea@example.com", "Fixed Cost": 1200}
    }
    return transporters.get(method, {})

def goods_transport_recommendation(source_city, destination_city, goods_amount, selected_basis):
    direct_route_data = data_df[
        (data_df["Source City"] == source_city) & (data_df["Destination City"] == destination_city)
    ]

    if not direct_route_data.empty:
        route_data = direct_route_data.iloc[0]

        method = route_data["Optimal Method"]
        contact_details = get_transporter_contact_details(method)

        fixed_cost = contact_details.get('Fixed Cost', 0)

        if selected_basis == "Cost":
            methods = ["Truck", "Train", "Air", "Sea"]
            overall_costs = {}

            for method in methods:
                route_data = data_df[
                    (data_df["Source City"] == source_city)
                    & (data_df["Destination City"] == destination_city)
                ].iloc[0]
                contact_details = get_transporter_contact_details(method)
                fixed_cost = contact_details.get('Fixed Cost', 0)

                if method == "Truck":
                    overall_cost = goods_amount * route_data["Cost per KM (Truck)"] * route_data["Distance (km)"] + fixed_cost
                else:
                    overall_cost = goods_amount * route_data[f"Cost per KM ({method})"] * route_data["Distance (km)"] + fixed_cost

                overall_costs[method] = overall_cost

            min_cost = min(overall_costs.values())
            optimal_methods = [method for method, cost in overall_costs.items() if cost == min_cost]

            result_text = f"The most optimal method(s) of transportation based on cost is/are {', and '.join(optimal_methods)}. \n" \
                          f"The goods will be transported from {source_city} to {destination_city}. \n" \
                          f"The overall cost will be {min_cost:.2f} \n"

        elif selected_basis == "Time":
            transport_methods = ["Truck", "Train", "Air", "Sea"]
            time_per_method = {method: calculate_time(route_data["Distance (km)"], method) for method in transport_methods}
            method = min(time_per_method, key=time_per_method.get)
            time_in_minutes = time_per_method[method]
            contact_details = get_transporter_contact_details(method)

            result_text = f"The most optimal method of transportation based on time is by {method}. \n" \
                          f"The goods will be transported from {source_city} to {destination_city}. \n" \
                          f"The estimated travel time will be {time_in_minutes:.2f} minutes. \n" \
                          f"Contact Details: \n" \
                          f"Name: {contact_details.get('Name', '')} \n" \
                          f"Phone: {contact_details.get('Phone', '')} \n" \
                          f"Email: {contact_details.get('Email', '')} \n"

        elif selected_basis == "Distance":
            distance = calculate_distance(source_city, destination_city)
            time_per_method = {method: calculate_time(distance, method) for method in ["Truck", "Train", "Air", "Sea"]}
            method = min(time_per_method, key=time_per_method.get)
            time_in_minutes = time_per_method[method]
            contact_details = get_transporter_contact_details(method)

            result_text = f"The most optimal method of transportation based on distance is by {method}. \n" \
                          f"The goods will be transported from {source_city} to {destination_city}. \n" \
                          f"The estimated distance is {distance:.2f} km. \n" \
                          f"Contact Details: \n" \
                          f"Name: {contact_details.get('Name', '')} \n" \
                          f"Phone: {contact_details.get('Phone', '')} \n" \
                          f"Email: {contact_details.get('Email', '')} \n"

        return result_text



def on_clear():
    source_city_entry.delete(0, tk.END)
    destination_city_entry.delete(0, tk.END)
    goods_amount_entry.delete(0, tk.END)
    result_text_area.delete(1.0, tk.END)

def submit_feedback():
    feedback = feedback_text_area.get("1.0", tk.END).strip()
    if feedback:
        feedback_file_path = "C:/Users/omert/OneDrive/Desktop/hehe/feedback.txt"
        
        with open(feedback_file_path, "a") as file:
            file.write(feedback + "\n")
        
        messagebox.showinfo("Feedback Submitted", "Thank you for your feedback!")
        feedback_window.destroy()
    else:
        messagebox.showwarning("Missing Feedback", "Please enter your feedback before submitting.")

def on_submit():
    source_city = source_city_entry.get()
    destination_city = destination_city_entry.get()
    goods_amount = goods_amount_entry.get()
    selected_basis = selected_basis_combobox.get()  # Get the selected basis from the combobox

    try:
        goods_amount = float(goods_amount)
        result_text = goods_transport_recommendation(source_city, destination_city, goods_amount, selected_basis)

        # Clear the previous result and display new result
        result_text_area.delete(1.0, tk.END)
        result_text_area.insert(tk.END, result_text)

    except ValueError:
        result_text = "Error: Goods amount must be a valid number."
        result_text_area.delete(1.0, tk.END)
        result_text_area.insert(tk.END, result_text)


def on_contact_us():
    contact_info = "Contact Us:\n\n" \
                  "Truck Transporters Corp.\n" \
                  "Phone: 123-456-7890\n" \
                  "Email: truck@example.com\n\n" \
                  "Train Transporters Corp.\n" \
                  "Phone: 987-654-3210\n" \
                  "Email: train@example.com\n\n" \
                  "Air Transporters Corp.\n" \
                  "Phone: 111-222-3333\n" \
                  "Email: air@example.com\n\n" \
                  "Sea Transporters Corp.\n" \
                  "Phone: 444-555-6666\n" \
                  "Email: sea@example.com"

    messagebox.showinfo("Contact Us", contact_info)

# Create the main application window
app = tk.Tk()
app.title("Goods Transport Recommendation")

# Add a title label
title_label = ttk.Label(app, text="Goods Transport Recommendation", font=("Helvetica", 16, "bold"))
title_label.grid(row=0, column=0, columnspan=4, pady=10)

# Create and place widgets
source_city_label = ttk.Label(app, text="Source City:", font=("Helvetica", 12))
source_city_label.grid(row=1, column=0, padx=10, pady=5, sticky=tk.E)

source_city_entry = ttk.Entry(app, font=("Helvetica", 12))
source_city_entry.grid(row=1, column=1, padx=10, pady=5)

destination_city_label = ttk.Label(app, text="Destination City:", font=("Helvetica", 12))
destination_city_label.grid(row=2, column=0, padx=10, pady=5, sticky=tk.E)

destination_city_entry = ttk.Entry(app, font=("Helvetica", 12))
destination_city_entry.grid(row=2, column=1, padx=10, pady=5)

goods_amount_label = ttk.Label(app, text="Amount of Goods (tons):", font=("Helvetica", 12))
goods_amount_label.grid(row=3, column=0, padx=10, pady=5, sticky=tk.E)

goods_amount_entry = ttk.Entry(app, font=("Helvetica", 12))
goods_amount_entry.grid(row=3, column=1, padx=10, pady=5)

selected_basis_label = ttk.Label(app, text="Select Basis for Optimization:", font=("Helvetica", 12))
selected_basis_label.grid(row=4, column=0, padx=10, pady=5, sticky=tk.E)

selected_basis_options = ["Cost", "Time", "Distance"]
selected_basis_combobox = ttk.Combobox(app, values=selected_basis_options, state="readonly", font=("Helvetica", 12))
selected_basis_combobox.grid(row=4, column=1, padx=10, pady=5)

result_text_area = scrolledtext.ScrolledText(app, width=60, height=10, wrap=tk.WORD, font=("Helvetica", 12))
result_text_area.grid(row=5, column=0, columnspan=4, padx=10, pady=5)

submit_button = ttk.Button(app, text="Submit", command=on_submit, width=15)
submit_button.grid(row=6, column=0, padx=10, pady=5, columnspan=2, sticky=tk.E)

contact_us_button = ttk.Button(app, text="Contact Us", command=on_contact_us, width=15)
contact_us_button.grid(row=6, column=2, padx=10, pady=5, columnspan=2, sticky=tk.W)

clear_button = ttk.Button(app, text="Clear", command=on_clear, width=15)
clear_button.grid(row=7, column=0, padx=10, pady=5, columnspan=2, sticky=tk.E)

feedback_button = ttk.Button(app, text="Feedback", command=on_feedback, width=15)
feedback_button.grid(row=7, column=2, padx=10, pady=5, columnspan=2, sticky=tk.W)

app = tk.Tk()
app.title("Decision Tree Model Accuracy")

main_frame = ttk.Frame(app, padding="20")
main_frame.pack()

label_title = ttk.Label(main_frame, text=" Good Transportation's Accuracy ", font=("Helvetica", 16))
label_title.pack(pady=(0, 10))

accuracy_label = ttk.Label(main_frame, text="Accuracy: {:.2f}%".format(accuracy * 100), font=("Helvetica", 12))
accuracy_label.pack(pady=20)

#update_button = ttk.Button(main_frame, text="Calculate Accuracy", command=display_accuracy)
#update_button.pack()

app.mainloop()


# In[16]:


print("hello")


# In[ ]:




