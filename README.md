# 🟢 Customer Segmentation Dashboard – K-Means Clustering

The **Customer Segmentation Dashboard** is a Machine Learning web application that groups customers into **distinct segments** using the **K-Means Clustering algorithm**.  
The system identifies customers with similar purchasing behavior and presents the results through an **interactive and business-friendly dashboard**.

---

## 📌 Project Objectives

- Segment customers based on purchasing behavior  
- Demonstrate the working of **K-Means Clustering (Unsupervised Learning)**  
- Help users discover **hidden customer groups without predefined labels**  
- Provide **clear business interpretations** of customer segments  
- Build an **interactive and visual web application** using Streamlit  

---

## 🧠 Machine Learning Approach

### 🔹 Algorithm Used
- K-Means Clustering  

### 🔹 Key Concepts
- Unsupervised Machine Learning  
- Feature Scaling using **StandardScaler**  
- Distance-based clustering  
- Cluster centroid optimization  

The K-Means algorithm groups customers by minimizing the distance between data points and their respective cluster centers.

---

## 📊 Dataset Information

The system uses the **Wholesale Customers Dataset**, which contains annual spending data of customers across multiple product categories.

### 📁 Input Features
- Fresh  
- Milk  
- Grocery  
- Frozen  
- Detergents_Paper  
- Delicassen  

These features represent customer spending patterns and purchasing behavior.

---

## ⚙️ Application Features

### 🔹 Clustering Controls (Sidebar)
- Select **minimum two numerical features** for clustering  
- Choose the **number of clusters (K)** using a slider (range: 2–10)  
- Optional **random state** input for reproducibility  
- **Run Clustering** button to explicitly execute the algorithm  

This design helps users understand that clustering is an **action-based process**.

---

## 📈 Customer Clusters Visualization

- 2D scatter plot based on selected features  
- Different colors represent different customer clusters  
- **Cluster centers** are clearly marked using larger symbols  
- Visualization updates dynamically based on user inputs  

---

## 📋 Cluster Summary

The dashboard provides a detailed summary table containing:
- Cluster ID  
- Number of customers in each cluster  
- Average values of selected features per cluster  

This summary helps business users quickly understand the composition of each customer segment.

---

## 💡 Business Interpretation

Each cluster is explained using **non-technical language**, for example:
- High-spending customers across specific categories  
- Budget-conscious customers with lower overall spending  
- Moderate customers with selective purchasing behavior  

These insights can be used for **targeted marketing**, **customer retention**, and **business strategy planning**.

---

## 🧭 User Guidance

> Customers in the same cluster exhibit similar purchasing behaviour  
> and can be targeted with similar business strategies.

---

## 🛠️ Technologies Used

- Python  
- Streamlit  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## ▶️ How to Run the Application

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
