import pandas as pd                                      
from sklearn.cluster import KMeans                       

data = pd.read_csv("Mall_Customers.csv")                 
X = data[["Annual Income (k$)","Spending Score (1-100)"]]  

model = KMeans(n_clusters=5, n_init=10, random_state=1)  
model.fit(X)                                             
data["Group"] = model.labels_                            

print(data["Group"].value_counts()) 
new_customer = [[75, 85]]                                
group = model.predict(new_customer)[0]                   
print("This customer belongs to group:", group) 
