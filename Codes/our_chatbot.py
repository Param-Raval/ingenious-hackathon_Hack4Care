#!/usr/bin/env python
# coding: utf-8

# In[17]:

# In[1]:

    
# Load libraries
def chatbot(input, iter1):

    import pydotplus
    from IPython.display import Image  
    from sklearn.datasets import load_iris
    from sklearn import tree
    import collections
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
    from sklearn.model_selection import train_test_split # Import train_test_split function
    from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

    global dataset
    dataset=pd.read_csv('Training.csv')

    X=dataset.iloc[:,:-1].values
    y=dataset.iloc[:,132].values

    if(iter1==0):

        # In[2]:


        

        # In[3]:


        y


        # In[4]:


        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


        # In[5]:


        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)


        # In[6]:


        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)


        # In[15]:


        tree.plot_tree(classifier.fit(X_train,y_train))


        # In[16]:
        import graphviz
        dot_data = tree.export_graphviz(classifier, out_file=None,class_names=y,  
        filled=True, rounded=True,special_characters=True) 
        graph = graphviz.Source(dot_data)

#    Image(graph.create_png())
    from io import BytesIO

    print(input)
    input = bytes.decode(input)
    #input='itching,skin_rash,nodal_skin_eruptions'
    symptoms=input.split(",")

    count=dataset.groupby(symptoms)

    symptoms1=[]
    for i in dataset:
        symptoms1.append(i)

    arr = count.groups[(1,1,1)]

    print(dataset.iloc[820,:])

    new_data = []
    for i in arr:
        print(i)
        #print(dataset.iloc[i,:])
        new_data.append(dataset.iloc[i,:])

    print(len((new_data)))
    new_data = np.array(new_data).reshape(len(new_data),133)

    #new_data[5][-1] = 'IDK'
    unq,tags, count=np.unique(new_data[:,-1], return_inverse=1, return_counts=1)

    if(iter1==1):
        return("DO YOU HAVE SYMPTOM: " + unq[0])
    else:
        return ("YOU HAVE " + unq[0])

    

#print(chatbot('itching,skin_rash,nodal_skin_eruptions'))
