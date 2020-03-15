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
    global new_syms
    dataset=pd.read_csv('symptom.csv')

    X=dataset.iloc[:,:-1].values
    y=dataset.iloc[:,-1].values

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
    if(input.find('remedies')==-1 and input.find('yes')==-1 and input.find('no')==-1):
        #input='itching,skin_rash,nodal_skin_eruptions'
        symptoms=input.split(",")

        count=dataset.groupby(symptoms)

        symptoms1=[]
        for i in dataset:
            symptoms1.append(i)

        arr = count.groups[(1,1)]


        new_data = []
        for i in arr:
            print(i)
            #print(dataset.iloc[i,:])
            new_data.append(dataset.iloc[i,:])

        print(len((new_data)))
        new_data = np.array(new_data).reshape(len(new_data),62)

        #new_data[5][-1] = 'IDK'
        unq,tags, count=np.unique(new_data[:,-1], return_inverse=1, return_counts=1)
        new_syms=[]
        for i in arr:
            new_syms.append(np.array(dataset.iloc[i,:]))
        df = pd.DataFrame(new_syms)
        np.asarray(new_syms)

        head=[]
        for i in range(0,62):
            if new_syms[0][i]==0 and new_syms[1][i]==1:
                print(i)
                print(dataset.columns[i])
                head.append(dataset.columns[i])
        if(iter1==0):
            return("DO YOU HAVE SYMPTOM: " + head[0])
        elif(iter1==1):
            return ("YOU HAVE " + unq[0])
        return("Give me your symptoms")
    if(input=="yes"):
        return("You may have " + new_syms[0][-1])
    if(input=="no"):
        return("You may have " + new_syms[1][-1])

    i=-1
    df = pd.read_csv('RTH.csv')
    data = df['Disease']
    #input = bytes.decode(input)
    disease = input.split(" ")
    data1 = disease[0]
    print(disease[0])
    for d in data:
        i=i+1
        if data1==d:
            print("yes")
            str=''
            str+=data1
            print(str)
            print(df.iloc[i,2])
            return(df.iloc[i,2])
            
        

#print(chatbot('itching,skin_rash,nodal_skin_eruptions'))
