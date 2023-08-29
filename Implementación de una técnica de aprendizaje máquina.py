#Import libraries and dataset
import pandas as pd
from sklearn.datasets import load_wine
import numpy as np
#Function to split the data in train and test
from sklearn.model_selection import train_test_split

#Load de dataset
wine = load_wine()
features = pd.DataFrame(wine.data, columns=wine.feature_names)
target = pd.Series(wine.target, name="target")
df = pd.concat([features, target], axis=1)

#Division de los datos
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

#Tree classificator model

#Counts the number of distinct types of target that we have on our df
def class_counts(df):
    counts = df["target"].value_counts().to_dict()
    return counts

#In order to split the data we have to ask a question, to know what conditional we are going to use to split our data
class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value
    #Returns the values that are above a condition
    def match(self, example):
        val = example[self.column]
        return val >= self.value
    #Function that return the partition (question over a feature and limit value)
    def __repr__(self):
        condition = ">="
        return "Is %s %s %s?" % (self.column, condition, str(self.value))

#This function divide our df into the rows that satisfy the condition and the ones that does not
def partition(df, question):
    true_df = []
    false_df = []
    for ind, row in df.iterrows():
        test = question.match(df.loc[ind])
        if test:
            true_df.append(df.loc[ind])
        else:
            false_df.append(df.loc[ind])
    return true_df,false_df

#This function calculate the gini of the node, retriving the impurity of the node
def gini(df):
    counts = class_counts(df)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(df.index))
        impurity -= prob_of_lbl**2
    return impurity

#This node calculate the information gain with a partition
def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

#This function finds the best partition based on the amount of information gained
def find_best_split(df):
    best_gain = 0  
    best_question = None  
    current_uncertainty = gini(df)

    labels = df.columns.tolist()
    for col in range(0, len(labels)-1): 

        for ind in range(0,len(df.index)):  

            question = Question(labels[col], df.iloc[ind,col])

            true_rows, false_rows = partition(df, question)

            #If the partition did not make a effect it continues with the next
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            true_rows = pd.DataFrame(true_rows, columns = df.columns.tolist())
            false_rows = pd.DataFrame(false_rows, columns = df.columns.tolist())
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

#This function let us know how many targets are predicted by a leaf
class Leaf:
    def __init__(self, df):
        self.predictions = class_counts(df)
 
class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

#Functions that builds the tree
def build_tree(df):
    gain, question = find_best_split(df)

    #In case we found a leaf
    if gain == 0:
        return Leaf(df)

    true_rows, false_rows = partition(df, question)
    true_rows = pd.DataFrame(true_rows, columns = df.columns.tolist())
    false_rows = pd.DataFrame(false_rows, columns = df.columns.tolist())
    
    #Recursive functions iterating either the positive outcome or negative, in order to find the leafs
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)

#Function to print the tree decisions
def print_tree(node, spacing=""):

    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    print (spacing + str(node.question))

    print (spacing + "--> True:")
    print_tree(node.true_branch, spacing + "  ")

    print (spacing + "--> False:")
    print_tree(node.false_branch, spacing + "  ")

#Function that will help us test the tree
def classify(df, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

#Percentage of precission inside a leaf
def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


#Build the tree with the train data
print("Starting to build... \n")
my_tree = build_tree(df_train)
#Print the results
#The predicted results are leafs
print_tree(my_tree)
print("Done :)")
#Test the tree with the test data
target_exp = df_test["target"]
ratios = []
for ind, row in df_test.iterrows():
    ratios.append([target_exp[ind],print_leaf(classify(df_test, my_tree))])

ratios = pd.DataFrame(ratios, columns = ["Actual","Predicted"])

#View the results
print(ratios.head(100))