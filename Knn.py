import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib as plt
import matplotlib.pyplot
import statistics

votes = pd.read_csv(r"E:\python Projects\Assigment 1 ML&BIO\house-votes-84.data.csv", header=None)

actual_Y = votes[0]  # Target variable
votes = votes.drop(columns=votes.columns[0])  # features

"""replacing for each absent vote (?) the voting decision of the majority."""
major_votes = str(votes.mode()).split()[17:]  # List of the majority of votes for each column
i = 0
for col in votes:
    votes[col] = votes[col].replace({"?": major_votes[i]})
    i = i + 1

"""assigning numerical values to categories"""
votes.replace(('n','y'),(0,1),inplace= True)
actual_Y.replace(('democrat', 'republican'), (0, 1), inplace=True)

# Create Decision Tree classifier object
dtc = DecisionTreeClassifier()

rangeList = [0.50, 0.60,0.70,0.80]  # training data sizes [50%..80%]
accuracies = []
treeSizes = []
for i in rangeList:
    temp_accuracies = []
    temp_treeSizes = []
    for x in range(5):  # repeat the experiment with five different random seeds for each training set size
        X_train, X_test, y_train, y_test = train_test_split(votes, actual_Y,shuffle=True,train_size=i)
        dtc = dtc.fit(X_train, y_train)
        predicted_y = dtc.predict(X_test)
        temp_accuracies.append(metrics.accuracy_score(y_test, predicted_y)*100)
        siz = dtc.tree_.node_count
        temp_treeSizes.append(siz)
    meanAccuracy = statistics.mean(temp_accuracies)
    meanTreeSize = statistics.mean(temp_treeSizes)
    print('experiment with train size:',i*100,'%\n_________________')
    print('accuracies:',temp_accuracies)
    print('Mean:', meanAccuracy)
    print('Max:', max(temp_accuracies))
    print('Min:', min(temp_accuracies))
    print('treeSizes:',temp_treeSizes)
    print('Mean:',meanTreeSize)
    print('Max:', max(temp_treeSizes))
    print('Min:', min(temp_treeSizes))
    print('##########')
    accuracies.append(meanAccuracy)
    treeSizes.append(meanTreeSize)

print("Mean accuracies starting from training size 50 to 80:", accuracies)
plt.pyplot.plot(rangeList, accuracies, color='black', linewidth=1, marker='o', markerfacecolor='pink', markersize=8)
plt.pyplot.xlabel('training set sizes')
plt.pyplot.ylabel('accuracies')
plt.pyplot.show()

print("Mean tree sizes starting from training size 50 to 80:",treeSizes)
plt.pyplot.plot(rangeList,treeSizes, color='black', linewidth=1, marker='o', markerfacecolor='pink', markersize=8)
plt.pyplot.xlabel('training set sizes')
plt.pyplot.ylabel('tree sizes')
plt.pyplot.show()


