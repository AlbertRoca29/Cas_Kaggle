from Data import *
from funcions import print_score

from import_models import *

from sklearn.feature_selection import SelectKBest, chi2, f_regression, mutual_info_classif
from sklearn.metrics import accuracy_score

models = [
    LogisticRegression(),
    SVC(kernel='rbf'),
    LinearSVC(), 
    Perceptron(), 
    SGDClassifier(),
    LinearDiscriminantAnalysis(),  
    KNeighborsClassifier(n_neighbors = 5),
    GaussianNB(),
    DecisionTreeClassifier(criterion='entropy'),
    RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
]


for classifier in models:  
    classifier.fit(X_train,y_train)
    print_score(classifier,X_train,y_train,X_test,y_test
                ,matriu = False) # si es vol veure la matriu de correlació posar True
    

plt.figure(figsize=(13,9))



def k_features_accuracy_evolution(classifier,X_train,y_train,X_test,y_test,n_final, Print= False):
    selector = [0] + [SelectKBest(score_func=chi2, k=i) for i in range(1,n_final+1)]
    acc =[]
    for i in range(1,n_final+1):
        X_train_r = selector[i].fit_transform(X_train,y_train)
        X_test_r = selector[i].transform(X_test)
        classifier.fit(X_train_r,y_train)
        acc.append(accuracy_score(y_test,classifier.predict(X_test_r)))
    plt.plot(range(1,n_final+1),acc,label = classifier)
    
    # Fer la pestanya del output força ampla si es vol printar
    if Print:
        print(type(classifier).__name__.ljust(30, ' '),np.array(acc))


np.set_printoptions(precision=4, linewidth=np.inf)
for classifier in models:  
    k_features_accuracy_evolution(classifier,X_train,y_train,X_test,y_test,18)
np.set_printoptions(precision=None, linewidth= None)
plt.xlabel("n_components")
plt.ylabel("Test accuracy")
plt.legend(bbox_to_anchor=(1.03, 1))
plt.ylim([0.825, 1.025])

plt.savefig('k_features_evolution.png',bbox_inches='tight',dpi = 1000)