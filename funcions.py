from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

def print_score(classifier,X_train,y_train,X_test,y_test,matriu=False,report=False):
    print('%-50s Accuracy Score: %.4f\n' % 
          (classifier,accuracy_score(y_test,classifier.predict(X_test))))
    if(report): print('Classification Report:\n{}'
                      .format(classification_report(y_test,classifier.predict(X_test))))
    if(matriu): print('Confusion Matrix:\n{}\n'
                      .format(confusion_matrix(y_test,classifier.predict(X_test))))