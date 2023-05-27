import pandas as pd
import numpy as np
import utility as ut

# Save Data: training and testing
def save_data(X_train,y_train,X_test,y_test):
  np.savetxt('X_train.csv', X_train, delimiter=',', fmt='%s')
  np.savetxt('y_train.csv', y_train, delimiter=',', fmt='%s')
  np.savetxt('X_test.csv', X_test, delimiter=',', fmt='%s')
  np.savetxt('y_test.csv', y_test, delimiter=',', fmt='%s')

# Binary Label
def binary_label(label):
  uniques = np.unique(label)
  y_binary_label = np.where(label[:,None] == uniques,1,0)
  return y_binary_label

# Load data csv
def load_data_csv():
  
  train = np.loadtxt('train.csv',delimiter=',')
  X_train = np.array(train[:,:-1])
  y_train = binary_label(np.array(train[:,-1]))

  test = np.loadtxt('test.csv',delimiter=',')
  X_test = test[:,:-1]
  y_test = binary_label(np.array(test[:,-1]))
  
  return X_train, y_train , X_test , y_test
  
# Beginning...
def main():        
  X_train, y_train, X_test, y_test = load_data_csv()
  save_data(X_train,y_train,X_test,y_test)

if __name__ == '__main__':   
	 main()


