from sklearn.metrics import accuracy_score
import pandas as pd
df4 = pd.read_csv("./outTest.csv")
df5 = pd.read_csv("./out.csv")
acc = accuracy_score(df4.values,df5.values)
print(acc)