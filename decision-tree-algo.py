import numpy as np
import operator
import pandas as pd
from Nodes import DecisionNode, LeafNode
from DecisionTree import DecisionTree

# df from the paper c4.5
df = pd.DataFrame({'Outlook': ['sunny', 'sunny', 'sunny', 'sunny', 'sunny', None, 'overcast', 'overcast', 'overcast', 'rain', 'rain', 'rain', 'rain', 'rain'],
#df = pd.DataFrame({'Outlook': ['sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'overcast', 'overcast', 'overcast', 'overcast', 'rain', 'rain', 'rain', 'rain', 'rain'],
    'Temperature': [75, 80, 85, 72, 69, 72, 83, 64, 81, 71, 65, 75, 68, 70],
    'Humidity': [70, 90, 85, 95, 70, 90, 78, 65, 75, 80, 70, 80, 80, 96], 
    'Windy': [True, True, False, False, False, True, False, True, False, True, True, False, False, False], 
    'target': ["Play", "Don't Play", "Don't Play", "Don't Play", "Play", "Play", "Play", "Play", "Play", "Don't Play", "Don't Play", "Play", "Play", "Play"]})
attributes_map = {'Outlook': 'categorical', 'Temperature': 'continuous', 
        'Humidity': 'continuous', 'Windy': 'boolean', 'target': 'categorical'}
dt = DecisionTree(attributes_map)
dt.fit(df)
rules  = dt.extract_rules()
print("\nRules extracted")
for rule in rules.keys():
    print("{}:".format(rule))
    print(rules[rule])
pred_df = df.fillna('?').iloc[[1,5]]
pred_df = pred_df.append({'Outlook': 'sunny', 'Temperature': 70, 'Humidity': '?', 'Windy': False, 'target': 'stocazzo'}, ignore_index=True)
predictions = dt.predict(pred_df.drop(columns=['target']))
print("\nPredictions")
print(predictions)
