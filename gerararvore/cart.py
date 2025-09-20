import pandas as pd
import numpy as np
import graphviz
from collections import Counter

class CART:
    def __init__(self, filename, target_col):
        self.df = pd.read_csv(filename)
        self.target = target_col
        self.tree = None

    
    def gini(self, y):
        counts = Counter(y)
        total = len(y)
        return 1 - sum((count/total)**2 for count in counts.values())

    def gini_split(self, df, attr, threshold=None):
        if threshold is None:
            
            values = df[attr].unique()
            best_gini, best_val = float("inf"), None
            for v in values:
                left = df[df[attr] == v]
                right = df[df[attr] != v]
                gini_left = self.gini(left[self.target]) if len(left) > 0 else 0
                gini_right = self.gini(right[self.target]) if len(right) > 0 else 0
                weighted_gini = (len(left)/len(df))*gini_left + (len(right)/len(df))*gini_right
                if weighted_gini < best_gini:
                    best_gini, best_val = weighted_gini, v
            return best_gini, best_val
        else:
           
            left = df[df[attr] <= threshold]
            right = df[df[attr] > threshold]
            gini_left = self.gini(left[self.target]) if len(left) > 0 else 0
            gini_right = self.gini(right[self.target]) if len(right) > 0 else 0
            weighted_gini = (len(left)/len(df))*gini_left + (len(right)/len(df))*gini_right
            return weighted_gini, threshold

    def best_split(self, df, attrs):
        best_attr, best_val, best_gini = None, None, float("inf")
        is_numeric = False

        for attr in attrs:
            if np.issubdtype(df[attr].dtype, np.number):
                
                values = sorted(df[attr].unique())
                thresholds = [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]
                for t in thresholds:
                    gini_val, _ = self.gini_split(df, attr, threshold=t)
                    if gini_val < best_gini:
                        best_gini, best_attr, best_val, is_numeric = gini_val, attr, t, True
            else:
                gini_val, val = self.gini_split(df, attr)
                if gini_val < best_gini:
                    best_gini, best_attr, best_val, is_numeric = gini_val, attr, val, False

        return best_attr, best_val, is_numeric

    
    def build_tree(self, df=None, attrs=None):
        if df is None:
            df = self.df
        if attrs is None:
            attrs = [c for c in df.columns if c != self.target]

        
        if len(df[self.target].unique()) == 1:
            return df[self.target].iloc[0]

        
        if not attrs:
            return df[self.target].mode()[0]

        best_attr, best_val, is_numeric = self.best_split(df, attrs)
        if best_attr is None:
            return df[self.target].mode()[0]

        if is_numeric:
            left = df[df[best_attr] <= best_val]
            right = df[df[best_attr] > best_val]
            tree = {f"{best_attr} <= {best_val}": {
                "yes": self.build_tree(left, attrs),
                "no": self.build_tree(right, attrs)
            }}
        else:
            left = df[df[best_attr] == best_val]
            right = df[df[best_attr] != best_val]
            tree = {f"{best_attr} == {best_val}": {
                "yes": self.build_tree(left, attrs),
                "no": self.build_tree(right, attrs)
            }}

        self.tree = tree
        return tree

    
    def plot_tree(self, tree=None, parent=None, graph=None):
        if tree is None:
            tree = self.tree
        if graph is None:
            graph = graphviz.Digraph()

        for attr, branches in tree.items():
            for val, subtree in branches.items():
                node_name = f"{attr} [{val}]"
                if isinstance(subtree, dict):
                    graph.node(node_name, label=node_name)
                    if parent:
                        graph.edge(parent, node_name)
                    self.plot_tree(subtree, node_name, graph)
                else:
                    leaf_name = f"{node_name}\n→ {subtree}"
                    graph.node(leaf_name, label=leaf_name, shape="box")
                    if parent:
                        graph.edge(parent, leaf_name)
        return graph
