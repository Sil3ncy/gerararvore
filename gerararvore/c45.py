import pandas as pd
import numpy as np
import math
from collections import Counter
import graphviz

class C45:
    def __init__(self, filename, target_col):
        self.df = pd.read_csv(filename)
        self.target = target_col
        self.tree = None

    def entropy(self, y):
        counts = Counter(y)
        total = len(y)
        return -sum((count/total) * math.log2(count/total) for count in counts.values())

    def info_gain(self, df, attr, threshold=None):
       
        if threshold is None:
            total_entropy = self.entropy(df[self.target])
            values = df[attr].unique()
            weighted_entropy = 0
            for v in values:
                subset = df[df[attr] == v]
                weighted_entropy += (len(subset)/len(df)) * self.entropy(subset[self.target])
            return total_entropy - weighted_entropy

        
        left = df[df[attr] <= threshold]
        right = df[df[attr] > threshold]
        total_entropy = self.entropy(df[self.target])
        weighted_entropy = 0
        for subset in [left, right]:
            if len(subset) > 0:
                weighted_entropy += (len(subset)/len(df)) * self.entropy(subset[self.target])
        return total_entropy - weighted_entropy

    def split_info(self, df, attr, threshold=None):
        if threshold is None:
            values = df[attr].unique()
            total = len(df)
            split = 0
            for v in values:
                prob = len(df[df[attr] == v]) / total
                split -= prob * math.log2(prob) if prob > 0 else 0
            return split if split != 0 else 1e-9
        else:
            left = df[df[attr] <= threshold]
            right = df[df[attr] > threshold]
            total = len(df)
            split = 0
            for subset in [left, right]:
                prob = len(subset) / total
                split -= prob * math.log2(prob) if prob > 0 else 0
            return split if split != 0 else 1e-9

    def gain_ratio(self, df, attr):
        if np.issubdtype(df[attr].dtype, np.number):  
            
            values = sorted(df[attr].unique())
            if len(values) <= 1:
                return -1, None
            thresholds = [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]
            best_ratio, best_threshold = -1, None
            for t in thresholds:
                gain = self.info_gain(df, attr, threshold=t)
                split = self.split_info(df, attr, threshold=t)
                ratio = gain / split if split > 0 else 0
                if ratio > best_ratio:
                    best_ratio, best_threshold = ratio, t
            return best_ratio, best_threshold
        else:
            gain = self.info_gain(df, attr)
            split = self.split_info(df, attr)
            ratio = gain / split if split > 0 else 0
            return ratio, None

    
    def best_attribute(self, df, attrs):
        best_attr, best_threshold, best_ratio = None, None, -1
        for attr in attrs:
            ratio, threshold = self.gain_ratio(df, attr)
            if ratio > best_ratio:
                best_attr, best_threshold, best_ratio = attr, threshold, ratio
        return best_attr, best_threshold

    def build_tree(self, df=None, attrs=None):
        if df is None:
            df = self.df
        if attrs is None:
            attrs = [c for c in df.columns if c != self.target]

        
        if len(df[self.target].unique()) == 1:
            return df[self.target].iloc[0]

        
        if not attrs:
            return df[self.target].mode()[0]

        best_attr, best_threshold = self.best_attribute(df, attrs)
        if best_attr is None:
            return df[self.target].mode()[0]

        
        if best_threshold is None:
            tree = {best_attr: {}}
            for v in df[best_attr].unique():
                subset = df[df[best_attr] == v]
                if subset.empty:
                    tree[best_attr][v] = df[self.target].mode()[0]
                else:
                    new_attrs = [a for a in attrs if a != best_attr]
                    tree[best_attr][v] = self.build_tree(subset, new_attrs)
        else:
            
            left = df[df[best_attr] <= best_threshold]
            right = df[df[best_attr] > best_threshold]
            tree = {f"{best_attr} <= {best_threshold}": {
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
