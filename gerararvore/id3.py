import pandas as pd
import math
from collections import Counter
import graphviz

class ID3:
    def __init__(self, filename, target_col):
        self.df = pd.read_csv(filename)
        self.target = target_col
        self.tree = None

    def entropy(self, y):
        counts = Counter(y)
        total = len(y)
        return -sum((count/total) * math.log2(count/total) for count in counts.values())

    def info_gain(self, df, attr):
        total_entropy = self.entropy(df[self.target])
        values = df[attr].unique()
        weighted_entropy = 0
        for v in values:
            subset = df[df[attr] == v]
            weighted_entropy += (len(subset) / len(df)) * self.entropy(subset[self.target])
        return total_entropy - weighted_entropy

    def best_attribute(self, df, attrs):
        gains = {attr: self.info_gain(df, attr) for attr in attrs}
        return max(gains, key=gains.get), gains

    def build_tree(self, df=None, attrs=None):
        if df is None:
            df = self.df
        if attrs is None:
            attrs = [c for c in df.columns if c != self.target]

        
        if len(df[self.target].unique()) == 1:
            return df[self.target].iloc[0]

        
        if not attrs:
            return df[self.target].mode()[0]

        
        best, gains = self.best_attribute(df, attrs)
        tree = {best: {}}

        for v in df[best].unique():
            subset = df[df[best] == v]
            if subset.empty:
                tree[best][v] = df[self.target].mode()[0]
            else:
                new_attrs = [a for a in attrs if a != best]
                tree[best][v] = self.build_tree(subset, new_attrs)

        self.tree = tree
        return tree

    def plot_tree(self, tree=None, parent=None, graph=None):
        if tree is None:
            tree = self.tree
        if graph is None:
            graph = graphviz.Digraph()

        for attr, branches in tree.items():
            for val, subtree in branches.items():
                node_name = f"{attr} = {val}"
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
