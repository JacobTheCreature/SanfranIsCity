import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def discretize_column(df, col, bins=4, labels=None):
    if labels is None:
        labels = [f'{col}_low', f'{col}_med', f'{col}_high', f'{col}_very_high'][:bins]
    df[f'{col}_binned'] = pd.cut(df[col], bins=bins, labels=labels, duplicates='drop')
    return df


def prepare_transaction_data(df, categorical_cols, numerical_cols=None):
    df_encoded = df.copy()
    
    if numerical_cols:
        for col in numerical_cols:
            if col in df.columns:
                df_encoded = discretize_column(df_encoded, col)
                categorical_cols.append(f'{col}_binned')
    
    transactions = []
    for _, row in df_encoded[categorical_cols].iterrows():
        transaction = [f"{col}={row[col]}" for col in categorical_cols if pd.notna(row[col])]
        transactions.append(transaction)
    
    return transactions


def mine_association_rules(transactions, min_support=0.05, min_confidence=0.5):
    if len(transactions) == 0:
        return pd.DataFrame()
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    if len(frequent_itemsets) == 0:
        return pd.DataFrame()
    
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    rules['lift'] = rules['lift'].round(3)
    rules['confidence'] = rules['confidence'].round(3)
    rules['support'] = rules['support'].round(4)
    
    return rules


def format_rules(rules, top_n=20):
    if len(rules) == 0:
        return pd.DataFrame()
    
    rules_sorted = rules.sort_values('lift', ascending=False).head(top_n)
    
    formatted = pd.DataFrame({
        'antecedents': rules_sorted['antecedents'].apply(lambda x: ', '.join(list(x))),
        'consequents': rules_sorted['consequents'].apply(lambda x: ', '.join(list(x))),
        'support': rules_sorted['support'],
        'confidence': rules_sorted['confidence'],
        'lift': rules_sorted['lift']
    })
    
    return formatted


def analyze_needle_associations(needle_df):
    categorical = []
    numerical = ['dist_to_bathroom_m', 'dist_to_encampment_m', 'bathrooms_within_500m', 'encampments_within_500m']
    
    if 'kmeans_cluster' in needle_df.columns:
        categorical.append('kmeans_cluster')
    if 'neighborhood' in needle_df.columns:
        categorical.append('neighborhood')
    if 'supervisor_district' in needle_df.columns:
        categorical.append('supervisor_district')
    if 'status' in needle_df.columns:
        categorical.append('status')
    if 'underserved' in needle_df.columns:
        categorical.append('underserved')
    
    available_numerical = [col for col in numerical if col in needle_df.columns]
    
    transactions = prepare_transaction_data(needle_df, categorical, available_numerical)
    rules = mine_association_rules(transactions, min_support=0.03, min_confidence=0.4)
    
    return format_rules(rules, top_n=20)


def analyze_encampment_associations(encampment_df):
    categorical = []
    numerical = ['dist_to_bathroom_m', 'needles_within_500m', 'bathrooms_within_500m']
    
    if 'kmeans_cluster' in encampment_df.columns:
        categorical.append('kmeans_cluster')
    if 'district' in encampment_df.columns:
        categorical.append('district')
    if 'sf_find_neighborhood' in encampment_df.columns:
        categorical.append('sf_find_neighborhood')
    if 'supervisor' in encampment_df.columns:
        categorical.append('supervisor')
    if 'underserved' in encampment_df.columns:
        categorical.append('underserved')
    
    if 'tents' in encampment_df.columns:
        numerical.append('tents')
    if 'structures' in encampment_df.columns:
        numerical.append('structures')
    
    available_numerical = [col for col in numerical if col in encampment_df.columns]
    
    transactions = prepare_transaction_data(encampment_df, categorical, available_numerical)
    rules = mine_association_rules(transactions, min_support=0.04, min_confidence=0.4)
    
    return format_rules(rules, top_n=20)


def analyze_bathroom_associations(bathroom_df):
    categorical = []
    numerical = ['needles_within_500m', 'encampments_within_500m']
    
    if 'kmeans_cluster' in bathroom_df.columns:
        categorical.append('kmeans_cluster')
    if 'supervisor_district' in bathroom_df.columns:
        categorical.append('supervisor_district')
    if 'analysis_neighborhood' in bathroom_df.columns:
        categorical.append('analysis_neighborhood')
    if 'resource_type' in bathroom_df.columns:
        categorical.append('resource_type')
    if 'access' in bathroom_df.columns:
        categorical.append('access')
    
    available_numerical = [col for col in numerical if col in bathroom_df.columns]
    
    transactions = prepare_transaction_data(bathroom_df, categorical, available_numerical)
    rules = mine_association_rules(transactions, min_support=0.05, min_confidence=0.4)
    
    return format_rules(rules, top_n=20)
