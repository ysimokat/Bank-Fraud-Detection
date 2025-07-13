#!/usr/bin/env python3
"""
Graph Neural Networks for Credit Card Fraud Detection
====================================================

This implements GNN to analyze transaction networks and detect fraud patterns
through relationship analysis between cardholders, merchants, and transactions.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAT, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import networkx as nx
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta

class TransactionGraphBuilder:
    """Build graph representation of transactions."""
    
    def __init__(self, time_window_hours=24):
        self.time_window = time_window_hours * 3600  # Convert to seconds
        self.scaler = StandardScaler()
        
    def build_transaction_graph(self, df, target_idx):
        """
        Build a graph centered around a target transaction.
        
        Nodes represent:
        - Transactions
        - Cards (anonymized)
        - Merchants (derived from patterns)
        
        Edges represent:
        - Card-Transaction relationships
        - Temporal relationships
        - Amount similarity relationships
        """
        target_row = df.iloc[target_idx]
        target_time = target_row['Time']
        
        # Get transactions within time window
        time_mask = (df['Time'] >= target_time - self.time_window) & \
                   (df['Time'] <= target_time + self.time_window)
        local_df = df[time_mask].copy()
        
        # Create graph
        G = nx.Graph()
        
        # Add transaction nodes
        for idx, row in local_df.iterrows():
            node_features = self._extract_node_features(row)
            G.add_node(f'txn_{idx}', 
                      type='transaction',
                      features=node_features,
                      is_fraud=row['Class'],
                      is_target=(idx == target_idx))
        
        # Add edges based on relationships
        self._add_similarity_edges(G, local_df)
        self._add_temporal_edges(G, local_df)
        self._add_pattern_edges(G, local_df)
        
        return G
    
    def _extract_node_features(self, row):
        """Extract features for a transaction node."""
        features = []
        
        # Basic features
        features.append(row['Amount'])
        features.append(np.log(row['Amount'] + 1))
        
        # Time features
        hour = (row['Time'] % (24 * 3600)) // 3600
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        
        # PCA features (top 10)
        for i in range(1, 11):
            if f'V{i}' in row:
                features.append(row[f'V{i}'])
        
        return np.array(features)
    
    def _add_similarity_edges(self, G, df):
        """Add edges between similar transactions."""
        amounts = df['Amount'].values
        indices = df.index.tolist()
        
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                # Similar amounts (within 10%)
                if abs(amounts[i] - amounts[j]) / (max(amounts[i], amounts[j]) + 1e-6) < 0.1:
                    G.add_edge(f'txn_{indices[i]}', f'txn_{indices[j]}', 
                              weight=1.0, type='amount_similarity')
    
    def _add_temporal_edges(self, G, df):
        """Add edges between temporally close transactions."""
        times = df['Time'].values
        indices = df.index.tolist()
        
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                time_diff = abs(times[i] - times[j])
                if time_diff < 300:  # Within 5 minutes
                    weight = 1.0 - (time_diff / 300)
                    G.add_edge(f'txn_{indices[i]}', f'txn_{indices[j]}', 
                              weight=weight, type='temporal')
    
    def _add_pattern_edges(self, G, df):
        """Add edges based on transaction patterns."""
        # Group by similar PCA patterns
        pca_cols = [col for col in df.columns if col.startswith('V')][:5]
        
        for col in pca_cols:
            # Discretize into bins
            bins = pd.qcut(df[col], q=5, duplicates='drop')
            
            for bin_label in bins.unique():
                nodes_in_bin = df[bins == bin_label].index
                
                # Connect transactions in same bin
                for i in range(len(nodes_in_bin)):
                    for j in range(i + 1, min(i + 3, len(nodes_in_bin))):
                        G.add_edge(f'txn_{nodes_in_bin[i]}', f'txn_{nodes_in_bin[j]}',
                                  weight=0.5, type='pattern')

class GraphAttentionFraudDetector(nn.Module):
    """
    Graph Attention Network for fraud detection.
    
    Uses attention mechanisms to focus on important transaction relationships.
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, dropout=0.2):
        super().__init__()
        
        # Graph attention layers
        self.gat1 = GAT(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GAT(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat3 = GAT(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)
        
        # Global graph features
        self.global_attention = nn.Linear(hidden_dim, 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),  # node + global features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Binary classification
        )
    
    def forward(self, x, edge_index, batch, target_mask):
        # Graph attention layers
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        x = self.gat3(x, edge_index)
        
        # Global graph representation
        global_max = global_max_pool(x, batch)
        global_mean = global_mean_pool(x, batch)
        
        # Attention-weighted global features
        attention_weights = torch.softmax(self.global_attention(x), dim=0)
        global_attention = torch.sum(x * attention_weights, dim=0, keepdim=True)
        
        # Get target node features
        target_features = x[target_mask]
        
        # Combine features
        combined = torch.cat([target_features, global_max, global_mean, global_attention], dim=1)
        
        # Classification
        output = self.classifier(combined)
        
        return output

class HybridGNNFraudDetector:
    """
    Hybrid system combining GNN with traditional models.
    """
    
    def __init__(self):
        self.graph_builder = TransactionGraphBuilder()
        self.gnn_model = None
        self.traditional_models = {}
        self.ensemble_weights = None
        
    def train(self, df, traditional_models):
        """Train the hybrid system."""
        print("🌐 Training Graph Neural Network for Fraud Detection...")
        
        self.traditional_models = traditional_models
        
        # Prepare graph data
        graph_data = self._prepare_graph_data(df)
        
        # Train GNN
        self.gnn_model = self._train_gnn(graph_data)
        
        # Learn ensemble weights
        self._learn_ensemble_weights(df)
        
        print("✅ Hybrid GNN system trained successfully")
    
    def _prepare_graph_data(self, df, sample_size=10000):
        """Prepare graph data for training."""
        print("📊 Building transaction graphs...")
        
        graphs = []
        
        # Sample transactions
        fraud_indices = df[df['Class'] == 1].index[:sample_size//2]
        normal_indices = df[df['Class'] == 0].sample(sample_size//2).index
        
        all_indices = list(fraud_indices) + list(normal_indices)
        
        for idx in all_indices:
            G = self.graph_builder.build_transaction_graph(df, idx)
            
            # Convert to PyTorch geometric data
            data = self._networkx_to_geometric(G, idx)
            graphs.append(data)
        
        return graphs
    
    def _networkx_to_geometric(self, G, target_idx):
        """Convert NetworkX graph to PyTorch Geometric format."""
        # Node features
        node_features = []
        node_to_idx = {}
        target_mask = []
        
        for i, (node, attrs) in enumerate(G.nodes(data=True)):
            node_to_idx[node] = i
            if 'features' in attrs:
                node_features.append(attrs['features'])
            target_mask.append(attrs.get('is_target', False))
        
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        
        # Edges
        edge_index = []
        edge_attr = []
        
        for u, v, attrs in G.edges(data=True):
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_index.append([node_to_idx[v], node_to_idx[u]])  # Undirected
            
            weight = attrs.get('weight', 1.0)
            edge_attr.extend([weight, weight])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Target
        target_node = list(G.nodes())[target_mask.index(True)]
        y = torch.tensor([G.nodes[target_node].get('is_fraud', 0)], dtype=torch.long)
        
        # Create data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.target_mask = torch.tensor(target_mask, dtype=torch.bool)
        
        return data
    
    def _train_gnn(self, graph_data):
        """Train the GNN model."""
        print("🧠 Training Graph Attention Network...")
        
        # Split data
        train_size = int(0.8 * len(graph_data))
        train_data = graph_data[:train_size]
        val_data = graph_data[train_size:]
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        
        # Initialize model
        input_dim = graph_data[0].x.shape[1]
        model = GraphAttentionFraudDetector(input_dim)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for epoch in range(50):
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                out = model(batch.x, batch.edge_index, batch.batch, batch.target_mask)
                loss = criterion(out, batch.y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/50, Loss: {total_loss/len(train_loader):.4f}")
        
        return model
    
    def predict_with_explanation(self, transaction_df, idx):
        """Make prediction with graph-based explanation."""
        # Build transaction graph
        G = self.graph_builder.build_transaction_graph(transaction_df, idx)
        
        # Get GNN prediction
        data = self._networkx_to_geometric(G, idx)
        self.gnn_model.eval()
        
        with torch.no_grad():
            gnn_output = self.gnn_model(data.x, data.edge_index, 
                                       torch.zeros(data.x.shape[0], dtype=torch.long),
                                       data.target_mask)
            gnn_pred = torch.softmax(gnn_output, dim=1)[0, 1].item()
        
        # Get traditional model predictions
        traditional_preds = {}
        for name, model in self.traditional_models.items():
            # Simplified - would need proper feature extraction
            traditional_preds[name] = 0.5  # Placeholder
        
        # Ensemble prediction
        final_pred = self._ensemble_predict(gnn_pred, traditional_preds)
        
        # Generate explanation
        explanation = self._generate_graph_explanation(G, gnn_pred, final_pred)
        
        return final_pred, explanation
    
    def _ensemble_predict(self, gnn_pred, traditional_preds):
        """Combine predictions using learned weights."""
        if self.ensemble_weights is None:
            # Simple average if weights not learned
            all_preds = [gnn_pred] + list(traditional_preds.values())
            return np.mean(all_preds)
        
        # Weighted combination
        weighted_sum = self.ensemble_weights['gnn'] * gnn_pred
        for name, pred in traditional_preds.items():
            weighted_sum += self.ensemble_weights.get(name, 0) * pred
        
        return weighted_sum
    
    def _learn_ensemble_weights(self, df):
        """Learn optimal ensemble weights."""
        # Simplified - in practice, use validation set
        self.ensemble_weights = {
            'gnn': 0.3,
            'random_forest': 0.4,
            'xgboost': 0.3
        }
    
    def _generate_graph_explanation(self, G, gnn_pred, final_pred):
        """Generate explanation based on graph structure."""
        explanation = {
            'prediction': 'Fraud' if final_pred > 0.5 else 'Normal',
            'confidence': final_pred,
            'gnn_contribution': gnn_pred,
            'graph_stats': {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'connected_frauds': sum(1 for n, d in G.nodes(data=True) 
                                      if d.get('is_fraud', 0) == 1)
            },
            'important_connections': []
        }
        
        # Find important connections
        target_node = [n for n, d in G.nodes(data=True) if d.get('is_target', False)][0]
        
        for neighbor in G.neighbors(target_node):
            if G.nodes[neighbor].get('is_fraud', 0) == 1:
                explanation['important_connections'].append({
                    'type': 'connected_to_fraud',
                    'edge_type': G.edges[target_node, neighbor].get('type', 'unknown'),
                    'weight': G.edges[target_node, neighbor].get('weight', 1.0)
                })
        
        return explanation

def demonstrate_gnn():
    """Demonstrate GNN capabilities."""
    print("🌐 Graph Neural Network Fraud Detection Demo")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv('/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/creditcard.csv')
    
    # Build sample graph
    builder = TransactionGraphBuilder()
    
    # Find a fraud transaction
    fraud_idx = df[df['Class'] == 1].index[0]
    
    print(f"\n📊 Building graph for transaction {fraud_idx}...")
    G = builder.build_transaction_graph(df, fraud_idx)
    
    print(f"✅ Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Analyze graph properties
    print("\n🔍 Graph Analysis:")
    print(f"   • Connected fraud transactions: {sum(1 for n, d in G.nodes(data=True) if d.get('is_fraud', 0) == 1)}")
    print(f"   • Average degree: {np.mean([d for n, d in G.degree()]):.2f}")
    print(f"   • Graph density: {nx.density(G):.4f}")
    
    # Show relationships
    target_node = [n for n, d in G.nodes(data=True) if d.get('is_target', False)][0]
    neighbors = list(G.neighbors(target_node))
    
    print(f"\n🔗 Target transaction connections:")
    for neighbor in neighbors[:5]:  # Show first 5
        edge_data = G.edges[target_node, neighbor]
        print(f"   • Connected to {neighbor}: {edge_data.get('type', 'unknown')} (weight: {edge_data.get('weight', 1.0):.2f})")
    
    print("\n💡 GNN Advantages:")
    print("   • Captures transaction relationships and patterns")
    print("   • Identifies fraud rings and coordinated attacks")
    print("   • Learns from network topology")
    print("   • Provides explainable connections")

if __name__ == "__main__":
    demonstrate_gnn()