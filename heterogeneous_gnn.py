#!/usr/bin/env python3
"""
Heterogeneous Graph Neural Network for Fraud Detection
======================================================

Implementation of advanced GNN with multiple node types:
1. User nodes (cardholders)
2. Merchant nodes (businesses)
3. Transaction nodes (individual transactions)
4. Heterogeneous message passing
5. Attention mechanisms across node types
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear, global_mean_pool
from torch_geometric.data import HeteroData, DataLoader
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeterogeneousGraphBuilder:
    """
    Build heterogeneous graphs with User, Merchant, and Transaction nodes.
    
    Graph structure:
    - Users ↔ Transactions (cardholder relationships)
    - Merchants ↔ Transactions (business relationships)
    - Users ↔ Users (similar behavior patterns)
    - Merchants ↔ Merchants (similar business patterns)
    - Transactions ↔ Transactions (temporal/similarity relationships)
    """
    
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.merchant_encoder = LabelEncoder()
        self.scalers = {}
        
    def build_heterogeneous_graph(self, df, max_users=1000, max_merchants=500):
        """
        Build heterogeneous graph from transaction data.
        
        Args:
            df: Transaction DataFrame
            max_users: Maximum number of users to include
            max_merchants: Maximum number of merchants to include
        
        Returns:
            HeteroData object
        """
        logger.info("Building heterogeneous graph")
        
        # Sample data for manageable graph size
        df_sample = df.sample(min(len(df), 50000)).copy()
        
        # Create synthetic user and merchant IDs
        df_sample = self._create_entities(df_sample, max_users, max_merchants)
        
        # Create graph data
        data = HeteroData()
        
        # Add node features
        self._add_node_features(data, df_sample)
        
        # Add edges
        self._add_edges(data, df_sample)
        
        # Add labels (transaction-level fraud)
        data['transaction'].y = torch.tensor(df_sample['Class'].values, dtype=torch.long)
        
        logger.info(f"Created heterogeneous graph:")
        logger.info(f"  Users: {data['user'].x.shape[0]}")
        logger.info(f"  Merchants: {data['merchant'].x.shape[0]}")
        logger.info(f"  Transactions: {data['transaction'].x.shape[0]}")
        logger.info(f"  Edges: {len(data.edge_types)}")
        
        return data
    
    def _create_entities(self, df, max_users, max_merchants):
        """Create synthetic user and merchant entities."""
        logger.info("Creating synthetic entities")
        
        # Create user IDs based on transaction patterns
        # Users with similar V-feature patterns are likely the same person
        user_features = df[['V1', 'V2', 'V3', 'V4', 'V5']].values
        
        # Simple clustering to create user groups
        from sklearn.cluster import KMeans
        user_kmeans = KMeans(n_clusters=min(max_users, len(df) // 10), random_state=42)
        df['user_id'] = user_kmeans.fit_predict(user_features)
        
        # Create merchant IDs based on amount and time patterns
        # Merchants with similar transaction patterns
        merchant_features = []
        for _, row in df.iterrows():
            # Create merchant signature based on amount patterns and V features
            hour = (row['Time'] % (24 * 3600)) // 3600
            amount_bin = min(4, int(np.log1p(row['Amount'])))
            v_signature = int(np.mean([row['V6'], row['V7'], row['V8']]) * 10) % 100
            merchant_features.append([hour, amount_bin, v_signature])
        
        merchant_features = np.array(merchant_features)
        merchant_kmeans = KMeans(n_clusters=min(max_merchants, len(df) // 20), random_state=42)
        df['merchant_id'] = merchant_kmeans.fit_predict(merchant_features)
        
        # Encode IDs
        df['user_id'] = self.user_encoder.fit_transform(df['user_id'])
        df['merchant_id'] = self.merchant_encoder.fit_transform(df['merchant_id'])
        
        logger.info(f"Created {len(np.unique(df['user_id']))} users and {len(np.unique(df['merchant_id']))} merchants")
        
        return df
    
    def _add_node_features(self, data, df):
        """Add features for each node type."""
        logger.info("Adding node features")
        
        # Transaction features (original V1-V28 + Amount + Time)
        transaction_features = []
        for col in ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]:
            if col in df.columns:
                transaction_features.append(df[col].values)
        
        transaction_features = np.column_stack(transaction_features)
        
        # Scale transaction features
        scaler = StandardScaler()
        transaction_features_scaled = scaler.fit_transform(transaction_features)
        self.scalers['transaction'] = scaler
        
        data['transaction'].x = torch.tensor(transaction_features_scaled, dtype=torch.float)
        
        # User features (aggregated from their transactions)
        user_features = []
        for user_id in np.unique(df['user_id']):
            user_transactions = df[df['user_id'] == user_id]
            
            # Aggregate features
            features = [
                len(user_transactions),  # Number of transactions
                user_transactions['Amount'].mean(),  # Average amount
                user_transactions['Amount'].std(),   # Amount variability
                user_transactions['Class'].mean(),   # Fraud rate
                user_transactions['Time'].min(),     # First transaction time
                user_transactions['Time'].max(),     # Last transaction time
                user_transactions[[f'V{i}' for i in range(1, 6)]].mean().mean(),  # V features avg
                user_transactions['Amount'].quantile(0.95) if len(user_transactions) > 1 else user_transactions['Amount'].iloc[0]  # 95th percentile
            ]
            
            user_features.append(features)
        
        user_features = np.array(user_features)
        user_features = np.nan_to_num(user_features)  # Handle NaN values
        
        # Scale user features
        scaler = StandardScaler()
        user_features_scaled = scaler.fit_transform(user_features)
        self.scalers['user'] = scaler
        
        data['user'].x = torch.tensor(user_features_scaled, dtype=torch.float)
        
        # Merchant features (aggregated from their transactions)
        merchant_features = []
        for merchant_id in np.unique(df['merchant_id']):
            merchant_transactions = df[df['merchant_id'] == merchant_id]
            
            features = [
                len(merchant_transactions),  # Number of transactions
                merchant_transactions['Amount'].mean(),  # Average amount
                merchant_transactions['Amount'].std(),   # Amount variability
                merchant_transactions['Class'].mean(),   # Fraud rate
                len(merchant_transactions['user_id'].unique()),  # Unique customers
                merchant_transactions['Time'].max() - merchant_transactions['Time'].min(),  # Time span
                merchant_transactions[[f'V{i}' for i in range(6, 11)]].mean().mean(),  # V features avg
                (merchant_transactions['Time'] % (24 * 3600) // 3600).mode()[0] if len(merchant_transactions) > 0 else 12  # Peak hour
            ]
            
            merchant_features.append(features)
        
        merchant_features = np.array(merchant_features)
        merchant_features = np.nan_to_num(merchant_features)
        
        # Scale merchant features
        scaler = StandardScaler()
        merchant_features_scaled = scaler.fit_transform(merchant_features)
        self.scalers['merchant'] = scaler
        
        data['merchant'].x = torch.tensor(merchant_features_scaled, dtype=torch.float)
    
    def _add_edges(self, data, df):
        """Add edges between different node types."""
        logger.info("Adding edges")
        
        # User-Transaction edges
        user_transaction_edges = []
        for idx, row in df.iterrows():
            user_transaction_edges.append([row['user_id'], idx])  # User to transaction
        
        user_transaction_edges = np.array(user_transaction_edges).T
        data['user', 'makes', 'transaction'].edge_index = torch.tensor(user_transaction_edges, dtype=torch.long)
        data['transaction', 'made_by', 'user'].edge_index = torch.tensor(user_transaction_edges[[1, 0]], dtype=torch.long)
        
        # Merchant-Transaction edges
        merchant_transaction_edges = []
        for idx, row in df.iterrows():
            merchant_transaction_edges.append([row['merchant_id'], idx])  # Merchant to transaction
        
        merchant_transaction_edges = np.array(merchant_transaction_edges).T
        data['merchant', 'processes', 'transaction'].edge_index = torch.tensor(merchant_transaction_edges, dtype=torch.long)
        data['transaction', 'processed_by', 'merchant'].edge_index = torch.tensor(merchant_transaction_edges[[1, 0]], dtype=torch.long)
        
        # User-User similarity edges
        user_user_edges = self._create_similarity_edges(
            data['user'].x.numpy(), threshold=0.8, max_edges_per_node=5
        )
        if len(user_user_edges) > 0:
            data['user', 'similar_to', 'user'].edge_index = torch.tensor(user_user_edges, dtype=torch.long)
        
        # Merchant-Merchant similarity edges
        merchant_merchant_edges = self._create_similarity_edges(
            data['merchant'].x.numpy(), threshold=0.8, max_edges_per_node=5
        )
        if len(merchant_merchant_edges) > 0:
            data['merchant', 'similar_to', 'merchant'].edge_index = torch.tensor(merchant_merchant_edges, dtype=torch.long)
        
        # Transaction-Transaction temporal edges
        transaction_temporal_edges = self._create_temporal_edges(df, max_time_diff=3600)
        if len(transaction_temporal_edges) > 0:
            data['transaction', 'temporal', 'transaction'].edge_index = torch.tensor(transaction_temporal_edges, dtype=torch.long)
    
    def _create_similarity_edges(self, features, threshold=0.8, max_edges_per_node=5):
        """Create similarity edges between nodes of the same type."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(features)
        edges = []
        
        for i in range(len(features)):
            # Find most similar nodes
            similarities = similarity_matrix[i]
            similar_indices = np.argsort(similarities)[::-1][1:max_edges_per_node+1]  # Exclude self
            
            for j in similar_indices:
                if similarities[j] > threshold:
                    edges.append([i, j])
        
        return np.array(edges).T if edges else np.empty((2, 0))
    
    def _create_temporal_edges(self, df, max_time_diff=3600):
        """Create temporal edges between consecutive transactions."""
        df_sorted = df.sort_values('Time').reset_index(drop=True)
        edges = []
        
        for i in range(len(df_sorted) - 1):
            time_diff = df_sorted.iloc[i+1]['Time'] - df_sorted.iloc[i]['Time']
            if time_diff <= max_time_diff:
                edges.append([i, i+1])
        
        return np.array(edges).T if edges else np.empty((2, 0))

class HeterogeneousGAT(nn.Module):
    """
    Heterogeneous Graph Attention Network for fraud detection.
    
    Features:
    1. Multi-head attention across different node types
    2. Heterogeneous message passing
    3. Node-type specific transformations
    4. Global graph-level prediction
    """
    
    def __init__(self, node_features, hidden_dim=64, num_heads=4, num_layers=2):
        """
        Args:
            node_features: Dictionary of feature dimensions for each node type
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            num_layers: Number of GNN layers
        """
        super(HeterogeneousGAT, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projections for each node type
        self.input_projections = nn.ModuleDict()
        for node_type, feature_dim in node_features.items():
            self.input_projections[node_type] = Linear(feature_dim, hidden_dim)
        
        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        
        for _ in range(num_layers):
            conv_dict = {}
            
            # Define all edge types and their convolutions
            edge_types = [
                ('user', 'makes', 'transaction'),
                ('transaction', 'made_by', 'user'),
                ('merchant', 'processes', 'transaction'),
                ('transaction', 'processed_by', 'merchant'),
                ('user', 'similar_to', 'user'),
                ('merchant', 'similar_to', 'merchant'),
                ('transaction', 'temporal', 'transaction')
            ]
            
            for edge_type in edge_types:
                conv_dict[edge_type] = GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=0.2,
                    add_self_loops=False
                )
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # Output layers for transaction classification
        self.transaction_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Attention for global aggregation
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=0.2
        )
        
    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        """
        Forward pass through heterogeneous GAT.
        
        Args:
            x_dict: Dictionary of node features for each type
            edge_index_dict: Dictionary of edge indices for each edge type
            batch_dict: Batch indices for each node type
        
        Returns:
            Transaction-level predictions
        """
        # Input projections
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.input_projections[node_type](x)
        
        # Heterogeneous convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply activation and residual connection
            for node_type in x_dict:
                x_dict[node_type] = F.relu(x_dict[node_type])
        
        # Extract transaction embeddings
        transaction_embeddings = x_dict['transaction']
        
        # Global attention (optional)
        if batch_dict is not None and 'transaction' in batch_dict:
            # Apply global attention across transaction embeddings
            transaction_embeddings = transaction_embeddings.unsqueeze(0)
            attn_output, _ = self.global_attention(
                transaction_embeddings, transaction_embeddings, transaction_embeddings
            )
            transaction_embeddings = attn_output.squeeze(0)
        
        # Classification
        logits = self.transaction_classifier(transaction_embeddings)
        
        return logits
    
    def get_embeddings(self, x_dict, edge_index_dict):
        """Get node embeddings for analysis."""
        # Input projections
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.input_projections[node_type](x)
        
        # Heterogeneous convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            for node_type in x_dict:
                x_dict[node_type] = F.relu(x_dict[node_type])
        
        return x_dict

class HeterogeneousFraudDetector:
    """
    Complete heterogeneous GNN fraud detection system.
    
    Features:
    1. Graph construction from transaction data
    2. Heterogeneous GNN training
    3. Evaluation and analysis
    4. Explainability features
    """
    
    def __init__(self, device='cpu'):
        """
        Args:
            device: Device to run computations on
        """
        self.device = device
        self.graph_builder = HeterogeneousGraphBuilder()
        self.model = None
        self.data = None
        self.train_mask = None
        self.test_mask = None
        
    def prepare_data(self, df, test_size=0.2):
        """
        Prepare heterogeneous graph data.
        
        Args:
            df: Transaction DataFrame
            test_size: Fraction for test set
        
        Returns:
            Prepared data and splits
        """
        logger.info("Preparing heterogeneous graph data")
        
        # Build graph
        self.data = self.graph_builder.build_heterogeneous_graph(df)
        
        # Create train/test splits
        n_transactions = self.data['transaction'].x.shape[0]
        indices = np.arange(n_transactions)
        
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, 
            stratify=self.data['transaction'].y.numpy(),
            random_state=42
        )
        
        # Create masks
        self.train_mask = torch.zeros(n_transactions, dtype=torch.bool)
        self.test_mask = torch.zeros(n_transactions, dtype=torch.bool)
        
        self.train_mask[train_indices] = True
        self.test_mask[test_indices] = True
        
        # Move to device
        self.data = self.data.to(self.device)
        self.train_mask = self.train_mask.to(self.device)
        self.test_mask = self.test_mask.to(self.device)
        
        logger.info(f"Prepared data with {len(train_indices)} training and {len(test_indices)} test transactions")
        
        return self.data
    
    def train_model(self, hidden_dim=64, num_heads=4, num_layers=2, 
                   epochs=200, lr=0.01, weight_decay=1e-5):
        """
        Train the heterogeneous GNN model.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads  
            num_layers: Number of GNN layers
            epochs: Training epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
        """
        logger.info("Training heterogeneous GNN model")
        
        if self.data is None:
            raise ValueError("Data must be prepared before training")
        
        # Get node feature dimensions
        node_features = {
            node_type: self.data[node_type].x.shape[1] 
            for node_type in self.data.node_types
        }
        
        # Initialize model
        self.model = HeterogeneousGAT(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        ).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Class weights for imbalanced data
        class_counts = torch.bincount(self.data['transaction'].y[self.train_mask])
        class_weights = len(self.train_mask) / (2 * class_counts.float())
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop
        self.model.train()
        train_losses = []
        train_accuracies = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(
                self.data.x_dict, 
                self.data.edge_index_dict
            )
            
            # Loss on training transactions
            loss = criterion(
                logits[self.train_mask], 
                self.data['transaction'].y[self.train_mask]
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            with torch.no_grad():
                train_pred = logits[self.train_mask].argmax(dim=1)
                train_acc = (train_pred == self.data['transaction'].y[self.train_mask]).float().mean()
            
            train_losses.append(loss.item())
            train_accuracies.append(train_acc.item())
            
            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss:.4f}, Train Acc = {train_acc:.4f}")
        
        logger.info("Training completed")
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies
        }
    
    def evaluate_model(self):
        """
        Evaluate the trained model.
        
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating heterogeneous GNN model")
        
        self.model.eval()
        
        with torch.no_grad():
            # Get predictions
            logits = self.model(
                self.data.x_dict, 
                self.data.edge_index_dict
            )
            
            # Test predictions
            test_logits = logits[self.test_mask]
            test_labels = self.data['transaction'].y[self.test_mask]
            
            # Probabilities and predictions
            test_probs = F.softmax(test_logits, dim=1)[:, 1]
            test_pred = test_logits.argmax(dim=1)
            
            # Convert to numpy
            test_labels_np = test_labels.cpu().numpy()
            test_pred_np = test_pred.cpu().numpy()
            test_probs_np = test_probs.cpu().numpy()
            
            # Calculate metrics
            from sklearn.metrics import classification_report, roc_auc_score, f1_score
            
            f1 = f1_score(test_labels_np, test_pred_np)
            roc_auc = roc_auc_score(test_labels_np, test_probs_np)
            
            # Detailed classification report
            class_report = classification_report(
                test_labels_np, test_pred_np, 
                target_names=['Normal', 'Fraud'],
                output_dict=True
            )
            
            logger.info("Evaluation Results:")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info(f"  ROC-AUC: {roc_auc:.4f}")
            logger.info(f"  Precision: {class_report['1']['precision']:.4f}")
            logger.info(f"  Recall: {class_report['1']['recall']:.4f}")
            
            return {
                'f1_score': f1,
                'roc_auc': roc_auc,
                'classification_report': class_report,
                'predictions': test_pred_np,
                'probabilities': test_probs_np,
                'labels': test_labels_np
            }
    
    def analyze_node_embeddings(self, save_path=None):
        """
        Analyze and visualize node embeddings.
        
        Args:
            save_path: Path to save visualization
        """
        if self.model is None:
            raise ValueError("Model must be trained before analysis")
        
        logger.info("Analyzing node embeddings")
        
        self.model.eval()
        
        with torch.no_grad():
            # Get embeddings
            embeddings_dict = self.model.get_embeddings(
                self.data.x_dict,
                self.data.edge_index_dict
            )
            
            # Analyze transaction embeddings
            transaction_embeddings = embeddings_dict['transaction'].cpu().numpy()
            transaction_labels = self.data['transaction'].y.cpu().numpy()
            
            # Dimensionality reduction for visualization
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            
            # PCA
            pca = PCA(n_components=2)
            embeddings_pca = pca.fit_transform(transaction_embeddings)
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_tsne = tsne.fit_transform(transaction_embeddings[:1000])  # Sample for speed
            labels_tsne = transaction_labels[:1000]
            
            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # PCA plot
            scatter = axes[0].scatter(
                embeddings_pca[:, 0], embeddings_pca[:, 1],
                c=transaction_labels, cmap='RdYlBu', alpha=0.6
            )
            axes[0].set_title('Transaction Embeddings - PCA')
            axes[0].set_xlabel('PC 1')
            axes[0].set_ylabel('PC 2')
            plt.colorbar(scatter, ax=axes[0])
            
            # t-SNE plot
            scatter = axes[1].scatter(
                embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                c=labels_tsne, cmap='RdYlBu', alpha=0.6
            )
            axes[1].set_title('Transaction Embeddings - t-SNE')
            axes[1].set_xlabel('t-SNE 1')
            axes[1].set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=axes[1])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Embedding analysis saved to {save_path}")
            
            plt.show()
            
            return {
                'embeddings': embeddings_dict,
                'pca': embeddings_pca,
                'tsne': embeddings_tsne
            }

def main():
    """Main function to demonstrate heterogeneous GNN."""
    logger.info("Starting Heterogeneous GNN Fraud Detection Demo")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load data
    try:
        df = pd.read_csv('creditcard.csv')
        logger.info(f"Dataset loaded: {len(df):,} transactions")
    except FileNotFoundError:
        logger.error("creditcard.csv not found. Please ensure the dataset is available.")
        return
    
    # Use subset for demo (GNNs are computationally intensive)
    df_demo = df.sample(min(20000, len(df))).reset_index(drop=True)
    logger.info(f"Using {len(df_demo):,} transactions for demo")
    
    # Initialize detector
    detector = HeterogeneousFraudDetector(device=device)
    
    # Prepare data
    data = detector.prepare_data(df_demo, test_size=0.2)
    
    # Train model
    training_history = detector.train_model(
        hidden_dim=64, 
        num_heads=4, 
        num_layers=2,
        epochs=100,
        lr=0.01
    )
    
    # Evaluate model
    results = detector.evaluate_model()
    
    # Analyze embeddings
    embedding_analysis = detector.analyze_node_embeddings(
        save_path="heterogeneous_gnn_embeddings.png"
    )
    
    # Save results
    import joblib
    save_data = {
        'detector': detector,
        'results': results,
        'training_history': training_history,
        'embedding_analysis': embedding_analysis
    }
    
    joblib.dump(save_data, 'heterogeneous_gnn_results.joblib')
    
    logger.info("Heterogeneous GNN demo completed!")
    
    return detector, results

if __name__ == "__main__":
    main()