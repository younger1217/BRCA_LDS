
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import random

logger = logging.getLogger(__name__)


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout

    def forward(self, x, adj=None):
        if adj is None:
            batch_size, num_features = x.shape
            adj = torch.eye(batch_size).to(x.device)
            indices = torch.nonzero(adj, as_tuple=False).t()
            values = adj[indices[0], indices[1]]
            adj = torch.sparse_coo_tensor(indices, values, adj.shape, device=x.device)

        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.25)

        return x

class MultiOmicsGATModel(nn.Module):

    def __init__(self, omics_dims: Dict[str, int], hidden_dim: int = 128,
                 num_classes: int = 2, num_layers: int = 3, dropout: float = 0.3,
                 num_heads: int = 8):
        super().__init__()

        self.omics_dims = omics_dims
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads

        self.omics_encoders = nn.ModuleDict()
        self.gcn_dims = [hidden_dim, hidden_dim, hidden_dim]

        for omics_type, input_dim in omics_dims.items():
            self.omics_encoders[omics_type] = GCN_E(
                in_dim=input_dim,
                hgcn_dim=self.gcn_dims,
                dropout=dropout
            )

        self.cross_omics_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.gating_network = nn.Sequential(
            nn.Linear(hidden_dim * len(omics_dims), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(omics_dims)),
            nn.Softmax(dim=-1)
        )

        self.representation_layers = nn.ModuleList()
        for i in range(num_layers):
            self.representation_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )

        self.attention_weights = None
        self.gating_weights = None
        self.omics_embeddings = None

        self.apply(xavier_init)

    def forward(self, omics_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = list(omics_data.values())[0].size(0)
        device = list(omics_data.values())[0].device

        omics_embeddings = {}
        for omics_type, data in omics_data.items():
            embeddings = self.omics_encoders[omics_type](data)
            omics_embeddings[omics_type] = embeddings

        self.omics_embeddings = omics_embeddings

        omics_list = list(omics_embeddings.keys())
        embedded_omics = torch.stack([omics_embeddings[omics] for omics in omics_list], dim=1)

        attended_omics, attention_weights = self.cross_omics_attention(
            embedded_omics, embedded_omics, embedded_omics
        )
        if attention_weights is not None:
            self.attention_weights = attention_weights
        else:
            num_omics = len(omics_list)
            self.attention_weights = torch.eye(num_omics).unsqueeze(0).repeat(batch_size, 1, 1)

        concat_omics = attended_omics.contiguous().reshape(batch_size, -1)
        gating_weights = self.gating_network(concat_omics)
        self.gating_weights = gating_weights

        weighted_omics = (attended_omics * gating_weights.unsqueeze(-1)).sum(dim=1)

        patient_repr = weighted_omics
        for layer in self.representation_layers:
            patient_repr = layer(patient_repr)

        logits = self.classifier(patient_repr)

        return logits

    def get_attention_weights(self) -> torch.Tensor:
        return self.attention_weights

    def get_gating_weights(self) -> torch.Tensor:
        return self.gating_weights

    def get_omics_embeddings(self) -> Dict[str, torch.Tensor]:
        return self.omics_embeddings


class ModelTrainer:

    def __init__(self, model: MultiOmicsGATModel, device: str = None, random_state: int = 42):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self._set_random_seeds(random_state)

        self.model = model.to(self.device)
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    def _set_random_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train_model(self, train_data: Dict[str, np.ndarray], train_labels: np.ndarray,
                   val_data: Dict[str, np.ndarray] = None, val_labels: np.ndarray = None,
                   epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-4,
                   patience: int = 30) -> Dict:

        train_tensors = self._convert_to_tensors(train_data)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(self.device)

        if val_data is not None:
            val_tensors = self._convert_to_tensors(val_data)
            val_labels_tensor = torch.tensor(val_labels, dtype=torch.long).to(self.device)

        class_weights = compute_class_weight(
            'balanced', classes=np.unique(train_labels), y=train_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        if epochs >= 10:
            try:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=1,
                    pct_start=0.3, anneal_strategy='cos'
                )
            except:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=lr*0.01
                )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs, 2), eta_min=lr*0.01
            )

        best_val_acc = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            train_logits = self.model(train_tensors)
            train_loss = criterion(train_logits, train_labels_tensor)

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                train_pred = torch.argmax(train_logits, dim=1)
                train_acc = accuracy_score(train_labels, train_pred.cpu())

            self.training_history['train_loss'].append(train_loss.item())
            self.training_history['train_acc'].append(train_acc)

            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(val_tensors)
                    val_loss = criterion(val_logits, val_labels_tensor)
                    val_pred = torch.argmax(val_logits, dim=1)
                    val_acc = accuracy_score(val_labels, val_pred.cpu())

                self.training_history['val_loss'].append(val_loss.item())
                self.training_history['val_acc'].append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.training_history

    def evaluate_model(self, test_data: Dict[str, np.ndarray],
                      test_labels: np.ndarray) -> Dict:

        self.model.eval()
        test_tensors = self._convert_to_tensors(test_data)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(test_tensors)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

        y_true = test_labels
        y_pred = predictions.cpu().numpy()
        y_proba = probabilities.cpu().numpy()

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        try:
            if len(np.unique(y_true)) > 2:
                auc_score = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            else:
                auc_score = roc_auc_score(y_true, y_proba[:, 1])
        except:
            auc_score = None

        results = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba,
            'true_labels': y_true
        }

        return results

    def _convert_to_tensors(self, data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        tensors = {}
        for omics_type, array in data.items():
            tensors[omics_type] = torch.tensor(array, dtype=torch.float32).to(self.device)
        return tensors

    def save_model(self, filepath: str, metadata: Dict = None):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'omics_dims': self.model.omics_dims,
                'hidden_dim': self.model.hidden_dim,
                'num_classes': self.model.num_classes,
                'num_layers': self.model.num_layers,
                'dropout': self.model.dropout,
                'num_heads': self.model.num_heads
            },
            'training_history': self.training_history
        }

        if metadata:
            save_dict['metadata'] = metadata

        torch.save(save_dict, filepath)


class CrossValidator:
    def __init__(self, model_class: type, model_params: Dict, save_dir: str = None):
        self.model_class = model_class
        self.model_params = model_params
        self.cv_results = []
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "models"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "predictions"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "probabilities"), exist_ok=True)

    def run_cross_validation(self, omics_data: Dict[str, np.ndarray], labels: np.ndarray,
                             cv_splits: List[Tuple], train_params: Dict = None,
                             data_processor=None) -> Dict:
        if train_params is None:
            train_params = {}

        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(cv_splits):
            train_data_raw = {k: v[train_idx] for k, v in omics_data.items()}
            test_data_raw = {k: v[test_idx] for k, v in omics_data.items()}
            train_labels = labels[train_idx]
            test_labels = labels[test_idx]

            if data_processor is not None:
                result = data_processor.preprocess_fold_data(train_data_raw, test_data_raw, train_labels, fold_id=fold + 1)
                if len(result) == 3:
                    train_data, test_data, modified_train_labels = result
                    train_labels = modified_train_labels
                else:
                    train_data, test_data = result
            else:
                train_data = train_data_raw
                test_data = test_data_raw

            fold_model_params = self.model_params.copy()
            fold_model_params["omics_dims"] = {k: v.shape[1] for k, v in train_data.items()}
            model = self.model_class(**fold_model_params)
            trainer = ModelTrainer(model, random_state=42)
            trainer.train_model(train_data, train_labels, **train_params)
            results = trainer.evaluate_model(test_data, test_labels)
            results["fold"] = fold + 1
            results["model_trainer"] = trainer

            if self.save_dir:
                self._save_fold_results(fold + 1, trainer, results, test_idx)
            fold_results.append(results)

        accuracies = [r["accuracy"] for r in fold_results]
        auc_scores = [r["auc_score"] for r in fold_results if r["auc_score"] is not None]
        cv_summary = {
            "fold_results": fold_results,
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "mean_auc": np.mean(auc_scores) if auc_scores else None,
            "std_auc": np.std(auc_scores) if auc_scores else None,
        }
        self.cv_results = cv_summary
        if self.save_dir:
            self.save_overall_results()
        return cv_summary

    def _save_fold_results(self, fold: int, trainer: ModelTrainer, results: Dict, test_indices: np.ndarray):
        import json
        from datetime import datetime

        timestamp = "reproducible_run"

        model_path = os.path.join(self.save_dir, "models", f"fold_{fold}_model.pth")
        model_metadata = {
            'fold': fold,
            'accuracy': results['accuracy'],
            'auc_score': results['auc_score'],
            'timestamp': timestamp,
            'test_indices': test_indices.tolist()
        }
        trainer.save_model(model_path, metadata=model_metadata)

        predictions_data = {
            'fold': fold,
            'test_indices': test_indices.tolist(),
            'true_labels': results['true_labels'].tolist(),
            'predictions': results['predictions'].tolist(),
            'accuracy': results['accuracy'],
            'auc_score': results['auc_score'],
            'classification_report': results['classification_report'],
            'timestamp': timestamp
        }

        predictions_path = os.path.join(self.save_dir, "predictions", f"fold_{fold}_predictions.json")
        with open(predictions_path, 'w', encoding='utf-8') as f:
            json.dump(predictions_data, f, indent=2, ensure_ascii=False)

        probabilities_data = {
            'fold': fold,
            'test_indices': test_indices.tolist(),
            'probabilities': results['probabilities'].tolist(),
            'class_names': [f'class_{i}' for i in range(results['probabilities'].shape[1])],
            'timestamp': timestamp
        }

        probabilities_path = os.path.join(self.save_dir, "probabilities", f"fold_{fold}_probabilities.json")
        with open(probabilities_path, 'w', encoding='utf-8') as f:
            json.dump(probabilities_data, f, indent=2, ensure_ascii=False)

        cm_path = os.path.join(self.save_dir, "predictions", f"fold_{fold}_confusion_matrix.csv")
        cm_df = pd.DataFrame(results['confusion_matrix'])
        cm_df.to_csv(cm_path, index=False)


    def save_overall_results(self):
        if not self.cv_results or not self.save_dir:
            return

        import json
        from datetime import datetime

        timestamp = "reproducible_run"

        summary_results = self.cv_results.copy()
        clean_fold_results = []

        for fold_result in summary_results['fold_results']:
            clean_result = fold_result.copy()
            clean_result.pop('model_trainer', None)
            if 'confusion_matrix' in clean_result:
                clean_result['confusion_matrix'] = clean_result['confusion_matrix'].tolist()
            if 'predictions' in clean_result:
                clean_result['predictions'] = clean_result['predictions'].tolist()
            if 'probabilities' in clean_result:
                clean_result['probabilities'] = clean_result['probabilities'].tolist()
            if 'true_labels' in clean_result:
                clean_result['true_labels'] = clean_result['true_labels'].tolist()
            clean_fold_results.append(clean_result)

        summary_results['fold_results'] = clean_fold_results
        summary_results['timestamp'] = timestamp

        summary_path = os.path.join(self.save_dir, "cv_summary_results.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)


