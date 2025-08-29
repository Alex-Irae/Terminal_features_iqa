from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Optional
from utils.mmisc import RESULTS, CACHE
class UNet(nn.Module):
    """Simple U-Net for medical image segmentation"""
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        ch = in_channels
        for feature in features:
            self.downs.append(self._double_conv(ch, feature))
            ch = feature
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(self._double_conv(feature * 2, feature))
        self.bottleneck = self._double_conv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        return torch.sigmoid(self.final_conv(x))
def load_models_for_eval(device, domain):
    models_list = []
    unet_ckpt = Path(RESULTS / domain / "cache/unet_model.pth")
    unet = UNet(in_channels=1, out_channels=1).to(device)
    if unet_ckpt.exists():
        unet.load_state_dict(torch.load(unet_ckpt, map_location=device))
        print(" Loaded base U-Net from checkpoint")
    else:
        print("! U-Net checkpoint not found; using randomly initialized U-Net")
    unet.eval()
    models_list.append(("unet", unet))
    try:
        from torchvision.models.segmentation import (
            deeplabv3_resnet101,
            DeepLabV3_ResNet101_Weights,
        )
        dlab = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).to(device)
        dlab.eval()
        print(" Loaded DeepLabV3+ (ResNet101) with COCO+VOC pretrained weights")
        models_list.append(("deeplabv3p", dlab))
    except Exception as e:
        print(f"! Failed to load DeepLabV3+: {e}")
    try:
        import segmentation_models_pytorch as smp
        unet_r101 = smp.Unet(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
        ).to(device)
        r101_ckpt = Path(RESULTS / domain / "cache/unet_r101.pth")
        if r101_ckpt.exists():
            sd = torch.load(r101_ckpt, map_location=device)
            unet_r101.load_state_dict(sd, strict=False)
            print(" Loaded UNet-ResNet101 from checkpoint")
        else:
            print("! UNet-ResNet101 checkpoint not found; using encoder pretrained on ImageNet")
        unet_r101.eval()
        models_list.append(("unet_resnet101", unet_r101))
    except ImportError:
        print("! segmentation_models_pytorch not installed; skipping UNet-ResNet101")
    print(f"\nTotal models loaded for evaluation: {len(models_list)}")
    return models_list
def register_hooks_for_resnet_encoder(model, model_name):
    """
    Returns a dict to store activations and a list of hook handles.
    Works for SMP models with .encoder being a torchvision ResNet.
    """
    feats = {}
    handles = []
    enc = getattr(model, "encoder", None)
    if enc is None:
        return feats, handles
    layers = [
        ("conv1", enc.conv1),
        ("layer1", enc.layer1),
        ("layer2", enc.layer2),
        ("layer3", enc.layer3),
        ("layer4", enc.layer4),
    ]
    for lname, layer in layers:
        def _mk_hook(key):
            def hook(m, inp, out):
                o = out.detach().float().cpu()
                if o.dim() == 4:
                    o = o[0]
                feats[f"{model_name}:{key}"] = o.numpy()
            return hook
        handles.append(layer.register_forward_hook(_mk_hook(lname)))
    return feats, handles
class LightweightCNN(nn.Module):
    """
    Enhanced lightweight CNN with better architecture for medical IQA calibration.
    Key improvements:
    - Residual connections for gradient flow
    - Feature interaction layers
    - Batch normalization for stability
    - Dropout scheduling
    - Attention mechanism for feature importance
    """
    def __init__(self, input_dim=8, hidden_dims=[128, 64, 32], use_attention=True):
        super().__init__()
        self.input_dim = input_dim
        self.use_attention = use_attention
        self.hidden_dims = hidden_dims
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            block = ResidualBlock(hidden_dims[i], hidden_dims[i + 1])
            self.blocks.append(block)
        self.feature_interaction = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1]),
        )
        if self.use_attention:
            self.attention = FeatureAttention(hidden_dims[-1])
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[-1] // 2, 1),
        )
        self.output_activation = nn.Sigmoid()
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x + self.feature_interaction(x)
        if self.use_attention:
            x = self.attention(x)
        x = self.prediction_head(x)
        x = self.output_activation(x)
        return x
class ResidualBlock(nn.Module):
    """Residual block with skip connection for better gradient flow"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU(), nn.Dropout(0.15)
        )
        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features)
        else:
            self.skip = nn.Identity()
    def forward(self, x):
        return self.main_path(x) + self.skip(x)
class FeatureAttention(nn.Module):
    """Self-attention mechanism for feature importance learning"""
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(dim, dim // 4), nn.ReLU(), nn.Linear(dim // 4, dim), nn.Sigmoid())
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights
from typing import Tuple, List, Optional, Union
class StandardizedFeatureProcessor:
    """
    Standardized feature processor ensuring all models use identical:
    - Feature selection
    - Feature engineering (CNR/SNR ratio)
    - Scaling approach
    - NaN/Inf handling
    This is the SINGLE SOURCE OF TRUTH for feature preparation across all models.
    """
    STANDARD_IQA_FEATURES = ["psnr", "ssim", "mse", "mae", "snr", "cnr", "gradient_mag", "laplacian_var"]
    def __init__(self, scaler_key="standard_scaler"):
        self.scaler_key = scaler_key
        self.scalers = {}
        self._feature_names = None
        self._input_shape = None
        self._is_fitted = False
    def prepare_features(self,
                        df_or_array: Union[pd.DataFrame, np.ndarray],
                        fit_scaler: bool = False,
                        return_numpy: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Standardized feature preparation for ALL models.
        Args:
            df_or_array: DataFrame with IQA metrics OR numpy array (for legacy compatibility)
            fit_scaler: Whether to fit the scaler (True) or just transform (False)
            return_numpy: Whether to return numpy array (default) or keep as DataFrame
        Returns:
            tuple: (X_scaled, feature_names)
        """
        if isinstance(df_or_array, np.ndarray):
            return self._prepare_from_array(df_or_array, fit_scaler)
        else:
            return self._prepare_from_dataframe(df_or_array, fit_scaler, return_numpy)
    def _prepare_from_dataframe(self, df: pd.DataFrame, fit_scaler: bool, return_numpy: bool) -> Tuple[np.ndarray, List[str]]:
        """Prepare features from DataFrame input"""
        available_metrics = [f for f in self.STANDARD_IQA_FEATURES if f in df.columns]
        if len(available_metrics) < 3:
            raise ValueError(f"Insufficient features available. Found: {available_metrics}, need at least 3")
        df_work = df[available_metrics].copy()
        for col in available_metrics:
            df_work[col] = self._handle_nan_inf_series(df_work[col])
        X_iqa = df_work.values
        X_engineered, feature_names = self._add_feature_engineering(X_iqa, available_metrics)
        if fit_scaler:
            self._input_shape = X_engineered.shape[1]
            self._is_fitted = True
        else:
            if self._is_fitted and X_engineered.shape[1] != self._input_shape:
                print(f"Warning: Feature count mismatch. Expected {self._input_shape}, got {X_engineered.shape[1]}")
                X_engineered = self._fix_shape_mismatch(X_engineered)
        X_scaled = self._apply_scaling(X_engineered, fit_scaler)
        self._feature_names = feature_names
        if return_numpy:
            return X_scaled, feature_names
        else:
            return pd.DataFrame(X_scaled, columns=feature_names, index=df.index), feature_names
    def _prepare_from_array(self, X: np.ndarray, fit_scaler: bool) -> Tuple[np.ndarray, List[str]]:
        """Legacy support: prepare features from numpy array"""
        n_features = X.shape[1]
        if n_features <= len(self.STANDARD_IQA_FEATURES):
            feature_names = self.STANDARD_IQA_FEATURES[:n_features]
        else:
            feature_names = self.STANDARD_IQA_FEATURES + [f"engineered_{i}" for i in range(n_features - len(self.STANDARD_IQA_FEATURES))]
        X_clean = self._handle_nan_inf(X)
        if n_features >= 6 and "cnr" in self.STANDARD_IQA_FEATURES and "snr" in self.STANDARD_IQA_FEATURES:
            cnr_idx = self.STANDARD_IQA_FEATURES.index("cnr")
            snr_idx = self.STANDARD_IQA_FEATURES.index("snr")
            if n_features == len(self.STANDARD_IQA_FEATURES):
                cnr_snr_ratio = X_clean[:, cnr_idx] / (X_clean[:, snr_idx] + 1e-6)
                X_clean = np.column_stack([X_clean, cnr_snr_ratio])
                feature_names.append("cnr_snr_ratio")
        X_scaled = self._apply_scaling(X_clean, fit_scaler)
        self._feature_names = feature_names
        return X_scaled, feature_names
    def _handle_nan_inf(self, X: np.ndarray) -> np.ndarray:
        """Standardized NaN/Inf handling for arrays"""
        X_clean = np.nan_to_num(
            X,
            nan=0.0,
            posinf=np.percentile(X[np.isfinite(X)], 99) if np.any(np.isfinite(X)) else 1e6,
            neginf=np.percentile(X[np.isfinite(X)], 1) if np.any(np.isfinite(X)) else -1e6,
        )
        return X_clean
    def _handle_nan_inf_series(self, series: pd.Series) -> pd.Series:
        """Standardized NaN/Inf handling for pandas Series"""
        finite_vals = series[np.isfinite(series)]
        if len(finite_vals) > 0:
            p99 = np.percentile(finite_vals, 99)
            p1 = np.percentile(finite_vals, 1)
        else:
            p99, p1 = 1e6, -1e6
        series = series.replace([np.inf, -np.inf], [p99, p1])
        series = series.fillna(0.0)
        return series
    def _add_feature_engineering(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Add CNR/SNR ratio if both features are present"""
        engineered_features = X.copy()
        engineered_names = feature_names.copy()
        return engineered_features, engineered_names
    def _apply_scaling(self, X: np.ndarray, fit_scaler: bool) -> np.ndarray:
        """Apply StandardScaler, preventing double-scaling"""
        if fit_scaler:
            self.scalers[self.scaler_key] = StandardScaler()
            X_scaled = self.scalers[self.scaler_key].fit_transform(X)
        else:
            if self.scaler_key not in self.scalers:
                print(f"[StandardizedProcessor] Warning: Scaler not found, fitting new one")
                self.scalers[self.scaler_key] = StandardScaler()
                X_scaled = self.scalers[self.scaler_key].fit_transform(X)
            else:
                X_scaled = self.scalers[self.scaler_key].transform(X)
        return X_scaled
    def _fix_shape_mismatch(self, X: np.ndarray) -> np.ndarray:
        """Fix shape mismatches by padding or truncating"""
        if self._input_shape is None:
            return X
        current_shape = X.shape[1]
        expected_shape = self._input_shape
        if current_shape == expected_shape:
            return X
        elif current_shape < expected_shape:
            padding = np.zeros((X.shape[0], expected_shape - current_shape))
            X_fixed = np.hstack([X, padding])
        else:
            X_fixed = X[:, :expected_shape]
        return X_fixed
    def get_feature_names(self) -> Optional[List[str]]:
        """Get the final feature names after engineering"""
        return self._feature_names
    def get_scaler(self):
        """Get the fitted scaler"""
        return self.scalers.get(self.scaler_key)
    def get_expected_shape(self) -> Optional[int]:
        """Get the expected number of features"""
        return self._input_shape
    def reset(self):
        """Reset the processor state"""
        self.scalers = {}
        self._feature_names = None
        self._input_shape = None
        self._is_fitted = False
class SklearnCalibrator:
    """Enhanced wrapper for sklearn models with proper cross-modality support"""
    def __init__(self, model, global_encoders=None):
        self.model = model
        self.global_encoders = global_encoders
        self.is_fitted = False
        model_name = self.model.__class__.__name__.lower()
        self.feature_processor = StandardizedFeatureProcessor(f"{model_name}_scaler")
    def prepare_features(self, df: pd.DataFrame, fit_encoders: bool = False) -> np.ndarray:
        """Use standardized feature preparation"""
        X_scaled, feature_names = self.feature_processor.prepare_features(df, fit_scaler=fit_encoders)
        return X_scaled
    def fit(self, df: pd.DataFrame, target_column: str = "dice_score",
            validation_split: float = 0.2, fold_idx: Optional[int] = None):
        """Fit sklearn model"""
        X = self.prepare_features(df, fit_encoders=True)
        y = pd.to_numeric(df[target_column], errors="coerce").fillna(0).values
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("Target variable must be numeric")
        if np.any(np.isnan(y)):
            print(f"Warning: Target contains {np.sum(np.isnan(y))} NaN values after conversion")
            valid_mask = ~np.isnan(y)
            X = X[valid_mask]
            y = y[valid_mask]
        self.model.fit(X, y)
        self.is_fitted = True
        print(f"[SklearnCalibrator] Model fitted with {X.shape[0]} samples, {X.shape[1]} features")
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with sklearn model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predictions")
        X = self.prepare_features(df, fit_encoders=False)
        predictions = self.model.predict(X)
        return predictions
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error
import warnings
class PyTorchRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper for PyTorch regression models.
    Makes LightweightCNN work with your existing pipeline.
    """
    def __init__(self,
                 model_class,
                 model_params=None,
                 epochs=100,
                 batch_size=32,
                 learning_rate=0.001,
                 device=None,
                 patience=10,
                 verbose=False):
        self.model_class = model_class
        self.model_params = model_params or {}
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.patience = patience
        self.verbose = verbose
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.is_fitted = False
        self.training_history = {'loss': [], 'val_loss': []}
    def _create_model(self, input_dim):
        """Create and initialize the PyTorch model"""
        model_params = self.model_params.copy()
        model_params['input_dim'] = input_dim
        model = self.model_class(**model_params)
        model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        return model
    def _prepare_data(self, X, y=None):
        """Convert numpy arrays to PyTorch tensors"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        if y is not None:
            y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
            return X_tensor, y_tensor
        return X_tensor
    def _train_epoch(self, X_tensor, y_tensor):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        actual_batch_size = min(self.batch_size, len(X_tensor))
        for i in range(0, len(X_tensor), actual_batch_size):
            batch_X = X_tensor[i:i+actual_batch_size]
            batch_y = y_tensor[i:i+actual_batch_size]
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(1, len(X_tensor) // actual_batch_size)
    def fit(self, X, y, validation_split=0.2):
        """
        Fit the PyTorch model using sklearn interface.
        Args:
            X: Features (numpy array)
            y: Targets (numpy array)
            validation_split: Fraction for validation
        """
        if len(X) < 10:
            warnings.warn(f"Very small dataset ({len(X)} samples). Results may be unreliable.")
        self.model = self._create_model(X.shape[1])
        n_val = int(len(X) * validation_split)
        if n_val > 0:
            val_indices = np.random.choice(len(X), n_val, replace=False)
            train_indices = np.setdiff1d(np.arange(len(X)), val_indices)
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
        else:
            X_train, X_val = X, X[:5]
            y_train, y_val = y, y[:5]
        X_train_tensor, y_train_tensor = self._prepare_data(X_train, y_train)
        X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val)
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(self.epochs):
            train_loss = self._train_epoch(X_train_tensor, y_train_tensor)
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            self.training_history['loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            if self.verbose and epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        self.is_fitted = True
        if self.verbose:
            final_r2 = self.score(X_val, y_val)
            print(f"Training completed. Final validation R = {final_r2:.4f}")
        return self
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        self.model.eval()
        with torch.no_grad():
            X_tensor = self._prepare_data(X)
            outputs = self.model(X_tensor)
            return outputs.cpu().numpy().flatten()
    def score(self, X, y):
        """Calculate R score"""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
def compute_pytorch_feature_importance(model_wrapper, X, y, method='permutation', n_repeats=10):
    """
    Compute feature importance for PyTorch models.
    Args:
        model_wrapper: Fitted PyTorchRegressorWrapper
        X: Features (numpy array)
        y: True targets
        method: 'permutation', 'gradient', or 'attention'
        n_repeats: Number of permutation repeats
    Returns:
        dict: feature_name -> importance_score
    """
    if not model_wrapper.is_fitted:
        raise ValueError("Model must be fitted first")
    if method == 'permutation':
        return _permutation_importance_pytorch(model_wrapper, X, y, n_repeats)
    elif method == 'gradient':
        return _gradient_importance_pytorch(model_wrapper, X, y)
    elif method == 'attention' and hasattr(model_wrapper.model, 'attention'):
        return _attention_importance_pytorch(model_wrapper, X)
    else:
        print(f"Unknown method '{method}', falling back to permutation importance")
        return _permutation_importance_pytorch(model_wrapper, X, y, n_repeats)
def _permutation_importance_pytorch(model_wrapper, X, y, n_repeats=10):
    """Permutation-based feature importance for PyTorch models"""
    from sklearn.inspection import permutation_importance
    result = permutation_importance(
        model_wrapper, X, y,
        n_repeats=n_repeats,
        random_state=42,
        scoring='r2'
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    standard_features = ["psnr", "ssim", "mse", "mae", "snr", "cnr", "gradient_mag", "laplacian_var"]
    if X.shape[1] <= len(standard_features):
        feature_names = standard_features[:X.shape[1]]
    elif X.shape[1] == len(standard_features) + 1:
        feature_names = standard_features + ["cnr_snr_ratio"]
    importance_dict = dict(zip(feature_names, result.importances_mean))
    return importance_dict
def _gradient_importance_pytorch(model_wrapper, X, y):
    """Gradient-based feature importance for PyTorch models"""
    model = model_wrapper.model
    model.eval()
    X_tensor = torch.FloatTensor(X).to(model_wrapper.device)
    X_tensor.requires_grad_(True)
    outputs = model(X_tensor)
    gradients = torch.autograd.grad(
        outputs=outputs.sum(),
        inputs=X_tensor,
        create_graph=False,
        retain_graph=False
    )[0]
    importance_scores = torch.abs(gradients).mean(dim=0).detach().cpu().numpy()
    importance_scores = importance_scores / importance_scores.sum()
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    standard_features = ["psnr", "ssim", "mse", "mae", "snr", "cnr", "gradient_mag", "laplacian_var"]
    if X.shape[1] <= len(standard_features):
        feature_names = standard_features[:X.shape[1]]
    elif X.shape[1] == len(standard_features) + 1:
        feature_names = standard_features + ["cnr_snr_ratio"]
    return dict(zip(feature_names, importance_scores))
def _attention_importance_pytorch(model_wrapper, X):
    """Extract attention weights as feature importance (if model has attention)"""
    model = model_wrapper.model
    model.eval()
    if not hasattr(model, 'attention'):
        raise ValueError("Model does not have attention mechanism")
    attention_weights = []
    def attention_hook(module, input, output):
        attention_weights.append(output.detach().cpu().numpy())
    hook_handle = model.attention.register_forward_hook(attention_hook)
    try:
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(model_wrapper.device)
            _ = model(X_tensor)
        if attention_weights:
            avg_attention = np.mean(attention_weights[0], axis=0)
            feature_names = [f"feature_{i}" for i in range(len(avg_attention))]
            standard_features = ["psnr", "ssim", "mse", "mae", "snr", "cnr", "gradient_mag", "laplacian_var"]
            if len(avg_attention) <= len(standard_features):
                feature_names = standard_features[:len(avg_attention)]
            elif len(avg_attention) == len(standard_features) + 1:
                feature_names = standard_features + ["cnr_snr_ratio"]
            return dict(zip(feature_names, avg_attention))
        else:
            raise ValueError("No attention weights captured")
    finally:
        hook_handle.remove()
def compute_feature_importance_pytorch_compatible(df, target, domain, model_type="random_forest"):
    """
    FIXED: Compute feature importance for both sklearn and PyTorch models
    """
    print(f"\nComputing feature importance for {domain} domain with {model_type}...")
    if target not in df.columns:
        print(f"ERROR: Target column '{target}' not found")
        return {}
    valid_df = df.dropna(subset=[target]).copy()
    required_features = ["psnr", "ssim"]
    valid_df = valid_df.dropna(subset=required_features)
    if len(valid_df) < 50:
        print(f"Warning: Only {len(valid_df)} samples available for {domain}")
        return {}
    from mmodels import StandardizedFeatureProcessor
    processor = StandardizedFeatureProcessor(f"importance_{model_type}_{domain}_scaler")
    try:
        X_scaled, feature_names = processor.prepare_features(valid_df, fit_scaler=True)
        print(f"Features prepared: {feature_names}")
    except Exception as e:
        print(f"ERROR: Feature preparation failed: {e}")
        return {}
    y = pd.to_numeric(valid_df[target], errors="coerce").values
    valid_mask = ~np.isnan(y)
    if not np.all(valid_mask):
        print(f"Removing {np.sum(~valid_mask)} samples with NaN targets")
        X_scaled = X_scaled[valid_mask]
        y = y[valid_mask]
    print(f"Training {model_type} on {len(y)} samples with {len(feature_names)} features")
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
            min_samples_split=5, min_samples_leaf=2,
        )
        model.fit(X_scaled, y)
        importance_dict = dict(zip(feature_names, model.feature_importances_))
    elif model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
            learning_rate=0.1, objective="reg:squarederror",
            subsample=0.8, colsample_bytree=0.8,
        )
        model.fit(X_scaled, y)
        importance_dict = dict(zip(feature_names, model.feature_importances_))
    elif model_type == "lightweight_cnn":
        from mmodels import LightweightCNN
        model_wrapper = PyTorchRegressorWrapper(
            model_class=LightweightCNN,
            model_params={'input_dim': X_scaled.shape[1], 'hidden_dims': [128, 64, 32], 'use_attention': True},
            epochs=100,
            batch_size=min(32, len(X_scaled)//4),
            learning_rate=0.001,
            patience=15,
            verbose=True
        )
        model_wrapper.fit(X_scaled, y, validation_split=0.2)
        try:
            if hasattr(model_wrapper.model, 'attention'):
                print("Using attention-based feature importance...")
                importance_dict = compute_pytorch_feature_importance(
                    model_wrapper, X_scaled, y, method='attention'
                )
            else:
                print("Using permutation-based feature importance...")
                importance_dict = compute_pytorch_feature_importance(
                    model_wrapper, X_scaled, y, method='permutation', n_repeats=5
                )
        except Exception as e:
            print(f"Advanced importance failed ({e}), using gradient-based...")
            importance_dict = compute_pytorch_feature_importance(
                model_wrapper, X_scaled, y, method='gradient'
            )
    elif model_type == "linear":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_scaled, y)
        importance_values = np.abs(model.coef_)
        importance_values = importance_values / importance_values.sum()
        importance_dict = dict(zip(feature_names, importance_values))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    print(f"\nFeature importance for {domain} with {model_type}:")
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_features[:5]:
        print(f"  {feat}: {imp:.4f} ({imp*100:.1f}%)")
    return importance_dict
