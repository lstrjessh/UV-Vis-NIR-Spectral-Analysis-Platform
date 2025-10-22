"""
Export utilities for models and results.
"""

import pickle
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
import zipfile
import io


class ModelExporter:
    """Export and import trained models."""
    
    @staticmethod
    def save_model(model_result: Any, filepath: str, format: str = 'pickle'):
        """
        Save model to file.
        
        Args:
            model_result: ModelResult object
            filepath: Output file path
            format: Export format ('pickle', 'joblib', 'onnx')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(model_result, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        elif format == 'joblib':
            joblib.dump(model_result, filepath, compress=3)
            
        elif format == 'onnx':
            # ONNX export for compatible models
            try:
                import skl2onnx
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                
                # Get input dimensions
                if hasattr(model_result.model, 'n_features_in_'):
                    n_features = model_result.model.n_features_in_
                else:
                    n_features = model_result.config.hyperparameters.get('n_features', 100)
                
                initial_type = [('float_input', FloatTensorType([None, n_features]))]
                onx = convert_sklearn(model_result.model, initial_types=initial_type)
                
                with open(filepath, "wb") as f:
                    f.write(onx.SerializeToString())
                    
            except ImportError:
                raise ImportError("ONNX export requires skl2onnx. Install with: pip install skl2onnx")
            except Exception as e:
                raise ValueError(f"ONNX export failed: {str(e)}")
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def load_model(filepath: str, format: str = 'pickle'):
        """
        Load model from file.
        
        Args:
            filepath: Input file path
            format: File format
            
        Returns:
            Loaded model result
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
                
        elif format == 'joblib':
            return joblib.load(filepath)
            
        elif format == 'onnx':
            try:
                import onnxruntime as ort
                
                # Create ONNX runtime session
                session = ort.InferenceSession(str(filepath))
                
                # Wrap in a compatible interface
                class ONNXModel:
                    def __init__(self, session):
                        self.session = session
                        self.input_name = session.get_inputs()[0].name
                        self.output_name = session.get_outputs()[0].name
                    
                    def predict(self, X):
                        return self.session.run(
                            [self.output_name],
                            {self.input_name: X.astype(np.float32)}
                        )[0]
                
                return ONNXModel(session)
                
            except ImportError:
                raise ImportError("ONNX runtime required. Install with: pip install onnxruntime")
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def export_ensemble(
        model_results: Dict[str, Any],
        output_dir: str,
        format: str = 'pickle'
    ):
        """
        Export multiple models as ensemble.
        
        Args:
            model_results: Dictionary of model results
            output_dir: Output directory
            format: Export format
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for name, result in model_results.items():
            filepath = output_dir / f"{name}.{format if format != 'onnx' else 'onnx'}"
            ModelExporter.save_model(result, filepath, format)
        
        # Save metadata
        metadata = {
            'models': list(model_results.keys()),
            'format': format,
            'exported_at': datetime.now().isoformat(),
            'n_models': len(model_results)
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def create_deployment_package(
        model_result: Any,
        output_path: str,
        include_preprocessor: bool = True,
        include_requirements: bool = True
    ):
        """
        Create deployment package with model and dependencies.
        
        Args:
            model_result: Model result to package
            output_path: Output ZIP file path
            include_preprocessor: Include preprocessing pipeline
            include_requirements: Include requirements.txt
        """
        output_path = Path(output_path)
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Save model
            model_bytes = io.BytesIO()
            pickle.dump(model_result, model_bytes)
            zf.writestr('model.pkl', model_bytes.getvalue())
            
            # Save config
            config_dict = {
                'name': model_result.config.name,
                'hyperparameters': model_result.hyperparameters,
                'metrics': model_result.metrics.to_dict()
            }
            zf.writestr('config.json', json.dumps(config_dict, indent=2))
            
            # Save preprocessor if available
            if include_preprocessor and hasattr(model_result, 'preprocessor'):
                prep_bytes = io.BytesIO()
                pickle.dump(model_result.preprocessor, prep_bytes)
                zf.writestr('preprocessor.pkl', prep_bytes.getvalue())
            
            # Create requirements.txt
            if include_requirements:
                requirements = [
                    'numpy>=1.20.0',
                    'scikit-learn>=1.0.0',
                    'pandas>=1.3.0',
                    'joblib>=1.0.0'
                ]
                
                # Add model-specific requirements
                if 'xgboost' in model_result.config.name.lower():
                    requirements.append('xgboost>=1.5.0')
                if 'torch' in str(type(model_result.model)):
                    requirements.append('torch>=1.9.0')
                
                zf.writestr('requirements.txt', '\n'.join(requirements))
            
            # Create simple prediction script
            prediction_script = '''
import pickle
import numpy as np

# Load model
with open('model.pkl', 'rb') as f:
    model_result = pickle.load(f)

def predict(X):
    """Make predictions using the loaded model."""
    return model_result.model.predict(X)

if __name__ == "__main__":
    # Example usage
    X_example = np.random.randn(10, 100)  # Adjust dimensions as needed
    predictions = predict(X_example)
    print(f"Predictions: {predictions}")
'''
            zf.writestr('predict.py', prediction_script)


class ResultsExporter:
    """Export analysis results in various formats."""
    
    @staticmethod
    def export_to_excel(
        model_results: Dict[str, Any],
        filepath: str,
        include_predictions: bool = True,
        include_hyperparameters: bool = True
    ):
        """
        Export results to Excel with multiple sheets.
        
        Args:
            model_results: Dictionary of model results
            filepath: Output Excel file path
            include_predictions: Include prediction sheets
            include_hyperparameters: Include hyperparameter sheets
        """
        filepath = Path(filepath)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for name, result in model_results.items():
                summary_data.append({
                    'Model': name,
                    'R2': result.metrics.r2,
                    'RMSE': result.metrics.rmse,
                    'MAE': result.metrics.mae,
                    'Training_Time': result.metrics.training_time
                })
            
            pd.DataFrame(summary_data).to_excel(
                writer, sheet_name='Summary', index=False
            )
            
            # Individual model sheets
            for name, result in model_results.items():
                # Metrics
                metrics_df = pd.DataFrame([result.metrics.to_dict()])
                metrics_df.to_excel(
                    writer, sheet_name=f'{name}_Metrics', index=False
                )
                
                # Predictions
                if include_predictions and result.predictions is not None:
                    pred_df = pd.DataFrame({
                        'Predictions': result.predictions,
                        'Residuals': result.residuals if result.residuals is not None else []
                    })
                    pred_df.to_excel(
                        writer, sheet_name=f'{name}_Predictions', index=False
                    )
                
                # Hyperparameters
                if include_hyperparameters and result.hyperparameters:
                    hyper_df = pd.DataFrame([result.hyperparameters])
                    hyper_df.to_excel(
                        writer, sheet_name=f'{name}_Hyperparams', index=False
                    )
    
    @staticmethod
    def export_to_csv(
        model_results: Dict[str, Any],
        output_dir: str
    ):
        """
        Export results to CSV files.
        
        Args:
            model_results: Dictionary of model results
            output_dir: Output directory for CSV files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Summary CSV
        summary_data = []
        for name, result in model_results.items():
            row = {
                'Model': name,
                **result.metrics.to_dict()
            }
            if result.hyperparameters:
                row.update({f'param_{k}': v for k, v in result.hyperparameters.items()})
            summary_data.append(row)
        
        pd.DataFrame(summary_data).to_csv(
            output_dir / 'summary.csv', index=False
        )
        
        # Individual model CSVs
        for name, result in model_results.items():
            # Predictions
            if result.predictions is not None:
                pred_df = pd.DataFrame({
                    'Predictions': result.predictions,
                    'Residuals': result.residuals if result.residuals is not None else []
                })
                pred_df.to_csv(
                    output_dir / f'{name}_predictions.csv', index=False
                )
            
            # Feature importance
            if result.feature_importance:
                importance_df = pd.DataFrame(
                    list(result.feature_importance.items()),
                    columns=['Feature', 'Importance']
                )
                importance_df.to_csv(
                    output_dir / f'{name}_feature_importance.csv', index=False
                )
    
    @staticmethod
    def generate_report(
        model_results: Dict[str, Any],
        output_path: str,
        format: str = 'html'
    ):
        """
        Generate comprehensive report.
        
        Args:
            model_results: Dictionary of model results
            output_path: Output file path
            format: Report format ('html', 'markdown')
        """
        from ..utils.metrics import compare_models, performance_summary_table
        
        # Create comparison
        comparison_df = compare_models(model_results)
        
        if format == 'html':
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Calibration Model Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #0066cc; }}
    </style>
</head>
<body>
    <h1>Calibration Model Analysis Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Model Performance Summary</h2>
    {comparison_df.to_html(index=False)}
    
    <h2>Best Model</h2>
    <p>Based on average ranking: <span class="metric">{comparison_df.iloc[0]['Model']}</span></p>
    
    <h2>Individual Model Details</h2>
"""
            
            for name, result in model_results.items():
                html_content += f"""
    <h3>{name}</h3>
    <ul>
        <li>R² Score: <span class="metric">{result.metrics.r2:.4f}</span></li>
        <li>RMSE: <span class="metric">{result.metrics.rmse:.4f}</span></li>
        <li>MAE: <span class="metric">{result.metrics.mae:.4f}</span></li>
        <li>Training Time: <span class="metric">{result.metrics.training_time:.2f}s</span></li>
    </ul>
"""
            
            html_content += """
</body>
</html>
"""
            
            with open(output_path, 'w') as f:
                f.write(html_content)
                
        elif format == 'markdown':
            md_content = f"""# Calibration Model Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Performance Summary

{comparison_df.to_markdown(index=False)}

## Best Model
Based on average ranking: **{comparison_df.iloc[0]['Model']}**

## Individual Model Details
"""
            
            for name, result in model_results.items():
                md_content += f"""
### {name}
- R² Score: **{result.metrics.r2:.4f}**
- RMSE: **{result.metrics.rmse:.4f}**
- MAE: **{result.metrics.mae:.4f}**
- Training Time: **{result.metrics.training_time:.2f}s**
"""
            
            with open(output_path, 'w') as f:
                f.write(md_content)
