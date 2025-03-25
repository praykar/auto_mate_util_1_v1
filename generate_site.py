import os
import json
import nbformat
import requests
from jinja2 import Environment, FileSystemLoader

class SiteGenerator:
    def __init__(self, notebooks_dir='sample_notebooks', output_dir='docs'):
        """
        Initialize site generator
        
        :param notebooks_dir: Directory containing sample notebooks
        :param output_dir: Output directory for generated site
        """
        self.notebooks_dir = notebooks_dir
        self.output_dir = output_dir
        
        # Ensure directories exist
        os.makedirs(self.notebooks_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup Jinja2 template environment
        self.template_env = Environment(loader=FileSystemLoader('templates'))
    
    def process_notebooks(self):
        """
        Process all notebooks in the notebooks directory
        """
        notebooks = []
        
        # Create sample notebooks if none exist
        if not os.listdir(self.notebooks_dir):
            self._create_sample_notebooks()
        
        for filename in os.listdir(self.notebooks_dir):
            if filename.endswith('.ipynb'):
                filepath = os.path.join(self.notebooks_dir, filename)
                notebook_data = self._process_notebook(filepath)
                notebooks.append(notebook_data)
        
        return notebooks
    
    def _process_notebook(self, filepath):
        """
        Process individual notebook
        
        :param filepath: Path to notebook file
        :return: Processed notebook data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Basic notebook metadata extraction
        metadata = {
            'name': os.path.basename(filepath),
            'cells': len(nb.cells),
            'ml_type': self._detect_ml_type(nb)
        }
        
        return metadata
    
    def _detect_ml_type(self, notebook):
        """
        Detect machine learning type from notebook
        
        :param notebook: Parsed notebook object
        :return: Detected ML type
        """
        ml_types = {
            'classification': ['sklearn.classification', 'logistic_regression'],
            'regression': ['sklearn.regression', 'linear_regression'],
            'neural_network': ['tensorflow', 'keras', 'pytorch'],
            'clustering': ['kmeans', 'dbscan']
        }
        
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell_text = cell.source.lower()
                for ml_type, keywords in ml_types.items():
                    if any(keyword in cell_text for keyword in keywords):
                        return ml_type
        
        return 'unknown'
    
    def _create_sample_notebooks(self):
        """
        Create sample notebooks if none exist
        """
        sample_notebooks = [
            {
                'filename': 'classification_example.ipynb',
                'content': {
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "source": "# Iris Classification Example"
                        },
                        {
                            "cell_type": "code",
                            "source": "from sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression"
                        }
                    ]
                }
            }
        ]
        
        for sample in sample_notebooks:
            filepath = os.path.join(self.notebooks_dir, sample['filename'])
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sample['content'], f)
    
    def generate_site(self):
        """
        Generate static site
        """
        # Process notebooks
        notebooks = self.process_notebooks()
        
        # Generate index page
        index_template = self.template_env.get_template('index.html')
        index_html = index_template.render(notebooks=notebooks)
        
        with open(os.path.join(self.output_dir, 'index.html'), 'w', encoding='utf-8') as f:
            f.write(index_html)
        
        # Generate individual notebook pages
        notebook_template = self.template_env.get_template('notebook.html')
        for notebook in notebooks:
            notebook_html = notebook_template.render(notebook=notebook)
            output_path = os.path.join(
                self.output_dir, 
                f"{notebook['name'].replace('.ipynb', '')}.html"
            )
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(notebook_html)

def main():
    generator = SiteGenerator()
    generator.generate_site()

if __name__ == '__main__':
    main()