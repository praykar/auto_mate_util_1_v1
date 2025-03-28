import os
import nbformat
import requests
import github
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

class NotebookProcessor:
    def __init__(self, hf_api_token):
        """
        Initialize the notebook processor with Hugging Face Inference API
        
        :param hf_api_token: Hugging Face API token
        """
        self.hf_api_token = hf_api_token
        self.upload_dir = 'uploaded_notebooks'
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Hugging Face Inference API endpoint
        self.hf_inference_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
        self.headers = {
            "Authorization": f"Bearer {self.hf_api_token}",
            "Content-Type": "application/json"
        }

    def validate_notebook(self, file_path):
        """
        Validate that the uploaded file is a valid Jupyter notebook
        
        :param file_path: Path to the notebook file
        :return: Boolean indicating notebook validity
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            return True
        except Exception as e:
            print(f"Notebook validation error: {e}")
            return False

    def extract_notebook_content(self, file_path):
        """
        Extract and process notebook content
        
        :param file_path: Path to the notebook file
        :return: Dictionary containing processed notebook information
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        processed_content = {
            'code_cells': [],
            'markdown_cells': [],
            'outputs': [],
            'ml_components': self._identify_ml_components(nb)
        }

        for cell in nb.cells:
            if cell.cell_type == 'code':
                processed_content['code_cells'].append(cell.source)
                processed_content['outputs'].append(
                    self._process_cell_output(cell)
                )
            elif cell.cell_type == 'markdown':
                processed_content['markdown_cells'].append(cell.source)

        return processed_content

    def _identify_ml_components(self, notebook):
        """
        Identify machine learning components in the notebook
        
        :param notebook: Parsed notebook object
        :return: Dictionary of identified ML components
        """
        ml_keywords = [
            'train_test_split', 'model.fit', 'predict', 
            'accuracy_score', 'classification_report',
            'sklearn', 'tensorflow', 'torch', 'keras'
        ]
        
        ml_components = {
            'preprocessing': [],
            'model_type': None,
            'training': False,
            'evaluation': False
        }

        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell_text = cell.source.lower()
                
                # Detect preprocessing
                if any(keyword in cell_text for keyword in ['scale', 'normalize', 'preprocess']):
                    ml_components['preprocessing'].append(cell.source)
                
                # Detect model types
                if 'logistic_regression' in cell_text:
                    ml_components['model_type'] = 'Logistic Regression'
                elif 'random_forest' in cell_text:
                    ml_components['model_type'] = 'Random Forest'
                elif 'neural_network' in cell_text or 'tensorflow' in cell_text:
                    ml_components['model_type'] = 'Neural Network'
                
                # Detect training and evaluation
                if 'model.fit' in cell_text:
                    ml_components['training'] = True
                if 'accuracy_score' in cell_text or 'classification_report' in cell_text:
                    ml_components['evaluation'] = True

        return ml_components

    def _process_cell_output(self, cell):
        """
        Process and sanitize cell outputs
        
        :param cell: Notebook cell
        :return: Processed output or None
        """
        if hasattr(cell, 'outputs') and cell.outputs:
            for output in cell.outputs:
                if output.output_type in ['stream', 'display_data', 'execute_result']:
                    return output
        return None

    def generate_explanation(self, content):
        """
        Generate explanations for technical content using Hugging Face Inference API
        
        :param content: Processed notebook content
        :return: Generated explanations
        """
        def query_hf_api(prompt):
            """
            Helper function to query Hugging Face Inference API
            
            :param prompt: Input prompt for text generation
            :return: Generated text
            """
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 250,
                    "return_full_text": False
                }
            }
            
            try:
                response = requests.post(
                    self.hf_inference_url, 
                    headers=self.headers, 
                    json=payload
                )
                response.raise_for_status()
                return response.json()[0]['generated_text']
            except Exception as e:
                print(f"HF API Error: {e}")
                return "Unable to generate explanation due to API error."

        # Generate project overview
        overview_prompt = (
            f"Provide a concise, beginner-friendly overview of this machine learning project. "
            f"Project details: {content['ml_components']}"
        )
        overview = query_hf_api(overview_prompt)

        # Generate technical explanation
        technical_prompt = (
            f"Explain the technical implementation of this machine learning solution in a clear, "
            f"accessible manner. Break down key code components and their purpose. "
            f"Code snippets: {' '.join(content['code_cells'][:3])}"
        )
        technical_details = query_hf_api(technical_prompt)

        return {
            'overview': overview,
            'technical_details': technical_details
        }

class GitHubDeployer:
    def __init__(self, github_token, repo_name):
        """
        Initialize GitHub deployment service
        
        :param github_token: GitHub personal access token
        :param repo_name: Name of the GitHub repository
        """
        self.g = github.Github(github_token)
        self.repo_name = repo_name
        self.repo = self.g.get_repo(repo_name)
        self.upload_dir = 'uploaded_notebooks'
        self.template_dir = 'templates'
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.template_dir, exist_ok=True)
        self._create_index_html()  # Always create/update index.html
        self._ensure_gh_pages_branch()

    def _ensure_gh_pages_branch(self):
        """Create gh-pages branch if it doesn't exist"""
        try:
            self.repo.get_branch("gh-pages")
        except github.GithubException:
            # Create gh-pages branch
            master = self.repo.get_branch("main")
            self.repo.create_git_ref(ref="refs/heads/gh-pages", sha=master.commit.sha)
            # Create initial index.html
            self._create_index_html()

    def _create_index_html(self):
        """Create index.html in the upload directory"""
        # First create in templates if not exists
        template_path = os.path.join(self.template_dir, 'index.html')
        if not os.path.exists(template_path):
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write("""<!DOCTYPE html>
                <html>
                    <head>
                        <title>Auto Notebook Visualizations</title>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1">
                        <style>
                            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
                            .container { max-width: 1200px; margin: 0 auto; }
                            .header { background: #f4f4f4; padding: 20px; border-radius: 5px; }
                            .notebook-list { list-style: none; padding: 0; }
                            .notebook-list li { margin: 10px 0; }
                            .notebook-list a { 
                                text-decoration: none;
                                color: #0366d6;
                                padding: 10px;
                                display: block;
                                background: #f8f9fa;
                                border-radius: 4px;
                            }
                            .notebook-list a:hover { background: #e9ecef; }
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <div class="header">
                                <h1>ML Notebook Visualizations</h1>
                                <p>Generated visualizations of Machine Learning notebooks</p>
                            </div>
                            <ul class="notebook-list" id="notebook-list"></ul>
                        </div>
                        <script>
                            async function loadNotebooks() {
                                const pathParts = window.location.pathname.split('/');
                                const username = pathParts[1];
                                const repoName = pathParts[2];
                                const apiUrl = `https://api.github.com/repos/${username}/${repoName}/contents?ref=gh-pages`;
                                
                                try {
                                    const response = await fetch(apiUrl);
                                    const files = await response.json();
                                    const notebooks = files.filter(f => f.name.endsWith('.html') && f.name !== 'index.html');
                                    
                                    const list = document.getElementById('notebook-list');
                                    if (notebooks.length === 0) {
                                        list.innerHTML = '<li>No notebooks available yet</li>';
                                        return;
                                    }
                                    
                                    notebooks.forEach(nb => {
                                        const li = document.createElement('li');
                                        const a = document.createElement('a');
                                        a.href = nb.name;
                                        a.textContent = nb.name.replace('.html', '');
                                        li.appendChild(a);
                                        list.appendChild(li);
                                    });
                                } catch (error) {
                                    console.error('Error loading notebooks:', error);
                                    document.getElementById('notebook-list').innerHTML = 
                                        '<li>Error loading notebooks. Please try again later.</li>';
                                }
                            }
                            
                            loadNotebooks();
                        </script>
                    </body>
                </html>""")
        
        # Copy from templates to upload directory
        index_path = os.path.join(self.upload_dir, 'index.html')
        with open(template_path, 'r', encoding='utf-8') as src:
            with open(index_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())

    def deploy_content(self, content, notebook_name):
        """
        Deploy generated content to GitHub Pages
        
        :param content: HTML content to deploy
        :param notebook_name: Name of the source notebook
        """
        # First save locally
        file_path = os.path.join(self.upload_dir, notebook_name.replace(".ipynb", ".html"))
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        # Then push to gh-pages branch
        try:
            # Convert local path to repo path
            repo_path = file_path.replace('\\', '/')
            # Try to update existing file
            try:
                file = self.repo.get_contents(repo_path, ref="gh-pages")
                self.repo.update_file(
                    path=repo_path,
                    message=f"Update {notebook_name} visualization",
                    content=content,
                    sha=file.sha,
                    branch="gh-pages"
                )
            except github.GithubException:
                # File doesn't exist, create new one
                self.repo.create_file(
                    path=repo_path,
                    message=f"Add {notebook_name} visualization",
                    content=content,
                    branch="gh-pages"
                )
                
            # Always update index.html
            index_path = os.path.join(self.upload_dir, 'index.html')
            with open(index_path, 'r', encoding='utf-8') as f:
                index_content = f.read()
            try:
                index_file = self.repo.get_contents("index.html", ref="gh-pages")
                self.repo.update_file(
                    path="index.html",
                    message="Update index.html",
                    content=index_content,
                    sha=index_file.sha,
                    branch="gh-pages"
                )
            except github.GithubException:
                self.repo.create_file(
                    path="index.html",
                    message="Create index.html",
                    content=index_content,
                    branch="gh-pages"
                )
        except Exception as e:
            print(f"Error pushing to gh-pages: {e}")
            raise

def create_flask_app(notebook_processor, github_deployer):
    """
    Create Flask web application for notebook upload and processing
    
    :param notebook_processor: NotebookProcessor instance
    :param github_deployer: GitHubDeployer instance
    :return: Flask app
    """
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = notebook_processor.upload_dir

    @app.route('/', methods=['GET', 'POST'])
    def upload_notebook():
        if request.method == 'POST':
            if 'notebook' not in request.files:
                return 'No file uploaded', 400
            
            file = request.files['notebook']
            if file.filename == '':
                return 'No selected file', 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if notebook_processor.validate_notebook(filepath):
                content = notebook_processor.extract_notebook_content(filepath)
                explanations = notebook_processor.generate_explanation(content)
                
                # Generate web page (simplified for this example)
                html_content = f"""
                <html>
                    <body>
                        <h1>ML Notebook Visualization</h1>
                        <h2>Project Overview</h2>
                        <p>{explanations['overview']}</p>
                        <h2>Technical Details</h2>
                        <p>{explanations['technical_details']}</p>
                    </body>
                </html>
                """
                
                github_deployer.deploy_content(html_content, filename)
                return 'Notebook processed and deployed successfully!', 200
            else:
                return 'Invalid notebook', 400

    return app

def main():
    # Configuration (would typically come from environment variables)
    HF_API_TOKEN = os.getenv('HF_API_TOKEN')
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
    GITHUB_REPO = 'praykar/autonotebook'
    
    notebook_processor = NotebookProcessor(HF_API_TOKEN)
    github_deployer = GitHubDeployer(GITHUB_TOKEN, GITHUB_REPO)
    # Enable GitHub Pages in the repository settings if not already enabled
    
    app = create_flask_app(notebook_processor, github_deployer)
    app.run(debug=True)

if __name__ == '__main__':
    main()