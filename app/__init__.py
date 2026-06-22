import os
import sys
from flask import Flask

# Add src to the Python path so we can import config and predictor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

def create_app():
    # We specify the templates folder path explicitly
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))
    app = Flask(__name__, template_folder=template_dir)
    
    # Startup validation of model files (US-10)
    import config
    from predictor import load_models
    
    print("Validating model and vectorizer files on startup...")
    try:
        load_models()
        print("Model verification succeeded! Classification service is active.")
    except Exception as e:
        app.logger.critical(f"FATAL: Model verification failed on startup: {str(e)}")
        # Exit with a non-zero status code as required by US-10
        print(f"FATAL STARTUP ERROR: {str(e)}")
        sys.exit(1)
        
    # Register routes blueprint
    from .routes import main_bp
    app.register_blueprint(main_bp)
        
    return app
