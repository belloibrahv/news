from flask import Blueprint, render_template, request, jsonify, abort
import sys
import os

# Ensure src is in import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config
import predictor

# Create Blueprint for main routes
main_bp = Blueprint("main", __name__)

@main_bp.route("/", methods=["GET"])
def index():
    """Renders the article submission page."""
    return render_template("index.html", error=None)

@main_bp.route("/classify", methods=["POST"])
def classify():
    """
    Accepts raw article text, cleans it, transforms it, running the SVM classifier.
    Returns predicted label, confidence, and preview of the text.
    """
    article_text = request.form.get("article_text", "")
    
    # FR-W09: Validate empty input
    if not article_text or not article_text.strip():
        # Empty input: return HTTP 400 and render index with error
        return render_template(
            "index.html", 
            error="Please paste an article before classifying."
        ), 400
        
    try:
        # Run classification pipeline
        result = predictor.predict_category(article_text)
        
        category = result["category"]
        confidence = result["confidence"]
        
        # Get matching badge color from configuration
        badge_color = config.CATEGORY_COLORS.get(category, "#1a56db")
        
        # Extract first 200 characters preview (FR-W07)
        article_preview = article_text.strip()[:200]
        if len(article_text.strip()) > 200:
            article_preview += "..."
            
        return render_template(
            "result.html",
            category=category,
            confidence=confidence,
            article_preview=article_preview,
            badge_color=badge_color
        )
        
    except ValueError as val_err:
        # Gracefully handle other validation issues (e.g. no valid words after preprocessing)
        return render_template("index.html", error=str(val_err)), 400
        
    except Exception as e:
        # FR-W10 / NFR-S02: Display neat internal server error without exposing internals
        app.logger.error(f"Error classifying article: {str(e)}")
        return render_template("error.html", message="Classification service is unavailable. Please contact the administrator."), 500

@main_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint to verify service readiness."""
    try:
        if predictor._classifier is not None and predictor._vectorizer is not None:
            return jsonify({
                "status": "ok",
                "model": "SVM",
                "vectoriser": "TF-IDF (50000 features, unigram+bigram)"
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Models are not fully loaded."
            }), 503
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Exception raised: {str(e)}"
        }), 503
