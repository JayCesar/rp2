import torch
import numpy as np
from pathlib import Path
import sys

# Add the ai-analysis directory to the path
sys.path.append(str(Path(__file__).parent))

from bertimbau_c1_inference import BERTimbauC1Predictor

def analyze_model_behavior():
    """Analyze why the model is predicting the same value for all inputs."""
    
    print("🔍 Analyzing model behavior...")
    
    # Load the model
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "bertimbau_c1_finetuned"
    
    try:
        predictor = BERTimbauC1Predictor(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f"📋 Model details:")
    print(f"   - Device: {predictor.device}")
    print(f"   - Max length: {predictor.max_length}")
    
    # Test with different inputs to see if we get different outputs
    test_texts = [
        "Este é um texto muito simples.",
        "Este é um texto completamente diferente com mais palavras e ideias complexas.",
        "A educação é fundamental para o desenvolvimento de qualquer sociedade moderna.",
        "O meio ambiente precisa de proteção urgente para as futuras gerações.",
        "", # Empty text
        "x", # Single character
        "a" * 500  # Long repetitive text
    ]
    
    print("\n🧪 Testing with different inputs:")
    predictions = []
    
    for i, text in enumerate(test_texts, 1):
        pred = predictor.predict_single_essay(text)
        predictions.append(pred)
        text_preview = text[:50] + "..." if len(text) > 50 else text
        print(f"   {i}. '{text_preview}' -> {pred:.6f}")
    
    # Check if all predictions are the same
    unique_predictions = set(predictions)
    print(f"\n📊 Analysis results:")
    print(f"   - Total predictions: {len(predictions)}")
    print(f"   - Unique predictions: {len(unique_predictions)}")
    print(f"   - All same? {'Yes' if len(unique_predictions) == 1 else 'No'}")
    
    if len(unique_predictions) == 1:
        print(f"   - Constant prediction: {list(unique_predictions)[0]:.6f}")
        print("\n❌ PROBLEM: Model is predicting the same value for all inputs!")
        
        # Analyze the model weights
        print("\n🔍 Analyzing model weights:")
        
        # Check if the regression head has learned anything meaningful
        regressor_weight = predictor.model.regressor.weight.data.cpu().numpy()
        regressor_bias = predictor.model.regressor.bias.data.cpu().numpy()
        
        print(f"   - Regressor weight shape: {regressor_weight.shape}")
        print(f"   - Regressor weight stats:")
        print(f"     * Mean: {np.mean(regressor_weight):.6f}")
        print(f"     * Std: {np.std(regressor_weight):.6f}")
        print(f"     * Min: {np.min(regressor_weight):.6f}")
        print(f"     * Max: {np.max(regressor_weight):.6f}")
        print(f"   - Regressor bias: {regressor_bias[0]:.6f}")
        
        # Check if weights are near initialization values
        weight_magnitude = np.linalg.norm(regressor_weight)
        print(f"   - Weight magnitude: {weight_magnitude:.6f}")
        
        if weight_magnitude < 0.01:
            print("   ⚠️  Weights seem very small - model might not have learned")
        
        # Test the model's internal representations
        print("\n🧠 Testing internal representations:")
        
        # Get representations for different texts
        predictor.model.eval()
        with torch.no_grad():
            representations = []
            for text in test_texts[:3]:  # Just first 3
                encoding = predictor.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=predictor.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(predictor.device)
                attention_mask = encoding['attention_mask'].to(predictor.device)
                
                # Get BERT outputs
                outputs = predictor.model.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                representations.append(pooled_output.cpu().numpy())
            
            # Compare representations
            for i, repr in enumerate(representations):
                print(f"   Text {i+1} representation stats:")
                print(f"     * Mean: {np.mean(repr):.6f}")
                print(f"     * Std: {np.std(repr):.6f}")
            
            # Check if representations are different
            if len(representations) > 1:
                diff = np.linalg.norm(representations[0] - representations[1])
                print(f"   Difference between repr 1 and 2: {diff:.6f}")
                
                if diff < 0.001:
                    print("   ⚠️  Representations are very similar - BERT might be frozen")
                else:
                    print("   ✅ Representations are different - BERT is working")
    
    else:
        print("   ✅ Model produces different predictions for different inputs")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    analyze_model_behavior()