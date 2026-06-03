"""
Quick start script to run the entire GNN pipeline in one go.

Usage:
    python quick_start.py
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """
    Run the complete pipeline: data generation → model training → results visualization
    """
    print("=" * 80)
    print("GNN LATTICE STABILITY PREDICTION - QUICK START")
    print("=" * 80)
    
    # Step 1: Generate data
    print("\n[1/3] Generating synthetic lattice data...")
    print("-" * 80)
    
    from data.generate_lattice_data import generate_dataset
    dataset = generate_dataset(num_samples=100, lattice_sizes=[2, 3, 4], 
                               output_dir='data')
    
    # Step 2: Train model
    print("\n[2/3] Training GNN model...")
    print("-" * 80)
    
    os.chdir('src')
    from train import main as train_main
    train_main()
    os.chdir('..')
    
    # Step 3: Results summary
    print("\n[3/3] Training complete!")
    print("-" * 80)
    print("\nResults Summary:")
    print("  - Model saved to: results/best_model.pt")
    print("  - Visualizations saved to: results/training_results.png")
    print("  - Dataset saved to: data/lattice_dataset.pkl")
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  1. Check results/training_results.png for performance visualization")
    print("  2. Review notes/README.md for detailed documentation")
    print("  3. Try advanced_inference.py to make predictions on new structures")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("  2. Check that you're running from the Mock GNN Workflow directory")
        print("  3. Ensure Python 3.8+ is being used")
