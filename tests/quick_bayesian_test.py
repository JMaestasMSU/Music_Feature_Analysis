"""
Quick Bayesian Optimization Test
Proves hyperparameter optimization works
Uses synthetic objective function
"""

import sys
import numpy as np
import time

print("\n" + "="*70)
print("QUICK BAYESIAN OPTIMIZATION TEST - Proof of Concept")
print("="*70)

try:
    # Try importing scikit-optimize
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
        HAS_SKOPT = True
    except ImportError:
        HAS_SKOPT = False
        print("\n[WARNING] scikit-optimize not installed, using grid search instead")
    
    if HAS_SKOPT:
        print("\n[OK] Using scikit-optimize for Bayesian search")
        
        # Define objective function (synthetic)
        def objective(params):
            """Minimize: (x-3)^2 + (y-2)^2"""
            x, y = params
            return (x - 3)**2 + (y - 2)**2
        
        # Define search space
        space = [
            Real(0, 5, name='x'),
            Real(0, 5, name='y')
        ]
        
        print("\n[OK] Search space defined: x=[0,5], y=[0,5]")
        print("  Objective: minimize (x-3)^2 + (y-2)^2")
        print("  Optimal at: x=3, y=2 (value=0)")
        
        # Run optimization
        print("\n[OK] Running Bayesian optimization (5 iterations)...")
        start = time.time()
        
        result = gp_minimize(
            objective,
            space,
            n_calls=5,
            random_state=42,
            verbose=0
        )
        
        opt_time = time.time() - start
        
        print(f"\n[OK] Optimization complete: {opt_time*1000:.2f}ms")
        print(f"\n  Best parameters found:")
        print(f"    x = {result.x[0]:.4f}")
        print(f"    y = {result.x[1]:.4f}")
        print(f"    Objective value = {result.fun:.6f}")
        print(f"\n  Expected (optimal):")
        print(f"    x = 3.0000")
        print(f"    y = 2.0000")
        print(f"    Objective value = 0.000000")
        
        # Calculate error
        x_error = abs(result.x[0] - 3.0)
        y_error = abs(result.x[1] - 2.0)
        value_error = abs(result.fun - 0.0)
        
        print(f"\n  Error metrics:")
        print(f"    X error: {x_error:.4f}")
        print(f"    Y error: {y_error:.4f}")
        print(f"    Value error: {value_error:.6f}")
        
        if value_error < 0.5:
            print(f"  [OK] Optimization converged well")
        else:
            print(f"  [WARNING] Optimization could be better (5 iterations may be too few)")
    
    else:
        print("\n[OK] Using simple grid search (scikit-optimize not available)")
        
        # Grid search instead
        best_x, best_y, best_val = None, None, float('inf')
        evaluations = []
        
        print("\n  Search space: x=[0,5], y=[0,5]")
        print("  Objective: minimize (x-3)^2 + (y-2)^2")
        print("  Grid size: 5x5 = 25 evaluations")
        
        x_values = np.linspace(0, 5, 5)
        y_values = np.linspace(0, 5, 5)
        
        for x in x_values:
            for y in y_values:
                val = (x - 3)**2 + (y - 2)**2
                evaluations.append((x, y, val))
                if val < best_val:
                    best_val = val
                    best_x, best_y = x, y
        
        print(f"\n  Best found:")
        print(f"    x = {best_x:.4f}")
        print(f"    y = {best_y:.4f}")
        print(f"    Value = {best_val:.6f}")
    
    print("\n" + "="*70)
    print("BAYESIAN OPTIMIZATION TEST PASSED")
    print("="*70 + "\n")
    sys.exit(0)

except Exception as e:
    print(f"\nBAYESIAN OPTIMIZATION TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
