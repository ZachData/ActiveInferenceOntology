import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Literal
from active_sampling import ActiveSampling  # Import base class

class DynamicSampling(ActiveSampling):
    """
    Implementation of dynamic sampling where the function being approximated varies with time.
    Inherits basic functionality from ActiveSampling class.
    """
    
    def __init__(self, 
                 basis_type: Literal['Gaussian', 'Cosine', 'Polynomial'] = 'Gaussian',
                 n_basis: int = 8,
                 cost: float = 0):
        """
        Initialize the Dynamic Sampling demonstration.
        
        Args:
            basis_type: Type of basis functions ('Gaussian', 'Cosine', 'Polynomial')
            n_basis: Number of elements in basis set (per dimension)
            cost: Cost parameter for early termination probability
        """
        super().__init__(basis_type, n_basis, cost)
        
    def _inference_dynamic(self, pE: np.ndarray, pC: np.ndarray, h: float, 
                         X_t: np.ndarray, X_x: np.ndarray, y: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute posteriors for dynamic model parameters.
        
        Args:
            pE: Prior mean
            pC: Prior covariance
            h: Likelihood precision
            X_t: Temporal design matrix
            X_x: Spatial design matrix
            y: Observed data
            
        Returns:
            Tuple of (posterior mean, posterior covariance)
        """
        X = np.kron(X_t, X_x)  # Kronecker product for space-time design matrix
        ih = 1/h
        L = X.T @ (ih * X) + np.linalg.inv(pC)
        Ep = np.linalg.solve(L, np.linalg.solve(pC, pE) + X.T * (ih * y))
        Cp = np.linalg.inv(L)
        return Ep, Cp
        
    def _compute_info_gain_dynamic(self, Ep: np.ndarray, Cp: np.ndarray, 
                                 X_t: np.ndarray, X_x: np.ndarray, h: float) -> np.ndarray:
        """
        Compute information gain for dynamic sampling.
        
        Args:
            Ep: Current parameter estimates
            Cp: Current parameter covariance
            X_t: Temporal design matrix
            X_x: Spatial design matrix
            h: Observation noise
            
        Returns:
            Information gain for each spatial location
        """
        I = np.zeros(X_x.shape[0])
        X = np.kron(X_t, X_x)
        
        for i in range(len(I)):
            Xi = np.kron(X_t, X_x[i:i+1, :])
            variance = Xi @ Cp @ Xi.T + h
            I[i] = 0.5 * np.log(variance) - 0.5 * np.log(h)
            
        return I
        
    def dynamic_sampling(self, sampling_type: Literal['rand', 'int'] = 'int'):
        """
        Main demonstration routine for dynamic sampling inference.
        
        Args:
            sampling_type: Type of sampling ('rand' for random, 'int' for intelligent)
        """
        # Domain setup
        x = np.arange(-100, 101)
        t = np.arange(len(x))
        
        # Prior setup
        pE = np.zeros(self.n_basis ** 2)  # Square for space-time parameters
        pC = np.eye(self.n_basis ** 2) / 8
        h = 1/16
        
        # Generate basis functions
        X_x = self._generate_basis(x, self.n_basis)  # Spatial basis
        X_t = self._generate_basis(t, self.n_basis)  # Temporal basis
        
        # Generate true parameters and sample data
        z = np.linalg.cholesky(pC) @ np.random.randn(self.n_basis ** 2) + pE
        X = np.kron(X_t, X_x)
        Y = np.reshape(X @ z + np.random.randn(len(x) * len(t)) * np.sqrt(h), 
                      (len(x), len(t)))
        
        # Setup figure
        plt.figure(figsize=(15, 10))
        
        # Initial state
        Ep = pE.copy()
        Cp = pC.copy()
        
        # Main sampling loop
        for i in range(len(t) // 8):
            j = 1 + i * 8  # Time index
            
            # Plot sample data
            plt.subplot(3, 2, 1)
            plt.plot(x, Y[:, j], '.', color=[0.6, 0.6, 0.8], markersize=8)
            plt.title('Sample Data')
            plt.xlabel('x')
            plt.ylabel('y')
            
            # Compute information gain and select action
            if sampling_type == 'int':
                I = self._compute_info_gain_dynamic(Ep, Cp, X_t[j:j+1, :], X_x, h)
            else:
                I = np.ones(len(x))
            
            # Add stop sampling option
            I = np.append(I, self.cost)
            
            # Action selection
            p = np.exp(64 * I) / np.sum(np.exp(64 * I))
            a = np.where(np.cumsum(p) > np.random.rand())[0][0]
            
            if a == len(I) - 1:
                print("Sampling Terminated - Sufficient information gained")
                break
                
            # Sample data point
            Xi = np.kron(X_t[j:j+1, :], X_x[a:a+1, :])
            y = Xi @ z + np.random.randn() * np.sqrt(h)
            
            # Update beliefs
            Ep, Cp = self._inference_dynamic(Ep, Cp, h, X_t[j:j+1, :], X_x[a:a+1, :], y)
            
            # Update visualization
            self._update_plots_dynamic(x, t, X_x, X_t, Ep, Cp, h, a, y, i, j, Y, z)
            
        plt.tight_layout()
        plt.show()
        
    def _update_plots_dynamic(self, x, t, X_x, X_t, Ep, Cp, h, a, y, i, j, Y, z):
        """Helper method to update visualization during dynamic sampling"""
        # Prediction plot
        plt.subplot(3, 2, 2)
        Xi = np.kron(X_t[j:j+1, :], X_x)
        mean = Xi @ Ep
        c = np.diag(Xi @ Cp @ Xi.T) + h
        
        plt.fill_between(x, mean - 1.64*np.sqrt(c), mean + 1.64*np.sqrt(c),
                        color=[0.8, 0.8, 0.9], alpha=0.5)
        plt.plot(x, mean, 'b-', linewidth=2)
        plt.plot(x[a], y, 'bo', markersize=10)
        plt.title('Prediction')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Parameter posterior plot
        plt.subplot(3, 2, 3)
        plt.bar(range(len(Ep)), Ep, color=[0.8, 0.8, 0.9])
        plt.errorbar(range(len(Ep)), Ep, yerr=1.64*np.sqrt(np.diag(Cp)),
                    fmt='none', color=[0.2, 0.2, 0.6])
        plt.title('Posterior modes')
        
        # Sampling choices plot
        plt.subplot(3, 2, 4)
        if hasattr(self, 'prev_a'):
            plt.plot([self.prev_a, a], [-i, -i], 'b-')
        plt.plot(a, -i, 'bo', markersize=5)
        plt.plot([a, a], [-i, -(i+1)], 'b-')
        self.prev_a = a
        plt.title('Choices')
        plt.axis('off')
        
        # True time evolution plot
        plt.subplot(3, 2, 5)
        Z = np.reshape(z, (self.n_basis, self.n_basis))
        plt.imshow((X_x @ Z @ X_t.T).T, aspect='auto', cmap='gray')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Time Evolution')
        
        # Inferred time evolution plot
        plt.subplot(3, 2, 6)
        Z_inferred = np.reshape(Ep, (self.n_basis, self.n_basis))
        plt.imshow((X_x @ Z_inferred @ X_t.T).T, aspect='auto', cmap='gray')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Inferred Time Evolution')
        
        plt.pause(0.1)

def main():
    # Demo with dynamic sampling
    demo = DynamicSampling(basis_type='Gaussian', n_basis=8, cost=0)
    
    print("Running random dynamic sampling demo...")
    demo.dynamic_sampling(sampling_type='rand')
    
    print("\nRunning intelligent dynamic sampling demo...")
    demo.dynamic_sampling(sampling_type='int')

if __name__ == "__main__":
    main()