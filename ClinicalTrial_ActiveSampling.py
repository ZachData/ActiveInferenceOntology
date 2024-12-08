import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Literal, List
from active_sampling import ActiveSampling
from scipy.special import expit  # For logistic function

class ClinicalTrialSampling(ActiveSampling):
    """
    Implementation of active sampling for clinical trials.
    Handles adaptive trial design with treatment and control groups,
    demographic factors, and survival analysis.
    """

    def _compute_info_gain_rct(self, Ep: np.ndarray, Cp: np.ndarray, X: np.ndarray, 
                              h: float, t: int, r: int) -> float:
        """
        Compute expected information gain for clinical trial decisions.
        
        Args:
            Ep: Current parameter estimates
            Cp: Current parameter covariance
            X: Design matrix
            h: Temperature parameter
            t: Time point index
            r: Randomization ratio index
            
        Returns:
            Expected information gain for the given time and randomization
        """
        R = np.array([[1/3, 1/2, 2/3], 
                     [2/3, 1/2, 1/3]])  # Randomization ratios [treatment; control]
        D = np.array([0.5, 0.5])        # Demographics prior (equal male/female)
        
        # Initialize accumulators
        E = 0       # Total expectation
        Er = [0, 0] # Conditional expectations
        H = [0, 0]  # Conditional entropies
        
        # Compute expectations and entropies for each treatment group
        for treat in range(2):  # treatment vs control
            for demo in range(2):  # demographics
                V = X[0:t, :].copy()
                V[:, -1] = treat - 0.5
                V[:, -2] = demo - 0.5
                
                # Zero-order term
                logits = V @ Ep
                l = np.prod(expit(h * logits))
                
                # First-order term
                if t > 1:
                    dldq = l * np.sum(h * V / (1 + np.exp(h * V @ Ep)), axis=0)
                else:
                    dldq = l * (h * V / (1 + np.exp(h * V @ Ep)))
                
                # Second-order term
                VV = np.zeros((len(Ep), len(Ep)))
                for k in range(V.shape[0]):
                    exp_term = np.exp(V[k, :] @ Ep)
                    VV += h**2 * np.outer(V[k, :], V[k, :]) * exp_term / (1 + exp_term)**2
                
                dldqq = np.outer(dldq, dldq) / l - l * VV
                
                # Accumulate expectations
                Er[treat] += D[demo] * (l + np.trace(Cp @ dldqq) / 2)
                
                # Compute entropy terms
                pr = l + np.trace(Cp @ dldqq) / 2
                H[treat] += D[demo] * (
                    -pr * np.log(pr) - (1 - pr) * np.log(1 - pr)
                    - np.trace(Cp @ (dldqq * np.log(l/(1-l)) 
                                   + (1/l + 1/(1-l)) * np.outer(dldq, dldq))) / 2
                )
        
        # Ensure probabilities are valid
        Er = np.clip(Er, 0.001, 0.999)
        E = R[:, r].T @ Er
        E = np.clip(E, 0.001, 0.999)
        
        # Compute final information gain
        I = -(E * np.log(E) + (1-E) * np.log(1-E))  # Entropy
        I -= R[:, r].T @ H                           # Conditional entropy
        
        # Add preference term (favoring long-term survival)
        I += (E * self.cost * (2 * np.exp((t - X.shape[0])/8) - 1) 
             - (1-E) * self.cost)
        
        return I
    
    def clinical_trial(self, sampling_type: Literal['rand', 'int'] = 'int'):
        """
        Run a clinical trial with active sampling.
        
        Args:
            sampling_type: Type of sampling ('rand' for random, 'int' for intelligent)
        """
        # Setup time points and basis functions
        t = np.arange(1, self.max_time + 1)
        X = self._generate_basis(t, self.n_basis)
        
        # Add demographic and treatment columns
        X = np.hstack([X, np.ones((len(t), 1)) / 2, np.ones((len(t), 1)) / 2])
        
        # Prior setup (favoring survival)
        pE = np.zeros(X.shape[1])
        pE[:self.n_basis] = 1  # Prior expectation of survival
        pC = np.eye(X.shape[1]) / 4
        h = 1  # Temperature parameter
        
        # Generate true parameters
        z = np.linalg.cholesky(pC) @ np.random.randn(X.shape[1]) + pE
        
        # Setup visualization
        plt.figure(figsize=(15, 12))
        
        # Initialize tracking variables
        Ep = pE.copy()
        Cp = pC.copy()
        treatment_counts = np.zeros(2)  # Track number in each group
        
        # Define randomization ratios
        ratios = [1/3, 1/2, 2/3]  # Proportion allocated to treatment
        
        # Plot initial survival curves
        self._plot_survival_curves(X, z, h)
        
        # Main trial loop
        for cohort in range(self.n_cohorts):
            # Compute information gain for each time point and randomization ratio
            if sampling_type == 'int':
                I = np.zeros((len(t), len(ratios)))
                for tt in range(len(t)):
                    for rr in range(len(ratios)):
                        I[tt, rr] = self._compute_info_gain_rct(Ep, Cp, X, h, tt+1, rr)
            else:
                I = np.ones((len(t), len(ratios)))
            
            # Select follow-up time and randomization ratio
            p = np.exp(4 * I) / np.sum(np.exp(4 * I))
            choice = np.random.multinomial(1, p.flatten())
            tt, rr = np.unravel_index(np.argmax(choice), p.shape)
            
            # Generate cohort data
            Y = np.zeros((self.n_participants, 3))
            for i in range(self.n_participants):
                # Assign demographics and treatment
                Y[i, 1] = np.random.rand() < 0.5  # Demographics
                Y[i, 2] = np.random.rand() < ratios[rr]  # Treatment assignment
                
                # Compute survival probability
                V = X[:tt+1, :].copy()
                V[:, -2] = Y[i, 1] - 0.5
                V[:, -1] = Y[i, 2] - 0.5
                logits = V @ z
                prob = np.prod(expit(h * logits))
                
                # Generate survival outcome
                Y[i, 0] = np.random.rand() < prob
                
            # Update treatment counts
            treatment_counts[0] += np.sum(Y[:, 2] > 0)  # Treatment group
            treatment_counts[1] += np.sum(Y[:, 2] < 0)  # Control group
            
            # Update beliefs
            Ep, Cp = self._variational_inference(Ep, Cp, h, X[:tt+1, :], Y)
            
            # Update visualization
            self._update_trial_plots(X, Ep, Cp, h, tt, Y, cohort, treatment_counts)
            
        plt.tight_layout()
        plt.show()
    
    def _plot_survival_curves(self, X: np.ndarray, z: np.ndarray, h: float):
        """Plot initial survival curves for different groups"""
        # Treatment + Female
        plt.subplot(4, 4, 1)
        V = X.copy()
        logits = V @ z
        surv = expit(h * logits)
        plt.bar(np.arange(len(surv)+1), 
                np.concatenate([[1], np.cumprod(surv)]),
                color=[0.8, 0.8, 0.9])
        plt.title('Treatment + Female')
        plt.box(False)
        
        # Treatment + Male
        plt.subplot(4, 4, 2)
        V[:, -2] = -0.5
        logits = V @ z
        surv = expit(h * logits)
        plt.bar(np.arange(len(surv)+1),
                np.concatenate([[1], np.cumprod(surv)]),
                color=[0.8, 0.8, 0.9])
        plt.title('Treatment + Male')
        plt.box(False)
        
        # Control + Female
        plt.subplot(4, 4, 5)
        V = X.copy()
        V[:, -1] = -0.5
        logits = V @ z
        surv = expit(h * logits)
        plt.bar(np.arange(len(surv)+1),
                np.concatenate([[1], np.cumprod(surv)]),
                color=[0.9, 0.8, 0.8])
        plt.title('Control + Female')
        plt.box(False)
        
        # Control + Male
        plt.subplot(4, 4, 6)
        V[:, -2] = -0.5
        logits = V @ z
        surv = expit(h * logits)
        plt.bar(np.arange(len(surv)+1),
                np.concatenate([[1], np.cumprod(surv)]),
                color=[0.9, 0.8, 0.8])
        plt.title('Control + Male')
        plt.box(False)
    
    def _update_trial_plots(self, X: np.ndarray, Ep: np.ndarray, Cp: np.ndarray,
                          h: float, tt: int, Y: np.ndarray, cohort: int,
                          treatment_counts: np.ndarray):
        """Update visualization during trial"""
        # Predicted survival curves
        plt.subplot(2, 2, 2)
        
        # Plot for treatment group (averaging over demographics)
        V = X.copy()
        V[:, -2] = 0  # Average demographics
        
        # Treatment group
        W = np.sqrt(np.diag(V @ Cp @ V.T))
        surv_treat = expit(h * (V @ Ep))
        cumulative_surv = np.concatenate([[1], np.cumprod(surv_treat)])
        upper = np.cumprod(expit(h * (W * 1.64 + V @ Ep))) - cumulative_surv[1:]
        lower = cumulative_surv[1:] - np.cumprod(expit(h * (-W * 1.64 + V @ Ep)))
        
        t = np.arange(len(cumulative_surv))
        plt.fill_between(t, cumulative_surv - np.concatenate([[0], lower]),
                        cumulative_surv + np.concatenate([[0], upper]),
                        color=[0.8, 0.8, 0.9], alpha=0.5)
        plt.plot(t, cumulative_surv, 'b-', linewidth=2)
        
        # Control group
        V[:, -1] = -0.5
        W = np.sqrt(np.diag(V @ Cp @ V.T))
        surv_ctrl = expit(h * (V @ Ep))
        cumulative_surv = np.concatenate([[1], np.cumprod(surv_ctrl)])
        upper = np.cumprod(expit(h * (W * 1.64 + V @ Ep))) - cumulative_surv[1:]
        lower = cumulative_surv[1:] - np.cumprod(expit(h * (-W * 1.64 + V @ Ep)))
        
        plt.fill_between(t, cumulative_surv - np.concatenate([[0], lower]),
                        cumulative_surv + np.concatenate([[0], upper]),
                        color=[0.9, 0.8, 0.8], alpha=0.5)
        plt.plot(t, cumulative_surv, 'r-', linewidth=2)
        
        # Plot current data points
        if np.sum(Y[:, 2] > 0) > 0:  # Treatment group
            plt.plot(tt, np.mean(Y[Y[:, 2] > 0, 0]),
                    'bo', markersize=10)
        if np.sum(Y[:, 2] < 0) > 0:  # Control group
            plt.plot(tt, np.mean(Y[Y[:, 2] < 0, 0]),
                    'ro', markersize=10)
        
        plt.xlim([0, self.n_cohorts + 1])
        plt.title('Treatment Allocations')
        plt.box(False)
        
        plt.pause(0.1)


def main():
    """Run demonstration of clinical trial with different sampling methods."""
    # Demo with random sampling
    print("Running clinical trial with random sampling...")
    demo = ClinicalTrialSampling(basis_type='Gaussian', n_basis=4, cost=0)
    demo.clinical_trial(sampling_type='rand')
    
    # Demo with intelligent sampling
    print("\nRunning clinical trial with intelligent sampling...")
    demo = ClinicalTrialSampling(basis_type='Gaussian', n_basis=4, cost=0)
    demo.clinical_trial(sampling_type='int')
    
    # Demo with survival preference
    print("\nRunning clinical trial with survival preference...")
    demo = ClinicalTrialSampling(basis_type='Gaussian', n_basis=4, cost=1)
    demo.clinical_trial(sampling_type='int')

if __name__ == "__main__":
    main(), 20])
        plt.ylim([0, 1])
        plt.title('Predicted Survival')
        
        # Parameter posterior plot
        plt.subplot(2, 2, 3)
        plt.bar(range(len(Ep)), Ep, color=[0.8, 0.8, 0.9])
        plt.errorbar(range(len(Ep)), Ep,
                    yerr=1.64*np.sqrt(np.diag(Cp)),
                    fmt='none', color=[0.2, 0.2, 0.6],
                    capsize=0)
        plt.title('Parameter Estimates')
        
        # Sampling choices plot
        plt.subplot(4, 2, 6)
        marker_size = (Y[:, 2].mean() + 0.5) * 20  # Size based on allocation ratio
        if cohort == 0:
            plt.plot(tt, -cohort, 'o',
                    markersize=marker_size,
                    color=[0.1, 0.1, 0.4])
            plt.plot([tt, tt], [-cohort, -(cohort+1)],
                    color=[0.1, 0.1, 0.4])
        else:
            plt.plot(tt, -cohort, 'o',
                    markersize=marker_size,
                    color=[0.1, 0.1, 0.4])
            plt.plot([self.prev_tt, tt], [-cohort, -cohort],
                    color=[0.1, 0.1, 0.4])
            plt.plot([tt, tt], [-cohort, -(cohort+1)],
                    color=[0.1, 0.1, 0.4])
        self.prev_tt = tt
        plt.xlim([0, 20])
        plt.title('Trial Decisions')
        plt.axis('off')
        
        # Treatment allocation plot
        plt.subplot(4, 2, 8)
        plt.bar([cohort], [-treatment_counts[1]],
                color=[0.8, 0.8, 0.9])
        plt.bar([cohort], [treatment_counts[0]],
                color=[0.9, 0.8, 0.8])
        plt.xlim([0def __init__(self,
                 basis_type: Literal['Gaussian', 'Cosine', 'Polynomial'] = 'Gaussian',
                 n_basis: int = 4,
                 cost: float = 0):
        """
        Initialize the Clinical Trial sampling demonstration.
        
        Args:
            basis_type: Type of basis functions
            n_basis: Number of elements in basis set
            cost: Cost/preference parameter for survival outcomes
        """
        super().__init__(basis_type, n_basis, cost)
        self.n_participants = 8  # Participants per cohort
        self.n_cohorts = 16     # Number of cohorts
        self.max_time = 20      # Maximum follow-up time (weeks)
        
    def _variational_inference(self, pE: np.ndarray, pC: np.ndarray, h: float,
                             X: np.ndarray, Y: np.ndarray, max_iter: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform variational Laplace inference for the clinical trial model.
        
        Args:
            pE: Prior mean
            pC: Prior covariance
            h: Temperature parameter
            X: Design matrix
            Y: Observed data [survival, demographics, treatment]
            max_iter: Maximum number of iterations
            
        Returns:
            Tuple of (posterior mean, posterior covariance)
        """
        Ep = pE.copy()
        Cp = pC.copy()
        
        for _ in range(max_iter):
            # Initialize accumulator variables
            LL = 0  # Log likelihood
            dLdq = np.zeros_like(Ep)  # Gradient
            dLdqq = np.zeros((len(Ep), len(Ep)))  # Hessian
            
            # Accumulate over subjects
            for j in range(Y.shape[0]):
                # Extract data
                survival = Y[j, 0]  # Survival status
                demo = Y[j, 1]      # Demographics
                treat = Y[j, 2]     # Treatment status
                
                # Design matrix for this subject
                V = X.copy()
                V[:, -1] = treat
                V[:, -2] = demo
                
                # Compute survival probability
                logits = V @ Ep
                l = np.prod(expit(h * logits))
                p = survival * l + (1 - survival) * (1 - l)
                
                # Accumulate log likelihood
                LL += np.log(p)
                
                # Compute gradients
                if V.shape[0] > 1:
                    dldq = l * np.sum(h * V / (1 + np.exp(h * V @ Ep)), axis=0)
                else:
                    dldq = l * (h * V / (1 + np.exp(h * V @ Ep)))
                
                dpdq = (2 * survival - 1) * dldq
                dLdq += dpdq / p
                
                # Compute Hessian
                VV = np.zeros_like(dLdqq)
                for k in range(V.shape[0]):
                    exp_term = np.exp(V[k, :] @ Ep)
                    VV += h**2 * np.outer(V[k, :], V[k, :]) * exp_term / (1 + exp_term)**2
                
                dldqq = np.outer(dldq, dldq) / l - l * VV
                dpdqq = (2 * survival - 1) * dldqq
                dLdqq += -np.outer(dpdq, dpdq) / p**2 + dpdqq / p
            
            # Add prior terms
            iC = np.linalg.pinv(pC)
            dLdqq = dLdqq - iC
            dLdq = dLdq + iC @ (pE - Ep)
            
            # Update parameters
            Ep = Ep - np.linalg.pinv(dLdqq) @ dLdq
            
        Cp = -np.linalg.pinv(dLdqq)
        return Ep, Cp