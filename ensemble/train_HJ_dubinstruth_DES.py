import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from typing import List, Tuple, Dict, Any
import pickle
import wandb
from datetime import datetime
from train_HJ_dubinstruth import DubinsEnvWrapper
# DESLIB imports - different DES methods you can choose from
from deslib.des.des_p import DESP
from deslib.des.des_knn import DESKNN
from deslib.des.des_clustering import DESClustering
from deslib.des.des_mi import DESMI
from deslib.des.meta_des import METADES
from deslib.des.probabilistic import RRC, DESKL, Logarithmic, Exponential
# from deslib.des.des_rrc import DESRRC
from deslib.des.knop import KNOP
# from deslib.des.mcb import MCB

# Static ensemble methods for comparison
from deslib.static.single_best import SingleBest
from deslib.static.oracle import Oracle
from deslib.static.stacked import StackedClassifier
# from deslib.static.simple_ensemble import SimpleEnsemble

# For loading your HJ policies
# PyHJ components for DDPG training
from PyHJ.data import Collector, VectorReplayBuffer, Batch
from PyHJ.trainer import offpolicy_trainer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.utils import TensorboardLogger, WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.policy import avoid_DDPGPolicy_annealing
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.exploration import GaussianNoise
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import yaml
import torch
from PyHJ.exploration import GaussianNoise
# point Matplotlib to /tmp (or any other writable dir)
os.environ['MPLCONFIGDIR'] = '/storage1/fs1/sibai/Active/ihab/tmp'
# make sure it exists
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
class HJValueFunctionClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for HJ value function that acts as a binary classifier.
    
    The classifier predicts:
    - Class 0 (unsafe): if HJ value < 0 (state is in backward reachable set)
    - Class 1 (safe): if HJ value >= 0 (state is NOT in backward reachable set)
    """
    
    def __init__(self, policy_path: str, device: str = 'cpu'):
        self.policy_path = policy_path
        self.device = device
        self.policy = None
        self.classes_ = np.array([0, 1])  # Required by sklearn
        


    def _load_policy(self):
        # --- 1) Build dummy envs to extract spaces ---
        train_envs = DummyVectorEnv([lambda: DubinsEnvWrapper() for _ in range(1)])
        state_space  = train_envs.observation_space[0]
        action_space = train_envs.action_space[0]
        state_shape  = state_space.shape
        # action_space.shape is () for Discrete — fall back to (n,)
        action_shape = tuple(action_space.shape) if hasattr(action_space, "shape") else (action_space.n,)
        max_action   = torch.tensor(action_space.high, device=self.device, dtype=torch.float32)

        # --- 2) Load & normalize YAML keys (hyphens → underscores) ---
        with open("train_HJ_configs.yaml") as f:
            raw_cfg = yaml.safe_load(f)
        cfg = {}
        for k, v in raw_cfg.items():
            nk = k.replace("-", "_")
            cfg[k]   = v
            cfg[nk]  = v

        # --- 3) Extract + cast hyperparameters ---
        # Critic
        critic_hidden     = cfg.get("critic_net", [128,128])
        critic_activation = cfg.get("critic_activation", "ReLU")
        critic_lr         = float(cfg.get("critic_lr", 1e-3))

        # Actor
        actor_hidden      = cfg.get("control_net", [128,128])
        actor_activation  = cfg.get("actor_activation", "ReLU")
        actor_lr          = float(cfg.get("actor_lr", 1e-4))

        # DDPG / PyHJ specific
        tau               = float(cfg.get("tau", 0.005))
        gamma             = float(cfg.get("gamma_pyhj", 0.99))
        sigma             = float(cfg.get("exploration_noise", 0.1))
        rew_norm          = bool(cfg.get("rew_norm", True))
        n_step            = int(cfg.get("n_step", 1))
        actor_grad_steps  = int(cfg.get("actor_gradient_steps", 1))

        # --- 4) Build critic + optimizer ---
        critic_net = Net(
            state_shape, action_shape,
            hidden_sizes=critic_hidden,
            activation=getattr(torch.nn, critic_activation),
            concat=True, device=self.device
        )
        critic      = Critic(critic_net, device=self.device).to(self.device)
        critic_optim = torch.optim.AdamW(critic.parameters(), lr=critic_lr)

        # --- 5) Build actor + optimizer ---
        actor_net = Net(
            state_shape,
            hidden_sizes=actor_hidden,
            activation=getattr(torch.nn, actor_activation),
            device=self.device
        )
        actor     = Actor(actor_net, action_shape, max_action=max_action, device=self.device).to(self.device)
        actor_optim = torch.optim.AdamW(actor.parameters(), lr=actor_lr)

        # --- 6) Create the annealing DDPG policy ---
        self.policy = avoid_DDPGPolicy_annealing(
            critic=critic,
            critic_optim=critic_optim,
            tau=tau,
            gamma=gamma,
            exploration_noise=GaussianNoise(sigma=sigma),
            reward_normalization=rew_norm,
            estimation_step=n_step,
            action_space=action_space,
            actor=actor,
            actor_optim=actor_optim,
            actor_gradient_steps=actor_grad_steps,
        )

        # --- 7) Load pretrained weights ---
        state_dict = torch.load(self.policy_path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

        
    def fit(self, X, y):
        """
        Fit method required by sklearn interface.
        For HJ classifiers, we just load the pre-trained policy.
        """
        self._load_policy()
        return self
        
    def predict(self, X):
        """
        Predict safety labels based on HJ values.
        
        Args:
            X: array-like of shape (n_samples, 3) containing [x, y, theta] states
            
        Returns:
            predictions: array of shape (n_samples,) with 0 (unsafe) or 1 (safe)
        """
        self._load_policy()
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        predictions = []
        
        for state in X:
            # Compute HJ value
            hj_value = self._compute_hj_value(state)
            # Classify: unsafe if HJ < 0, safe if HJ >= 0
            pred = 1 if hj_value >= 0 else 0
            predictions.append(pred)
            
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities. 
        We use a sigmoid transformation of the HJ value to get probabilities.
        """
        self._load_policy()
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        probabilities = []
        
        for state in X:
            hj_value = self._compute_hj_value(state)
            # Transform HJ value to probability using sigmoid
            # Higher HJ value -> higher probability of being safe (class 1)
            prob_safe = 1 / (1 + np.exp(-hj_value))
            prob_unsafe = 1 - prob_safe
            probabilities.append([prob_unsafe, prob_safe])
            
        return np.array(probabilities)
    
    def _compute_hj_value(self, state):
        """Compute HJ value for a single state [x, y, theta]"""
        from PyHJ.data import Batch
        
        # Prepare state
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        batch = Batch(obs=state_tensor, info=Batch())
        
        # Get action from old actor and Q-value from critic
        with torch.no_grad():
            action = self.policy(batch, model="actor_old").act
            q_value = self.policy.critic(batch.obs, action)
            
        return float(q_value.cpu().item())


def _get_avoidable_dubins(state, hazards, hazard_size, dt=0.05, v_const=1.0):
    """
    Check if a Dubins car state is in the backward reachable set (can avoid all hazards).
    Returns True if avoidable (safe), False if not avoidable (unsafe).
    """
    x, y, theta = state
    
    # Check if already in collision
    for hazard_pos in hazards:
        dist = np.linalg.norm([x, y] - hazard_pos)
        if dist <= hazard_size:
            return False
    
    # For each hazard, check if we can avoid it using the safest policy
    for hazard_pos in hazards:
        hazard_vec = hazard_pos - np.array([x, y])
        dist = np.linalg.norm(hazard_vec)
        
        # If already far enough, skip this hazard
        if dist > 3.0:
            continue
            
        # Current velocity vector
        velocity_vec = np.array([v_const * np.cos(theta), v_const * np.sin(theta)])
        
        # Check if we're heading towards the hazard
        dot_product = np.dot(velocity_vec, hazard_vec)
        if dot_product <= 0:
            continue
            
        # Use safest policy: maximum angular acceleration away from hazard
        cross_product = np.cross(velocity_vec, hazard_vec)
        
        if cross_product >= 0:
            safest_action = np.array([-1.0], dtype=np.float32)
        else:
            safest_action = np.array([1.0], dtype=np.float32)
        
        # Simulate forward with safest policy
        sim_state = np.array([x, y, theta], dtype=np.float32)
        max_sim_time = 10.0
        sim_time = 0.0
        
        collision_detected = False
        while sim_time < max_sim_time:
            dtheta = float(np.clip(safest_action[0], -1.0, 1.0))
            x_sim, y_sim, theta_sim = sim_state
            
            x_new = x_sim + v_const * np.cos(theta_sim) * dt
            y_new = y_sim + v_const * np.sin(theta_sim) * dt
            theta_new = theta_sim + dtheta * dt
            
            sim_state = np.array([x_new, y_new, theta_new], dtype=np.float32)
            
            dist_to_hazard = np.linalg.norm([x_new, y_new] - hazard_pos)
            if dist_to_hazard <= hazard_size:
                collision_detected = True
                break
            
            current_vel = np.array([v_const * np.cos(theta_new), v_const * np.sin(theta_new)])
            current_hazard_vec = hazard_pos - np.array([x_new, y_new])
            current_dot = np.dot(current_vel, current_hazard_vec)
            
            if current_dot <= 0 and dist_to_hazard > hazard_size * 2:
                break
                
            if abs(x_new) > 3.0 or abs(y_new) > 3.0:
                break
                
            sim_time += dt
        
        if collision_detected:
            return False
    
    return True


def generate_dataset(n_samples: int = 10000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset of states and their ground truth safety labels.
    
    Returns:
        X: array of shape (n_samples, 3) with [x, y, theta] states
        y: array of shape (n_samples,) with binary labels (0=unsafe, 1=safe)
    """
    np.random.seed(seed)
    
    # Define state space bounds
    x_min, x_max = -3.0, 3.0
    y_min, y_max = -3.0, 3.0
    theta_min, theta_max = -np.pi, np.pi
    
    # Generate random states
    X = np.random.uniform(
        low=[x_min, y_min, theta_min],
        high=[x_max, y_max, theta_max],
        size=(n_samples, 3)
    )
    
    # Define hazards
    hazards = [np.array([0.4, -1.2]), np.array([-0.4, 1.2])]
    hazard_size = 0.8
    
    # Compute ground truth labels
    y = np.array([
        1 if _get_avoidable_dubins(state, hazards, hazard_size) else 0
        for state in X
    ])
    
    print(f"Generated {n_samples} samples:")
    print(f"Safe states: {np.sum(y == 1)} ({np.mean(y == 1):.2%})")
    print(f"Unsafe states: {np.sum(y == 0)} ({np.mean(y == 0):.2%})")
    
    return X, y


def load_hj_ensemble(checkpoint_epochs: List[int], base_path: str, device: str = 'cpu') -> List[HJValueFunctionClassifier]:
    """
    Load multiple HJ value function classifiers from checkpoints.
    
    Args:
        checkpoint_epochs: List of epoch numbers to load
        base_path: Base path to the checkpoint directory
        device: Device to load models on
        
    Returns:
        List of HJValueFunctionClassifier instances
    """
    ensemble = []
    
    for epoch in checkpoint_epochs:
        policy_path = f"{base_path}/epoch_{epoch:03d}/policy.pth"
        classifier = HJValueFunctionClassifier(policy_path, device)
        ensemble.append(classifier)
        print(f"Loaded HJ classifier from epoch {epoch}")
    
    return ensemble


def evaluate_des_methods(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    ensemble: List[HJValueFunctionClassifier]
) -> Dict[str, Any]:
    """
    Evaluate different DES methods using the HJ ensemble.
    
    Returns:
        Dictionary with results for each method
    """
    
    # Fit all classifiers in the ensemble
    print("Fitting HJ ensemble classifiers...")
    for i, clf in enumerate(ensemble):
        print(f"Fitting classifier {i+1}/{len(ensemble)}")
        clf.fit(X_train, y_train)
    
    # Get predictions from all classifiers for competence estimation
    print("Computing predictions for competence estimation...")
    train_predictions = np.array([clf.predict(X_train) for clf in ensemble]).T
    
    results = {}
    
    # Different DES methods to try
    des_methods = {
        # === Dynamic Ensemble Selection Methods ===
        
        # 1. DES-P: Uses percentage of correct classifications in local region
        'DES-P': DESP(pool_classifiers=ensemble, k=7),
        
        # 2. DES-KNN: Uses k-nearest neighbors for competence estimation
        'DES-KNN': DESKNN(pool_classifiers=ensemble, k=7),
        
        # 3. DES-Clustering: Uses clustering for competence region definition
        'DES-Clustering': DESClustering(pool_classifiers=ensemble),
        
        # 4. DES-MI: Uses mutual information for competence estimation
        'DES-MI': DESMI(pool_classifiers=ensemble, k=7),
        
        # 5. META-DES: Uses meta-features for competence estimation
        'META-DES': METADES(pool_classifiers=ensemble, k=7),
        
        # 6. DES-RRC: Uses randomized reference classifiers
        # 'DES-RRC': DESRRC(pool_classifiers=ensemble, k=7),
        
        # 7. KNOP: K-Nearest Oracle Pooling
        'KNOP': KNOP(pool_classifiers=ensemble, k=7),
        
        # 8. MCB: Multiple Classifier Behavior
        # 'MCB': MCB(pool_classifiers=ensemble, k=7),
        
        # === Probabilistic DES Methods ===
        
        # 9. RRC: Randomized Reference Classifier
        'RRC': RRC(pool_classifiers=ensemble),
        
        # 10. DES-KL: Uses Kullback-Leibler divergence
        'DES-KL': DESKL(pool_classifiers=ensemble, k=7),
        
        # 11. Logarithmic: Uses logarithmic combination
        'Logarithmic': Logarithmic(pool_classifiers=ensemble, k=7),
        
        # 12. Exponential: Uses exponential combination
        'Exponential': Exponential(pool_classifiers=ensemble, k=7),
        
        # === Static Methods for Comparison ===
        
        # 13. Single Best: Uses the best performing classifier
        'Single Best': SingleBest(pool_classifiers=ensemble),
        
        # 14. Oracle: Theoretical upper bound (selects best classifier per sample)
        'Oracle': Oracle(pool_classifiers=ensemble),
        
        # 15. Simple Ensemble: Simple voting
        # 'Simple Ensemble': SimpleEnsemble(pool_classifiers=ensemble),
        
        # 16. Stacked: Uses meta-learning
        'Stacked': StackedClassifier(pool_classifiers=ensemble),
    }
    
    for method_name, method in des_methods.items():
        print(f"\nEvaluating {method_name}...")
        try:
            # Fit the DES method
            method.fit(X_train, y_train)
            
            # Predict on test set
            y_pred = method.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            results[method_name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'method': method
            }
            
            print(f"{method_name} accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            results[method_name] = {
                'accuracy': 0.0,
                'predictions': None,
                'method': None,
                'error': str(e)
            }
    
    return results


def plot_results(results: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray):
    """
    Plot comparison of different DES methods.
    """
    # Extract accuracies
    methods = []
    accuracies = []
    
    for method_name, result in results.items():
        if result['predictions'] is not None:
            methods.append(method_name)
            accuracies.append(result['accuracy'])
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    methods = [methods[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    # Plot accuracy comparison
    plt.figure(figsize=(15, 8))
    plt.barh(methods, accuracies)
    plt.xlabel('Accuracy')
    plt.title('DES Methods Comparison - Test Accuracy')
    plt.grid(axis='x', alpha=0.3)
    
    # Add accuracy values on bars
    for i, (method, acc) in enumerate(zip(methods, accuracies)):
        plt.text(acc + 0.001, i, f'{acc:.4f}', va='center')
    
    plt.tight_layout()
    plt.show()
    
    # Confusion matrix for best method
    best_method = methods[0]
    best_predictions = results[best_method]['predictions']
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Unsafe', 'Safe'], 
                yticklabels=['Unsafe', 'Safe'])
    plt.title(f'Confusion Matrix - {best_method}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Print detailed results for top methods
    print("\n" + "="*50)
    print("DETAILED RESULTS")
    print("="*50)
    
    for i, method in enumerate(methods[:5]):  # Top 5 methods
        result = results[method]
        print(f"\n{i+1}. {method}")
        print(f"   Accuracy: {result['accuracy']:.4f}")
        if result['predictions'] is not None:
            print(f"   Classification Report:")
            print(classification_report(y_test, result['predictions'], 
                                      target_names=['Unsafe', 'Safe'], 
                                      digits=4))


def main():
    """
    Main pipeline for training and testing DES with HJ ensemble.
    """
    # Configuration
    DEVICE = 'cpu'  # Change to 'cuda' if you have GPU
    N_SAMPLES = 10000
    TEST_SIZE = 0.3
    SEED = 42
    
    # Checkpoint configuration
    BASE_CHECKPOINT_PATH = "/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/runs/ddpg_hj_dubins/20250706-164456"
    # CHECKPOINT_EPOCHS = [55, 56, 57, 58, 59]  # Use checkpoints 55-59
    CHECKPOINT_EPOCHS = [59, 2, 3, 4, 5]  # Use checkpoints 55-59

    # Initialize wandb
    run_name = f"des_hj_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project="des-hj-ensemble", name=run_name)
    
    print("="*60)
    print("DYNAMIC ENSEMBLE SELECTION WITH HJ VALUE FUNCTIONS")
    print("="*60)
    
    # Step 1: Generate dataset
    print("\n1. Generating dataset...")
    X, y = generate_dataset(n_samples=N_SAMPLES, seed=SEED)
    
    # Step 2: Split into train/test
    print("\n2. Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 3: Load HJ ensemble
    print("\n3. Loading HJ ensemble...")
    ensemble = load_hj_ensemble(CHECKPOINT_EPOCHS, BASE_CHECKPOINT_PATH, DEVICE)
    
    # Step 4: Evaluate individual classifiers
    print("\n4. Evaluating individual HJ classifiers...")
    individual_accuracies = []
    for i, clf in enumerate(ensemble):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        individual_accuracies.append(acc)
        print(f"HJ Classifier {i+1} (epoch {CHECKPOINT_EPOCHS[i]}): {acc:.4f}")
    
    wandb.log({
        "individual_accuracies": {
            f"epoch_{CHECKPOINT_EPOCHS[i]}": acc 
            for i, acc in enumerate(individual_accuracies)
        }
    })
    
    # Step 5: Evaluate DES methods
    print("\n5. Evaluating DES methods...")
    results = evaluate_des_methods(X_train, y_train, X_test, y_test, ensemble)
    
    # Step 6: Log results to wandb
    print("\n6. Logging results...")
    wandb_results = {}
    for method_name, result in results.items():
        if result['predictions'] is not None:
            wandb_results[f"des_accuracy/{method_name}"] = result['accuracy']
    
    wandb.log(wandb_results)
    
    # Step 7: Plot and analyze results
    print("\n7. Plotting results...")
    plot_results(results, X_test, y_test)
    
    # Step 8: Save results
    print("\n8. Saving results...")
    results_path = f"des_results_{run_name}.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    wandb.save(results_path)
    
    print(f"\nResults saved to: {results_path}")
    print("DES evaluation complete!")
    
    wandb.finish()


if __name__ == "__main__":
    main()