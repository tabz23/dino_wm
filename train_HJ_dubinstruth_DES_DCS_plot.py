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
from deslib.des.knop import KNOP
# Static ensemble methods for comparison
from deslib.static.single_best import SingleBest
from deslib.static.oracle import Oracle
from deslib.static.stacked import StackedClassifier

#DCS libraries
from deslib.dcs.ola     import OLA
from deslib.dcs.lca     import LCA
from deslib.dcs.mcb     import MCB
from deslib.dcs.rank    import Rank
from deslib.dcs.a_posteriori import APosteriori

# For loading your HJ policies
from PyHJ.data import Collector, VectorReplayBuffer, Batch
from PyHJ.trainer import offpolicy_trainer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.utils import TensorboardLogger, WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.policy import avoid_DDPGPolicy_annealing
from torch.utils.tensorboard import SummaryWriter
import yaml
import os

# Set matplotlib config directory
os.environ['MPLCONFIGDIR'] = '/storage1/fs1/sibai/Active/ihab/tmp'
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
        if self.policy is not None:
            return  # Already loaded
            
        # --- 1) Build dummy envs to extract spaces ---
        train_envs = DummyVectorEnv([lambda: DubinsEnvWrapper() for _ in range(1)])
        state_space  = train_envs.observation_space[0]
        action_space = train_envs.action_space[0]
        state_shape  = state_space.shape
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
        critic_hidden     = cfg.get("critic_net", [128,128])
        critic_activation = cfg.get("critic_activation", "ReLU")
        critic_lr         = float(cfg.get("critic_lr", 1e-3))
        actor_hidden      = cfg.get("control_net", [128,128])
        actor_activation  = cfg.get("actor_activation", "ReLU")
        actor_lr          = float(cfg.get("actor_lr", 1e-4))
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
        """Fit method required by sklearn interface."""
        self._load_policy()
        return self
        
    def predict(self, X):
        """Predict safety labels based on HJ values."""
        self._load_policy()
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        predictions = []
        for state in X:
            hj_value = self._compute_hj_value(state)
            pred = 1 if hj_value >= 0 else 0
            predictions.append(pred)
            
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        self._load_policy()
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        probabilities = []
        for state in X:
            hj_value = self._compute_hj_value(state)
            prob_safe = 1 / (1 + np.exp(-hj_value))
            prob_unsafe = 1 - prob_safe
            probabilities.append([prob_unsafe, prob_safe])
            
        return np.array(probabilities)
    
    def _compute_hj_value(self, state):
        """Compute HJ value for a single state [x, y, theta]"""
        from PyHJ.data import Batch
        
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        batch = Batch(obs=state_tensor, info=Batch())
        
        with torch.no_grad():
            action = self.policy(batch, model="actor_old").act
            q_value = self.policy.critic(batch.obs, action)
            
        return float(q_value.cpu().item())


def _get_avoidable_dubins(state, hazards, hazard_size, dt=0.05, v_const=1.0):
    """Check if a Dubins car state is avoidable (safe)."""
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
        
        if dist > 3.0:
            continue
            
        velocity_vec = np.array([v_const * np.cos(theta), v_const * np.sin(theta)])
        dot_product = np.dot(velocity_vec, hazard_vec)
        if dot_product <= 0:
            continue
            
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
    """Generate dataset of states and their ground truth safety labels."""
    np.random.seed(seed)
    
    x_min, x_max = -3.0, 3.0
    y_min, y_max = -3.0, 3.0
    theta_min, theta_max = -np.pi, np.pi
    
    X = np.random.uniform(
        low=[x_min, y_min, theta_min],
        high=[x_max, y_max, theta_max],
        size=(n_samples, 3)
    )
    
    hazards = [np.array([0.4, -1.2]), np.array([-0.4, 1.2])]
    hazard_size = 0.8
    
    y = np.array([
        1 if _get_avoidable_dubins(state, hazards, hazard_size) else 0
        for state in X
    ])
    
    print(f"Generated {n_samples} samples:")
    print(f"Safe states: {np.sum(y == 1)} ({np.mean(y == 1):.2%})")
    print(f"Unsafe states: {np.sum(y == 0)} ({np.mean(y == 0):.2%})")
    
    return X, y


def load_hj_ensemble(checkpoint_epochs: List[int], base_path: str, device: str = 'cpu') -> List[HJValueFunctionClassifier]:
    """Load multiple HJ value function classifiers from checkpoints."""
    ensemble = []
    
    for epoch in checkpoint_epochs:
        policy_path = f"{base_path}/epoch_{epoch:03d}/policy.pth"
        classifier = HJValueFunctionClassifier(policy_path, device)
        ensemble.append(classifier)
        print(f"Loaded HJ classifier from epoch {epoch}")
    
    return ensemble


def compute_ground_truth_brs(hazards, hazard_size, x_min=-3.0, x_max=3.0, y_min=-3.0, y_max=3.0, nx=100, ny=100):
    """Compute the ground truth backward reachable set for all hazards."""
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    # thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
    thetas=[0]
    
    brs_data = {}
    
    for theta in thetas:
        print(f"Computing BRS for theta = {theta:.2f}")
        avoidable = np.zeros((nx, ny), dtype=bool)
        
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                state = [x, y, theta]
                avoidable[ix, iy] = _get_avoidable_dubins(state, hazards, hazard_size)
        
        brs_data[theta] = avoidable
    
    return brs_data


def compute_hj_value_grid(classifier, theta, x_min=-3.0, x_max=3.0, y_min=-3.0, y_max=3.0, nx=100, ny=100):
    """Compute HJ values over a grid for a given theta."""
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    
    hj_values = np.zeros((nx, ny))
    
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            state = [x, y, theta]
            hj_values[ix, iy] = classifier._compute_hj_value(state)
    
    return hj_values


def compute_des_ensemble_hj_grid(des_method, ensemble, theta, x_min=-3.0, x_max=3.0, y_min=-3.0, y_max=3.0, nx=100, ny=100):
    """Compute HJ values using DES ensemble selection over a grid."""
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    
    hj_values = np.zeros((nx, ny))
    selected_models = np.zeros((nx, ny), dtype=int)
    
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            state = np.array([[x, y, theta]])  # Shape (1, 3) for single sample
            
            try:
                # Get the competence of each classifier for this state
                competences = des_method.estimate_competence(state)
                
                # Select the best classifier(s)
                if hasattr(des_method, 'select') and callable(des_method.select):
                    selected_indices = des_method.select(competences)
                    # print(f"Selected classifier for state {state}: {selected_index}")
                else:
                    # Fallback: select the classifier with highest competence
                    selected_indices = [np.argmax(competences[0])]
                    # print(f"Selected classifier for state {state} with best competence: {selected_index}")
                
                # Use the first selected classifier to compute HJ value
                selected_clf_idx = selected_indices[0] if selected_indices else 0
                selected_models[ix, iy] = selected_clf_idx
                
                # Compute HJ value using the selected classifier
                hj_values[ix, iy] = ensemble[selected_clf_idx]._compute_hj_value([x, y, theta])
                
            except Exception as e:
                # Fallback: use the first classifier
                print("fell back", str(des_method))
                selected_models[ix, iy] = 0
                hj_values[ix, iy] = ensemble[0]._compute_hj_value([x, y, theta])
    
    return hj_values, selected_models


def compute_dcs_selector_hj_grid(##maybe works but very slow
    selector,   # an already‐.fit() OLA/LCA/MCB
    ensemble,   # list of HJValueFunctionClassifier
    theta,
    x_min, x_max, y_min, y_max,
    nx=100, ny=100
):
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)

    hj_vals     = np.zeros((nx, ny))
    selected_i  = np.zeros((nx, ny), dtype=int)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            # print(i,x,j,y)
            # 1) pack query
            q = np.array([[x, y, theta]], dtype=np.float32)

            # 2) region of competence
            region = selector.get_competence_region(q)  # maybe shape (1, k) or (k,)
            region = np.atleast_2d(region).astype(int)  # ensure (1, k) of ints

            # 3) competence estimates
            comps = selector.estimate_competence(region)  # shape (1, M)

            # 4) attempt to select via the library
            raw_sel = selector.select(comps)
            raw_sel = np.array(raw_sel)

            if raw_sel.ndim == 1 and raw_sel.shape[0] == 1:
                # typical case: array([idx])
                idx = int(raw_sel[0])
            else:
                # fallback: pick highest competence yourself
                idx = int(np.argmax(comps[0]))

            # 5) store selection + HJ value
            selected_i[i, j] = idx
            hj_vals[i, j]    = ensemble[idx]._compute_hj_value([x, y, theta])
    print("hj_vals",hj_vals)
    print("selected_i",selected_i)


    # Define the save directory
    save_dir = "/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/selected_members"
    os.makedirs(save_dir, exist_ok=True)

    # Save hj_vals and selected_i
    hj_vals_path = os.path.join(save_dir, "hj_vals.npy")
    selected_i_path = os.path.join(save_dir, "selected_i.npy")

    np.savetxt(os.path.join(save_dir, "hj_vals.txt"), hj_vals, fmt="%.5f")
    np.savetxt(os.path.join(save_dir, "selected_i.txt"), selected_i, fmt="%d")


    print(f"Saved HJ values to {hj_vals_path}")
    print(f"Saved selected indices to {selected_i_path}")

    return hj_vals, selected_i


def plot_hj_ensemble_comparison(
    ensemble,
    des_methods: Dict[str,Any],
    dcs_methods: Dict[str,Any],
    brs_data,
    checkpoint_epochs,
    x_min=-3., x_max=3., y_min=-3., y_max=3., nx=100, ny=100
):
    """Plot HJ values for individual ensemble members and DES methods."""
    hazards = [np.array([0.4, -1.2]), np.array([-0.4, 1.2])]
    hazard_size = 0.8
    # thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
    thetas = [0.0]
    
    # Load all ensemble members
    print("Loading ensemble members for plotting...")
    for clf in ensemble:
        clf._load_policy()
    
    for theta in thetas:
        print(f"Plotting for theta = {theta:.2f}")
        
        # Compute HJ values for individual ensemble members
        individual_hj_values = []
        for i, clf in enumerate(ensemble):
            print(f"Computing HJ values for ensemble member {i+1}/{len(ensemble)}")
            hj_vals = compute_hj_value_grid(clf, theta, x_min, x_max, y_min, y_max, nx, ny)
            individual_hj_values.append(hj_vals)
        
        # Get working DES methods
        working_methods = [(name, method) for name, method in des_methods.items() 
                          if method is not None and hasattr(method, 'estimate_competence')]
        
        # Create figure with subplots
        n_methods = len(working_methods)
        n_individual = len(ensemble)
        n_cols = 4  # Individual models, top 3 DES methods, ground truth
        n_rows = max(2, (n_individual + n_methods + 2) // n_cols + 1)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot individual ensemble members
        for i, hj_vals in enumerate(individual_hj_values):
            if plot_idx < len(axes):
                im = axes[plot_idx].imshow(
                    hj_vals.T, extent=(x_min, x_max, y_min, y_max), 
                    origin="lower", cmap='RdYlBu_r'
                )
                axes[plot_idx].set_title(f'Individual Model {i+1}\n(Epoch {checkpoint_epochs[i]})')
                axes[plot_idx].set_xlabel('x')
                axes[plot_idx].set_ylabel('y')
                
                # Add hazards and goal
                for hazard_pos in hazards:
                    circle = plt.Circle(hazard_pos, hazard_size, fill=False, color='red', linewidth=2)
                    axes[plot_idx].add_patch(circle)
                goal_circle = plt.Circle([2.2, 2.2], 0.3, fill=False, color='green', linewidth=2)
                axes[plot_idx].add_patch(goal_circle)
                
                plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
                plot_idx += 1
        
        # Plot top DES methods
        for i, (method_name, method) in enumerate(working_methods[:]):  # Top 3 methods
            if plot_idx < len(axes):
                print(f"Computing DES ensemble HJ for {method_name}")
                try:
                    des_hj_vals, selected_models = compute_des_ensemble_hj_grid(
                        method, ensemble, theta, x_min, x_max, y_min, y_max, nx, ny
                    )
                    
                    im = axes[plot_idx].imshow(
                        des_hj_vals.T, extent=(x_min, x_max, y_min, y_max), 
                        origin="lower", cmap='RdYlBu_r'
                    )
                    axes[plot_idx].set_title(f'DES: {method_name}')
                    axes[plot_idx].set_xlabel('x')
                    axes[plot_idx].set_ylabel('y')
                    
                    # Add hazards and goal
                    for hazard_pos in hazards:
                        circle = plt.Circle(hazard_pos, hazard_size, fill=False, color='red', linewidth=2)
                        axes[plot_idx].add_patch(circle)
                    goal_circle = plt.Circle([2.2, 2.2], 0.3, fill=False, color='green', linewidth=2)
                    axes[plot_idx].add_patch(goal_circle)
                    
                    plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
                    plot_idx += 1
                    
                except Exception as e:
                    print(f"Error plotting {method_name}: {e}")
        # ---- now plot DCS methods in the next columns ----
        for name, method in list(dcs_methods.items()):
            if plot_idx < len(axes):
                print(f"Computing DCS HJ for {name}")
                hj_vals, sel_models = compute_dcs_selector_hj_grid(
                    method, ensemble, theta,
                    x_min, x_max, y_min, y_max, nx, ny
                )
                im = axes[plot_idx].imshow(
                    hj_vals.T, extent=(x_min, x_max, y_min, y_max),
                    origin='lower', cmap='RdYlBu_r'
                )
                axes[plot_idx].set_title(f'DCS: {name}')
                axes[plot_idx].set_xlabel('x'); axes[plot_idx].set_ylabel('y')
                for hz in hazards:
                    axes[plot_idx].add_patch(plt.Circle(hz, hazard_size, fill=False, color='red'))
                axes[plot_idx].add_patch(plt.Circle([2.2,2.2],0.3,fill=False,color='green'))
                plt.colorbar(im, ax=axes[plot_idx], fraction=0.046,pad=0.04)
                plot_idx += 1
        # Plot ground truth BRS
        if plot_idx < len(axes) and theta in brs_data:
            axes[plot_idx].imshow(
                brs_data[theta].T, extent=(x_min, x_max, y_min, y_max), 
                origin="lower", cmap='RdYlBu'
            )
            axes[plot_idx].set_title(f'Ground Truth BRS')
            axes[plot_idx].set_xlabel('x')
            axes[plot_idx].set_ylabel('y')
            
            # Add hazards and goal
            for hazard_pos in hazards:
                circle = plt.Circle(hazard_pos, hazard_size, fill=False, color='red', linewidth=2)
                axes[plot_idx].add_patch(circle)
            goal_circle = plt.Circle([2.2, 2.2], 0.3, fill=False, color='green', linewidth=2)
            axes[plot_idx].add_patch(goal_circle)
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle(f'HJ Value Functions Comparison (θ = {theta:.2f})', fontsize=16, y=0.98)
        plt.show()
        
        # Log to wandb
        wandb.log({f"hj_comparison/theta_{theta:.2f}": wandb.Image(fig)})
        plt.close(fig)


def evaluate_des_methods(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    ensemble: List[HJValueFunctionClassifier]
) -> Dict[str, Any]:
    """Evaluate different DES methods using the HJ ensemble."""
    
    # Fit all classifiers in the ensemble
    print("Fitting HJ ensemble classifiers...")
    for i, clf in enumerate(ensemble):
        print(f"Fitting classifier {i+1}/{len(ensemble)}")
        clf.fit(X_train, y_train)
    
    results = {}
    
    # Different DES methods to try
    des_methods = {
        # 'DES-P': DESP(pool_classifiers=ensemble, k=7),
        # 'DES-KNN': DESKNN(pool_classifiers=ensemble, k=7),
        # 'DES-Clustering': DESClustering(pool_classifiers=ensemble),
        # 'DES-MI': DESMI(pool_classifiers=ensemble, k=7),
        # 'META-DES': METADES(pool_classifiers=ensemble, k=7),
        # 'KNOP': KNOP(pool_classifiers=ensemble, k=7),
        # 'RRC': RRC(pool_classifiers=ensemble),
        # 'DES-KL': DESKL(pool_classifiers=ensemble, k=7),
        # 'Logarithmic': Logarithmic(pool_classifiers=ensemble, k=7),
        # 'Exponential': Exponential(pool_classifiers=ensemble, k=7),
        # 'Single Best': SingleBest(pool_classifiers=ensemble),
        # 'Oracle': Oracle(pool_classifiers=ensemble),
        # 'Stacked': StackedClassifier(pool_classifiers=ensemble),
    }
    
    fitted_methods = {}
    
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
            
            fitted_methods[method_name] = method
            
            print(f"{method_name} accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            results[method_name] = {
                'accuracy': 0.0,
                'predictions': None,
                'method': None,
                'error': str(e)
            }
    
    return results, fitted_methods
from tqdm import tqdm
def evaluate_dcs_methods(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    ensemble: List[HJValueFunctionClassifier]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Exactly the same as DES, but for DCS methods (OLA, LCA, MCB)."""
    dcs_methods = {
        'OLA': OLA(pool_classifiers=ensemble, k=20),
        # 'LCA': LCA(pool_classifiers=ensemble, k=7),
        # 'MCB': MCB(pool_classifiers=ensemble, k=7),
        # 'RANK': Rank(pool_classifiers=ensemble,k=7)
    }
    results, fitted = {}, {}
    for name, method in tqdm(dcs_methods.items(), desc="Evaluating DCS methods"):
        print(f"\nEvaluating DCS {name}...")
        try:
            method.fit(X_train, y_train)
            y_pred = method.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = {'accuracy': acc, 'predictions': y_pred, 'method': method}
            fitted[name] = method
            print(f"{name} accuracy: {acc:.4f}")
        except Exception as e:
            print(f"Error with DCS {name}: {e}")
            results[name] = {'accuracy': 0., 'predictions': None, 'method': None, 'error': str(e)}
    return results, fitted

def plot_results(results: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray):
    """Plot comparison of different DES methods."""
    methods = [m for m,r in results.items() if r['predictions'] is not None]
    if not methods:
        print("No methods to plot—skipping accuracy bar & confusion matrix.")
        return
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
    """Main pipeline for training and testing DES with HJ ensemble."""
    # Configuration
    DEVICE = 'cuda'  # Change to 'cuda' if you have GPU
    N_SAMPLES = 10000
    TEST_SIZE = 0.1
    SEED = 42
    
    # Grid parameters for HJ plotting
    X_MIN, X_MAX = -3.0, 3.0
    Y_MIN, Y_MAX = -3.0, 3.0
    NX, NY = 30,30
    
    # Checkpoint configuration
    BASE_CHECKPOINT_PATH = "/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/runs/ddpg_hj_dubins/20250706-164456"
    CHECKPOINT_EPOCHS = [1, 2, 59, 3, 32]  # Use checkpoints 55-59
    
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
    print("2. Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

    # 3. Load ensemble
    print("3. Loading HJ ensemble...")
    ensemble = load_hj_ensemble(CHECKPOINT_EPOCHS, BASE_CHECKPOINT_PATH, DEVICE)

    # 4. Evaluate individual
    print("4. Evaluating individual HJ classifiers...")
    indiv_accs = []
    for i, clf in enumerate(ensemble):
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        a = accuracy_score(y_test, preds)
        indiv_accs.append(a)
        print(f"   Classifier {i+1} (epoch {CHECKPOINT_EPOCHS[i]}): {a:.4f}")
    wandb.log({f"indiv_acc/epoch_{CHECKPOINT_EPOCHS[i]}": acc for i, acc in enumerate(indiv_accs)})

    # 5. Evaluate DES methods
    print("5. Evaluating DES methods...")
    results_des, fitted_des = evaluate_des_methods(X_train, y_train, X_test, y_test, ensemble)

    # 5b. Evaluate DCS methods
    print("5b. Evaluating DCS methods...")
    results_dcs, fitted_dcs = evaluate_dcs_methods(X_train, y_train, X_test, y_test, ensemble)

    # 6. Log DES results
    wandb.log({f"des_accuracy/{k}": v['accuracy'] for k, v in results_des.items() if v['predictions'] is not None})
    # 6b. Log DCS results
    wandb.log({f"dcs_accuracy/{k}": v['accuracy'] 
               for k,v in results_dcs.items() if v['predictions'] is not None})
    print("now plotting basic results")
    # 7. Plot basic results
    plot_results(results_des, X_test, y_test)
    plot_results(results_dcs, X_test, y_test)

    # 8. Compute ground truth BRS
    hazards = [np.array([0.4, -1.2]), np.array([-0.4, 1.2])]
    hazard_size = 0.8
    brs_data = compute_ground_truth_brs(hazards, hazard_size, X_MIN, X_MAX, Y_MIN, Y_MAX, NX, NY)

    print("now plotting HJ comparisons")
    # 9. Plot HJ comparisons
    plot_hj_ensemble_comparison(
    ensemble=ensemble,
    des_methods=fitted_des,
    dcs_methods=fitted_dcs,
    brs_data=brs_data,
    checkpoint_epochs=CHECKPOINT_EPOCHS,
    x_min=X_MIN,
    x_max=X_MAX,
    y_min=Y_MIN,
    y_max=Y_MAX,
    nx=NX,
    ny=NY
    )


    # 10. Save and finish
    # result_file = f"/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/runs/ddpg_hj_dubins-DES/{run_name}.pkl"
    # with open(result_file, 'wb') as f:
    #     pickle.dump(results, f)
    # wandb.save(result_file)
    # print(f"Results saved to {result_file}")
    # wandb.finish()

if __name__ == "__main__":
    main()
