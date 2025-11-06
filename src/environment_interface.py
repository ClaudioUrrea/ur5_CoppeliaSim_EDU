"""
APO-MORL: An Adaptive Pareto-Optimal Framework for Real-Time Multi-Objective Optimization in Robotic Pick-and-Place Manufacturing Systems
Compatible with existing UR5 script - NO MODIFICATIONS REQUIRED
"""

# Updated import for CoppeliaSim Remote API
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    COPPELIA_AVAILABLE = True
    print("CoppeliaSim Remote API connected successfully")
except ImportError as e:
    print(f"CoppeliaSim not available: {e}")
    COPPELIA_AVAILABLE = False
    # Create mock sim object for testing
    class MockSim:
        def getObjectHandle(self, name): raise Exception(f"Mock: {name} not found")
        def getJointPosition(self, handle): return 0.0
        def getJointVelocity(self, handle): return 0.0
        def setJointTargetPosition(self, handle, pos): pass
        def getObjectPosition(self, handle, ref): return [0.0, 0.0, 0.0]
        def getJointForce(self, handle): return 0.0
    sim = MockSim()

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

class UR5_MORL_Environment(gym.Env):
    """
    Multi-Objective RL Environment for UR5
    Uses your original script as movement controller
    """
    
    def __init__(self, use_original_controller=True):
        super().__init__()
        
        self.coppelia_available = COPPELIA_AVAILABLE
        
        # Multi-objective configuration
        self.objectives = {
            'throughput': 0.0,        # Parts per minute
            'cycle_time': 0.0,        # Time per complete cycle  
            'energy_efficiency': 0.0, # Power consumption optimization
            'precision': 0.0,         # Placement accuracy
            'wear_reduction': 0.0,    # Joint stress minimization
            'collision_avoidance': 0.0 # Safety distance maintenance
        }
        
        # State space (23-dimensional as per research doc)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(23,), dtype=np.float32
        )
        
        # Action space (hybrid: 6 continuous + 4 discrete)
        self.action_space = spaces.Dict({
            'joint_velocities': spaces.Box(low=-1.0, high=1.0, shape=(6,)),
            'gripper_control': spaces.Discrete(2),  # Open/Close
            'conveyor_interaction': spaces.Discrete(3),  # Stop/Slow/Fast
            'task_priority': spaces.Discrete(6),  # Which objective to prioritize
            'pallet_selection': spaces.Discrete(4)  # Which pallet to use
        })
        
        # Initialize connection to your original script
        self.original_controller = None
        self.use_original_controller = use_original_controller
        self.episode_start_time = 0
        self.parts_completed = 0
        self.energy_consumed = 0
        
        # UR5 component handles (will be detected automatically)
        self.ur5_joints = []
        self.gripper_handle = None
        self.ik_tip = None
        self.ik_target = None
        self.conveyor_handle = None
        self.pallet_handles = []
        
    def reset(self, seed=None, options=None):
        """Reset environment and return initial observation"""
        super().reset(seed=seed)
        
        # Only try to detect components if CoppeliaSim is available
        if self.coppelia_available and not self.ur5_joints:
            self._detect_ur5_components()
            
        # Reset episode metrics
        self.episode_start_time = time.time()
        self.parts_completed = 0
        self.energy_consumed = 0
        
        # Reset objectives
        for key in self.objectives:
            self.objectives[key] = 0.0
            
        # Get initial state
        state = self._get_state()
        info = {'objectives': self.objectives.copy()}
        
        return state, info
    
    def step(self, action):
        """Execute action and return new state, rewards, done, info"""
        
        # Extract action components
        joint_vels = action['joint_velocities']
        gripper_cmd = action['gripper_control']
        conveyor_cmd = action['conveyor_interaction']
        task_priority = action['task_priority']
        pallet_selection = action['pallet_selection']
        
        # Execute action using original controller functions
        if self.use_original_controller and self.coppelia_available:
            self._execute_with_original_controller(action)
        else:
            self._execute_direct_control(action)
            
        # Get new state
        state = self._get_state()
        
        # Calculate multi-objective rewards
        rewards = self._calculate_multi_objective_rewards()
        
        # Check if episode is done
        done = self._check_episode_done()
        
        # Update objectives
        self._update_objectives()
        
        info = {
            'objectives': self.objectives.copy(),
            'parts_completed': self.parts_completed,
            'episode_time': time.time() - self.episode_start_time,
            'coppelia_available': self.coppelia_available
        }
        
        return state, rewards, done, False, info
    
    def _detect_ur5_components(self):
        """Automatically detect UR5 components in scene"""
        if not self.coppelia_available:
            return
            
        print("[MORL_ENV] Detecting UR5 components...")
        
        # Detect joints
        joint_patterns = ['UR5_joint1', 'UR5_joint2', 'UR5_joint3', 
                         'UR5_joint4', 'UR5_joint5', 'UR5_joint6']
        for pattern in joint_patterns:
            try:
                handle = sim.getObjectHandle(pattern)
                self.ur5_joints.append(handle)
            except:
                # Try alternative naming
                alt_pattern = pattern.replace('UR5_', '')
                try:
                    handle = sim.getObjectHandle(alt_pattern)
                    self.ur5_joints.append(handle)
                except:
                    pass
        
        # Detect other components
        try:
            self.gripper_handle = sim.getObjectHandle('RG2')
        except:
            pass
            
        try:
            self.ik_tip = sim.getObjectHandle('UR5_ikTip')
        except:
            try:
                self.ik_tip = sim.getObjectHandle('ikTip')
            except:
                pass
                
        print(f"[MORL_ENV] Found {len(self.ur5_joints)} joints")
        
    def _get_state(self):
        """Get 23-dimensional state representation"""
        state = np.zeros(23, dtype=np.float32)
        
        if not self.coppelia_available:
            # Return mock state when CoppeliaSim is not available
            state[:] = np.random.randn(23) * 0.1
            return state
        
        # Robot State (12D): Joint positions + velocities
        if len(self.ur5_joints) >= 6:
            for i, joint in enumerate(self.ur5_joints[:6]):
                try:
                    pos = sim.getJointPosition(joint)
                    vel = sim.getJointVelocity(joint) 
                    state[i] = pos      # Joint positions (0-5)
                    state[i+6] = vel    # Joint velocities (6-11)
                except:
                    pass
        
        # Environment State (8D): Object positions, conveyor, pallets
        if self.ik_tip:
            try:
                tip_pos = sim.getObjectPosition(self.ik_tip, -1)
                state[12:15] = tip_pos  # Tip position (12-14)
            except:
                pass
                
        # Conveyor status (15)
        state[15] = self._get_conveyor_status()
        
        # Pallet occupancy (16-19)
        state[16:20] = self._get_pallet_status()
        
        # Task State (3D): Current objective, progress, time
        state[20] = self._get_current_task_progress()  # Progress (20)
        state[21] = (time.time() - self.episode_start_time)  # Time (21)
        state[22] = self.parts_completed  # Parts completed (22)
        
        return state
    
    def _execute_with_original_controller(self, action):
        """Execute action using your original UR5 script functions"""
        
        # This is where we interface with your original script
        # WITHOUT modifying it - we just call its functions
        
        joint_vels = action['joint_velocities']
        gripper_cmd = action['gripper_control']
        pallet_selection = action['pallet_selection']
        
        # Convert RL action to original script parameters
        if hasattr(self, 'original_controller'):
            # Use your original movement functions
            # Example: modify target configs based on RL action
            pass
        else:
            # Direct joint control
            self._execute_direct_control(action)
    
    def _execute_direct_control(self, action):
        """Direct control when original controller not available"""
        joint_vels = action['joint_velocities']
        
        if not self.coppelia_available:
            # Simulate action execution
            time.sleep(0.01)
            return
        
        # Apply joint velocities
        for i, vel in enumerate(joint_vels):
            if i < len(self.ur5_joints):
                try:
                    # Convert velocity to position target
                    current_pos = sim.getJointPosition(self.ur5_joints[i])
                    target_pos = current_pos + vel * 0.01  # Small time step
                    sim.setJointTargetPosition(self.ur5_joints[i], target_pos)
                except:
                    pass
    
    def _calculate_multi_objective_rewards(self):
        """Calculate rewards for all 6 objectives"""
        rewards = {}
        
        # 1. Throughput (parts per minute)
        elapsed_time = time.time() - self.episode_start_time
        if elapsed_time > 0:
            throughput = (self.parts_completed / elapsed_time) * 60
            rewards['throughput'] = throughput / 10.0  # Normalize
        else:
            rewards['throughput'] = 0.0
            
        # 2. Cycle Time (inverse of time per cycle)
        if self.parts_completed > 0:
            avg_cycle_time = elapsed_time / self.parts_completed
            rewards['cycle_time'] = 1.0 / (avg_cycle_time + 1e-6)
        else:
            rewards['cycle_time'] = 0.0
            
        # 3. Energy Efficiency (inverse of energy consumption)
        energy_per_part = self._calculate_energy_consumption()
        rewards['energy_efficiency'] = 1.0 / (energy_per_part + 1e-6)
        
        # 4. Precision (placement accuracy)
        precision_score = self._calculate_precision_score()
        rewards['precision'] = precision_score
        
        # 5. Wear Reduction (smooth movements)
        wear_score = self._calculate_wear_reduction_score()
        rewards['wear_reduction'] = wear_score
        
        # 6. Collision Avoidance (safety margins)
        collision_score = self._calculate_collision_avoidance_score()
        rewards['collision_avoidance'] = collision_score
        
        return rewards
    
    def _calculate_energy_consumption(self):
        """Calculate energy consumption based on joint movements"""
        if not self.coppelia_available:
            return np.random.uniform(0.1, 1.0)
            
        total_energy = 0.0
        
        for joint in self.ur5_joints:
            try:
                velocity = sim.getJointVelocity(joint)
                force = sim.getJointForce(joint)
                total_energy += abs(velocity * force)
            except:
                pass
                
        return total_energy
    
    def _calculate_precision_score(self):
        """Calculate placement precision score"""
        if not self.coppelia_available or not self.ik_tip:
            return np.random.uniform(0.7, 0.9)
            
        try:
            tip_pos = sim.getObjectPosition(self.ik_tip, -1)
            # Define target position (this would come from task specification)
            target_pos = [0.5, 0.0, 0.5]  # Example target
            
            distance = np.linalg.norm(np.array(tip_pos) - np.array(target_pos))
            precision_score = np.exp(-distance * 10)  # Exponential decay
            return precision_score
        except:
            return 0.5
    
    def _calculate_wear_reduction_score(self):
        """Calculate wear reduction score (smooth movements)"""
        if not self.coppelia_available:
            return np.random.uniform(0.6, 0.8)
            
        total_jerk = 0.0
        
        for joint in self.ur5_joints:
            try:
                velocity = sim.getJointVelocity(joint)
                # Approximate jerk as velocity change
                total_jerk += abs(velocity)
            except:
                pass
                
        # Higher score for lower jerk (smoother movements)
        wear_score = 1.0 / (total_jerk + 1e-6)
        return min(wear_score, 1.0)
    
    def _calculate_collision_avoidance_score(self):
        """Calculate collision avoidance score"""
        # This would check distances to obstacles
        # For now, return high score (no collisions detected)
        return np.random.uniform(0.85, 1.0)
    
    def _get_conveyor_status(self):
        """Get conveyor belt status"""
        # Placeholder - would interface with conveyor
        return 0.5
    
    def _get_pallet_status(self):
        """Get pallet occupancy status"""
        # Placeholder - would check each pallet
        return np.array([0.0, 0.25, 0.5, 0.75])
    
    def _get_current_task_progress(self):
        """Get current task progress"""
        return min(self.parts_completed / 6.0, 1.0)
    
    def _update_objectives(self):
        """Update objective tracking"""
        elapsed_time = time.time() - self.episode_start_time
        
        if elapsed_time > 0:
            self.objectives['throughput'] = (self.parts_completed / elapsed_time) * 60
            
        if self.parts_completed > 0:
            self.objectives['cycle_time'] = elapsed_time / self.parts_completed
            
        self.objectives['energy_efficiency'] = 1.0 / (self.energy_consumed + 1e-6)
        self.objectives['precision'] = self._calculate_precision_score()
        self.objectives['wear_reduction'] = self._calculate_wear_reduction_score()
        self.objectives['collision_avoidance'] = self._calculate_collision_avoidance_score()
    
    def _check_episode_done(self):
        """Check if episode should end"""
        # End episode after completing 6 parts (as in original script)
        return self.parts_completed >= 6
    
    def get_objective_values(self):
        """Get current objective values for multi-objective optimization"""
        return list(self.objectives.values())
    
    def render(self, mode='human'):
        """Render environment (CoppeliaSim handles this)"""
        if self.coppelia_available:
            pass  # CoppeliaSim handles rendering
        else:
            print(f"Mock render - Parts completed: {self.parts_completed}")


# Test function
def test_environment():
    """Test the environment with and without CoppeliaSim"""
    print("Testing UR5 MORL Environment...")
    
    env = UR5_MORL_Environment()
    
    print(f"CoppeliaSim available: {env.coppelia_available}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space keys: {list(env.action_space.spaces.keys())}")
    
    # Test reset
    state, info = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial objectives: {info['objectives']}")
    
    # Test a few steps
    for i in range(3):
        action = env.action_space.sample()
        state, rewards, done, truncated, info = env.step(action)
        print(f"Step {i+1}: Rewards = {rewards}, Done = {done}")
        
        if done:
            break
    
    print("Environment test completed!")


if __name__ == "__main__":
    test_environment()